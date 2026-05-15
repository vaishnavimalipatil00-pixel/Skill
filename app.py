import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PragyanAI – Program Intelligence Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL STYLE
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg,#0f0c29,#302b63,#24243e);
    color: #fff;
}
section[data-testid="stSidebar"] * { color: #e0d7ff !important; }
section[data-testid="stSidebar"] .stFileUploader label { color:#fff !important; }

/* hero banner */
.hero {
    background: linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
    border-radius:16px; padding:36px 40px 28px;
    margin-bottom:28px; border:1px solid #e94560;
}
.hero h1 { font-family:'Syne',sans-serif; font-size:2.6rem;
            color:#fff; margin:0 0 6px; }
.hero p  { color:#a0b4d6; font-size:1.05rem; margin:0; }
.hero .tag { display:inline-block; background:#e94560; color:#fff;
             font-size:.75rem; font-weight:700; padding:3px 10px;
             border-radius:20px; margin-bottom:12px; letter-spacing:.08em; }

/* KPI cards */
.kpi-row { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }
.kpi {
    flex:1; min-width:140px;
    background:linear-gradient(135deg,#1e1e3f,#2a2a5e);
    border:1px solid #3a3a7a; border-radius:14px;
    padding:20px 18px; text-align:center;
}
.kpi .val { font-family:'Syne',sans-serif; font-size:2rem;
            color:#e94560; line-height:1; }
.kpi .lbl { color:#a0b4d6; font-size:.8rem; margin-top:6px; }

/* section header */
.sec-hdr { font-family:'Syne',sans-serif; font-size:1.3rem;
           color:#e94560; border-left:4px solid #e94560;
           padding-left:12px; margin:30px 0 14px; }

/* insight box */
.insight {
    background:#1a1a3e; border-left:4px solid #f5a623;
    border-radius:8px; padding:14px 18px; margin:10px 0;
    color:#e8e0ff; font-size:.92rem; line-height:1.6;
}

/* badge */
.badge-hot  { background:#e94560; color:#fff; padding:2px 9px;
              border-radius:12px; font-size:.75rem; font-weight:700; }
.badge-mid  { background:#f5a623; color:#1a1a2e; padding:2px 9px;
              border-radius:12px; font-size:.75rem; font-weight:700; }
.badge-low  { background:#4a4a6a; color:#ccc; padding:2px 9px;
              border-radius:12px; font-size:.75rem; font-weight:700; }

/* table */
.styled-table { width:100%; border-collapse:collapse; font-size:.87rem; }
.styled-table th { background:#e94560; color:#fff; padding:9px 12px;
                   text-align:left; font-weight:600; }
.styled-table td { padding:8px 12px; border-bottom:1px solid #2a2a4a;
                   color:#c8d6ea; }
.styled-table tr:nth-child(even) td { background:#1a1a30; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
DARK_BG   = "#0f0c29"
CARD_BG   = "#1a1a3e"
ACCENT    = "#e94560"
ACCENT2   = "#f5a623"
ACCENT3   = "#00d2ff"
TEXT_CLR  = "#c8d6ea"
GRID_CLR  = "#2a2a5a"

def dark_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.6)
    return fig, ax

def dark_figs(rows, cols, w=14, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    axlist = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in axlist:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_CLR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(TEXT_CLR)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_CLR)
        ax.grid(color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.5)
    return fig, axlist

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
PALETTE = [ACCENT, ACCENT2, ACCENT3, "#a78bfa", "#34d399", "#f472b6"]

def duration_band(m):
    if m < 3:   return "<3 mo"
    elif m <= 6: return "3–6 mo"
    elif m <= 12: return "6–12 mo"
    else:         return ">12 mo"

def projects_band(p):
    if p < 2:   return "<2"
    elif p <= 5: return "3–5"
    else:        return "6+"

def tools_band(t):
    if t < 3:   return "<3"
    elif t <= 8: return "3–8"
    else:        return "8+"

def subjects_band(s):
    if s < 5:   return "<5"
    elif s <= 10: return "5–10"
    else:         return ">10"

def cost_band(c):
    if c < 50000:    return "<50K"
    elif c <= 200000: return "50K–2L"
    else:             return ">2L"

def bar_chart(ax, cats, vals, title, ylabel, colors=None):
    if colors is None:
        colors = PALETTE[:len(cats)]
    bars = ax.bar(cats, vals, color=colors, edgecolor=DARK_BG, linewidth=0.8, zorder=3)
    ax.set_title(title, fontsize=11, pad=10, color=TEXT_CLR, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(vals)*0.01,
                f"{b.get_height():.2f}", ha="center", va="bottom",
                fontsize=8, color=TEXT_CLR)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=9)

# ─────────────────────────────────────────────
#  SIDEBAR – FILE UPLOAD ONLY
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 PragyanAI")
    st.markdown("**Program Intelligence & ROI Engine**")
    st.markdown("---")
    uploaded = st.file_uploader(
        "📂 Upload your dataset CSV",
        type=["csv"],
        help="Upload the student_skill_program_Analysis CSV file"
    )
    st.markdown("---")
    st.markdown("### 🔍 Filters")
    filters_ready = False

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="tag">🔥 INTELLIGENCE ENGINE</span>
  <h1>PragyanAI Program Intelligence & ROI</h1>
  <p>Multi-dimensional analysis of 200K student-program records — faculty, curriculum, projects,
     tools, duration and their true impact on skill outcomes & placements.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
if uploaded is None:
    st.info("👈  Upload your CSV file from the sidebar to begin analysis.")
    st.markdown("""
    <div class="insight">
    <b>Expected columns include:</b><br>
    Program_ID, Program_Name, Program_Type, Mode, Duration_Months, Total_Hours,
    Subjects_Count, Projects_Count, Tools_Count, Cost,
    Faculty_Experience, Faculty_Degree, Industry_Experience, Company_Background,
    Student_ID, College_Tier, Department, CGPA,
    Skill_Level_Before, Skill_Level_After, Skill_Improvement, Status,
    Placement_Status, Salary_LPA, ROI
    </div>
    """, unsafe_allow_html=True)
    st.stop()

@st.cache_data(show_spinner="Loading dataset …")
def load(f):
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()
    return df

df_raw = load(uploaded)
df = df_raw.copy()

# ── normalise column names (flexible matching) ──────────────────────────────
col_map = {}
for c in df.columns:
    lc = c.lower().replace(" ", "_")
    col_map[c] = lc
df.rename(columns=col_map, inplace=True)

def find_col(df, *candidates):
    for c in candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None

C_PROG_TYPE  = find_col(df, "program_type", "prog_type", "type")
C_MODE       = find_col(df, "mode")
C_DURATION   = find_col(df, "duration_months", "duration")
C_SUBJECTS   = find_col(df, "subjects_count", "subjects")
C_PROJECTS   = find_col(df, "projects_count", "projects")
C_TOOLS      = find_col(df, "tools_count", "tools")
C_COST       = find_col(df, "cost")
C_FAC_EXP    = find_col(df, "faculty_experience", "fac_exp")
C_FAC_DEG    = find_col(df, "faculty_degree", "degree")
C_IND_EXP    = find_col(df, "industry_experience", "ind_exp")
C_COMP_BG    = find_col(df, "company_background", "company_bg")
C_TIER       = find_col(df, "college_tier", "tier")
C_DEPT       = find_col(df, "department", "dept")
C_CGPA       = find_col(df, "cgpa")
C_SKILL_B    = find_col(df, "skill_level_before", "skill_before", "before")
C_SKILL_A    = find_col(df, "skill_level_after", "skill_after", "after")
C_SKILL_IMP  = find_col(df, "skill_improvement", "improvement")
C_STATUS     = find_col(df, "placement_status", "status")
C_SALARY     = find_col(df, "salary_lpa", "salary")
C_ROI        = find_col(df, "roi")
C_PROG_NAME  = find_col(df, "program_name", "prog_name")

# numeric coerce
for col in [C_DURATION, C_SUBJECTS, C_PROJECTS, C_TOOLS, C_COST,
            C_FAC_EXP, C_CGPA, C_SKILL_B, C_SKILL_A, C_SKILL_IMP, C_SALARY, C_ROI]:
    if col:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# derived columns (if missing)
if C_SKILL_IMP is None and C_SKILL_B and C_SKILL_A:
    df["skill_improvement"] = df[C_SKILL_A] - df[C_SKILL_B]
    C_SKILL_IMP = "skill_improvement"

if C_ROI is None and C_SALARY and C_COST:
    df["roi"] = df[C_SALARY] / (df[C_COST] / 100000)
    C_ROI = "roi"

# placement flag
if C_STATUS:
    df["placed_flag"] = df[C_STATUS].astype(str).str.lower().str.contains("placed|yes|1").astype(int)
else:
    df["placed_flag"] = 0

# ── sidebar filters (dynamic) ───────────────────────────────────────────────
with st.sidebar:
    if C_PROG_TYPE:
        types = ["All"] + sorted(df[C_PROG_TYPE].dropna().unique().tolist())
        sel_type = st.selectbox("Program Type", types)
    else:
        sel_type = "All"

    if C_MODE:
        modes = ["All"] + sorted(df[C_MODE].dropna().unique().tolist())
        sel_mode = st.selectbox("Mode", modes)
    else:
        sel_mode = "All"

    if C_DURATION:
        dur_min, dur_max = int(df[C_DURATION].min()), int(df[C_DURATION].max())
        sel_dur = st.slider("Duration (months)", dur_min, dur_max, (dur_min, dur_max))
    else:
        sel_dur = None

dff = df.copy()
if sel_type != "All" and C_PROG_TYPE:
    dff = dff[dff[C_PROG_TYPE] == sel_type]
if sel_mode != "All" and C_MODE:
    dff = dff[dff[C_MODE] == sel_mode]
if sel_dur and C_DURATION:
    dff = dff[(dff[C_DURATION] >= sel_dur[0]) & (dff[C_DURATION] <= sel_dur[1])]

# ─────────────────────────────────────────────
#  KPI STRIP
# ─────────────────────────────────────────────
total_records   = len(dff)
avg_roi         = dff[C_ROI].mean()      if C_ROI      else np.nan
avg_salary      = dff[C_SALARY].mean()   if C_SALARY   else np.nan
avg_skill_imp   = dff[C_SKILL_IMP].mean()if C_SKILL_IMP else np.nan
placement_rate  = dff["placed_flag"].mean() * 100
avg_cost        = dff[C_COST].mean()     if C_COST     else np.nan

k1,k2,k3,k4,k5,k6 = st.columns(6)
def kpi_md(val, lbl, fmt="{:.2f}"):
    v = fmt.format(val) if not np.isnan(val) else "N/A"
    return f'<div class="kpi"><div class="val">{v}</div><div class="lbl">{lbl}</div></div>'

st.markdown(f"""
<div class="kpi-row">
  {kpi_md(total_records,  "Total Records",    "{:,}")}
  {kpi_md(avg_roi,        "Avg ROI",          "{:.2f}x")}
  {kpi_md(avg_salary,     "Avg Salary (LPA)", "₹{:.1f}")}
  {kpi_md(avg_skill_imp,  "Avg Skill Gain",   "{:.2f}")}
  {kpi_md(placement_rate, "Placement Rate",   "{:.1f}%")}
  {kpi_md(avg_cost/1000 if avg_cost else np.nan, "Avg Cost (K₹)", "₹{:.0f}K")}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "⏱ Duration Analysis",
    "🛠 Projects & Tools",
    "📚 Subjects & Cost",
    "👨‍🏫 Faculty Impact",
    "🎓 Mode & Program Type",
    "💡 Skill Improvement",
    "💰 ROI & Value",
    "🏆 Segmentation",
    "🔑 Key Insights",
    "📋 Data Explorer"
])

# ══════════════════════════════════════════════
#  TAB 0 – OVERVIEW
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="sec-hdr">Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # Program Type distribution
    with c1:
        if C_PROG_TYPE:
            vc = dff[C_PROG_TYPE].value_counts()
            fig, ax = dark_fig(6, 4)
            wedges, texts, autotexts = ax.pie(
                vc.values, labels=vc.index,
                colors=PALETTE[:len(vc)],
                autopct="%1.1f%%", startangle=140,
                textprops={"color": TEXT_CLR, "fontsize": 9},
                wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.5}
            )
            for at in autotexts: at.set_color("#fff")
            ax.set_title("Program Type Distribution", color=TEXT_CLR, fontsize=11, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    with c2:
        if C_MODE:
            vc2 = dff[C_MODE].value_counts()
            fig, ax = dark_fig(6, 4)
            bars = ax.barh(vc2.index, vc2.values, color=PALETTE[:len(vc2)],
                           edgecolor=DARK_BG, linewidth=0.8)
            ax.set_title("Mode Distribution", color=TEXT_CLR, fontsize=11, fontweight="bold")
            ax.set_xlabel("Count", color=TEXT_CLR)
            for b in bars:
                ax.text(b.get_width() + vc2.max()*0.01, b.get_y() + b.get_height()/2,
                        f"{int(b.get_width()):,}", va="center", color=TEXT_CLR, fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Correlation heatmap
    st.markdown('<div class="sec-hdr">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = [c for c in [C_DURATION, C_SUBJECTS, C_PROJECTS, C_TOOLS,
                             C_COST, C_FAC_EXP, C_CGPA, C_SKILL_IMP,
                             C_SALARY, C_ROI] if c]
    if len(num_cols) >= 4:
        corr = dff[num_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(CARD_BG)
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(num_cols)))
        ax.set_yticks(range(len(num_cols)))
        short = [c.replace("_", " ").replace("count","cnt").title() for c in num_cols]
        ax.set_xticklabels(short, rotation=35, ha="right", color=TEXT_CLR, fontsize=8)
        ax.set_yticklabels(short, color=TEXT_CLR, fontsize=8)
        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                ax.text(j, i, f"{corr.values[i,j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if abs(corr.values[i,j]) > 0.5 else TEXT_CLR)
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title("Feature Correlation Heatmap", color=TEXT_CLR, fontsize=12, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════
#  TAB 1 – DURATION
# ══════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="sec-hdr">Duration vs Skill Improvement & Placement</div>', unsafe_allow_html=True)
    if C_DURATION and C_SKILL_IMP:
        dff["dur_band"] = dff[C_DURATION].apply(duration_band)
        band_order = ["<3 mo", "3–6 mo", "6–12 mo", ">12 mo"]

        grp = dff.groupby("dur_band").agg(
            avg_skill_imp=(C_SKILL_IMP, "mean"),
            avg_placement=("placed_flag", "mean"),
            count=(C_SKILL_IMP, "count")
        ).reindex(band_order).dropna()

        fig, axs = dark_figs(1, 2, w=13, h=4)
        bar_chart(axs[0], grp.index, grp["avg_skill_imp"],
                  "Avg Skill Improvement by Duration", "Avg Skill Gain",
                  colors=[ACCENT3 if b == "6–12 mo" else PALETTE[i % len(PALETTE)] for i, b in enumerate(grp.index)])
        bar_chart(axs[1], grp.index, grp["avg_placement"]*100,
                  "Placement Rate (%) by Duration", "Placement %",
                  colors=[ACCENT if b == "6–12 mo" else PALETTE[i % len(PALETTE)] for i, b in enumerate(grp.index)])
        fig.tight_layout(pad=2)
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="insight">
        🔥 <b>Sweet Spot = 6–12 months</b><br>
        Programs in this range show the highest skill improvement and placement rates.
        <br>• &lt;3 months → insufficient depth
        <br>• &gt;12 months → diminishing returns (plateau effect)
        </div>""", unsafe_allow_html=True)

        # scatter
        fig, ax = dark_fig(11, 4)
        sc = ax.scatter(dff[C_DURATION], dff[C_SKILL_IMP],
                        c=dff["placed_flag"], cmap="RdYlGn",
                        alpha=0.35, s=12, edgecolors="none")
        ax.set_xlabel("Duration (months)", color=TEXT_CLR)
        ax.set_ylabel("Skill Improvement", color=TEXT_CLR)
        ax.set_title("Duration vs Skill Improvement (coloured by Placement)", color=TEXT_CLR, fontweight="bold")
        plt.colorbar(sc, ax=ax, label="Placed (1=Yes)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("Duration or Skill Improvement column not found in dataset.")

# ══════════════════════════════════════════════
#  TAB 2 – PROJECTS & TOOLS
# ══════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="sec-hdr">Projects & Tools vs Outcomes</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if C_PROJECTS:
            dff["proj_band"] = dff[C_PROJECTS].apply(projects_band)
            pord = ["<2", "3–5", "6+"]
            pg = dff.groupby("proj_band").agg(
                placement=("placed_flag", "mean"),
                skill_gain=(C_SKILL_IMP, "mean") if C_SKILL_IMP else ("placed_flag", "mean")
            ).reindex(pord).dropna()
            fig, ax = dark_fig(5, 3.5)
            bar_chart(ax, pg.index, pg["placement"]*100,
                      "Projects → Placement Rate (%)", "Placement %",
                      colors=[PALETTE[i] for i in range(len(pg))])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    with c2:
        if C_TOOLS:
            dff["tools_band"] = dff[C_TOOLS].apply(tools_band)
            tord = ["<3", "3–8", "8+"]
            tg = dff.groupby("tools_band").agg(
                roi=(C_ROI, "mean") if C_ROI else ("placed_flag","mean"),
                placement=("placed_flag","mean")
            ).reindex(tord).dropna()
            fig, ax = dark_fig(5, 3.5)
            col_key = "roi" if C_ROI else "placement"
            lbl = "Avg ROI" if C_ROI else "Placement %"
            vals = tg[col_key] if C_ROI else tg[col_key]*100
            bar_chart(ax, tg.index, vals,
                      "Tools Count → ROI", lbl,
                      colors=[PALETTE[i] for i in range(len(tg))])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    if C_PROJECTS and C_COST:
        st.markdown('<div class="sec-hdr">Cost per Project (Efficiency)</div>', unsafe_allow_html=True)
        dff["cost_per_project"] = dff[C_COST] / (dff[C_PROJECTS].replace(0, np.nan))
        fig, ax = dark_fig(11, 3.5)
        ax.hist(dff["cost_per_project"].dropna(), bins=60,
                color=ACCENT2, edgecolor=DARK_BG, linewidth=0.4, alpha=0.9)
        ax.axvline(dff["cost_per_project"].median(), color=ACCENT3,
                   linestyle="--", linewidth=1.5, label="Median")
        ax.set_xlabel("Cost per Project (₹)", color=TEXT_CLR)
        ax.set_ylabel("Count", color=TEXT_CLR)
        ax.set_title("Distribution: Cost per Project", color=TEXT_CLR, fontweight="bold")
        ax.legend(facecolor=CARD_BG, labelcolor=TEXT_CLR)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("""
        <div class="insight">
        ⚠️ <b>High cost + low projects = worst programs.</b>
        A low cost-per-project ratio signals efficient, hands-on curriculum design.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 3 – SUBJECTS & COST
# ══════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="sec-hdr">Subjects Depth & Cost vs ROI</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if C_SUBJECTS and C_SKILL_IMP:
            dff["subj_band"] = dff[C_SUBJECTS].apply(subjects_band)
            sord = ["<5","5–10",">10"]
            sg = dff.groupby("subj_band")[C_SKILL_IMP].mean().reindex(sord).dropna()
            fig, ax = dark_fig(5, 3.5)
            bar_chart(ax, sg.index, sg.values,
                      "Subjects Count → Skill Improvement", "Avg Skill Gain",
                      colors=PALETTE[:len(sg)])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("""
            <div class="insight">
            📚 More subjects ≠ better learning. 5–10 subjects is the sweet spot.
            Overloading students reduces depth per subject.
            </div>""", unsafe_allow_html=True)

    with c2:
        if C_COST and C_ROI:
            dff["cost_band"] = dff[C_COST].apply(cost_band)
            cord = ["<50K","50K–2L",">2L"]
            cg = dff.groupby("cost_band")[C_ROI].mean().reindex(cord).dropna()
            fig, ax = dark_fig(5, 3.5)
            bar_chart(ax, cg.index, cg.values,
                      "Cost Band → Avg ROI", "Avg ROI",
                      colors=[PALETTE[i] for i in range(len(cg))])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("""
            <div class="insight">
            💰 Mid-range programs (₹50K–2L) give the <b>best ROI</b>.
            High-cost programs (>₹3L) carry significant financial risk.
            </div>""", unsafe_allow_html=True)

    if C_COST and C_PROJECTS:
        st.markdown('<div class="sec-hdr">Cost vs Projects Scatter</div>', unsafe_allow_html=True)
        fig, ax = dark_fig(11, 4)
        sc = ax.scatter(dff[C_COST]/1000, dff[C_PROJECTS],
                        c=dff["placed_flag"], cmap="RdYlGn",
                        alpha=0.3, s=10, edgecolors="none")
        ax.set_xlabel("Cost (₹K)", color=TEXT_CLR)
        ax.set_ylabel("Projects Count", color=TEXT_CLR)
        ax.set_title("Cost vs Projects (coloured by Placement)", color=TEXT_CLR, fontweight="bold")
        plt.colorbar(sc, ax=ax, label="Placed")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════
#  TAB 4 – FACULTY
# ══════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="sec-hdr">Faculty Impact Analysis</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if C_FAC_EXP and C_SKILL_IMP:
            dff["fac_exp_band"] = pd.cut(dff[C_FAC_EXP], bins=[0,4,8,20],
                                          labels=["0–4 yrs","5–8 yrs",">8 yrs"])
            fg = dff.groupby("fac_exp_band")[C_SKILL_IMP].mean().dropna()
            fig, ax = dark_fig(5, 3.5)
            bar_chart(ax, fg.index.astype(str), fg.values,
                      "Faculty Experience → Skill Gain", "Avg Skill Gain",
                      colors=PALETTE[:len(fg)])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    with c2:
        if C_IND_EXP and C_SALARY:
            ig = dff.groupby(C_IND_EXP)[C_SALARY].mean().dropna()
            fig, ax = dark_fig(5, 3.5)
            bar_chart(ax, ig.index.astype(str), ig.values,
                      "Industry Experience → Avg Salary (LPA)", "Salary LPA",
                      colors=[ACCENT, ACCENT3])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    if C_COMP_BG and C_ROI:
        st.markdown('<div class="sec-hdr">Company Background vs ROI</div>', unsafe_allow_html=True)
        cg = dff.groupby(C_COMP_BG)[C_ROI].mean().dropna().sort_values(ascending=False)
        fig, ax = dark_fig(8, 3.5)
        bar_chart(ax, cg.index.astype(str), cg.values,
                  "Faculty Company Background → Avg ROI", "Avg ROI",
                  colors=PALETTE[:len(cg)])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    if C_FAC_DEG and C_SKILL_IMP:
        dg = dff.groupby(C_FAC_DEG)[C_SKILL_IMP].mean().dropna().sort_values(ascending=False)
        fig, ax = dark_fig(8, 3.5)
        bar_chart(ax, dg.index.astype(str), dg.values,
                  "Faculty Degree → Avg Skill Improvement", "Avg Skill Gain",
                  colors=PALETTE[:len(dg)])
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown("""
        <div class="insight">
        👨‍🏫 Faculty with <b>industry experience</b> produce higher-earning graduates regardless of degree.
        PhD alone (without real-world background) shows moderate impact only.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 5 – MODE & PROGRAM TYPE
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="sec-hdr">Learning Mode & Program Type Impact</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if C_MODE and C_SKILL_IMP:
            mg = dff.groupby(C_MODE).agg(
                skill=(C_SKILL_IMP,"mean"),
                placement=("placed_flag","mean")
            ).dropna()
            fig, ax = dark_fig(5, 4)
            x = np.arange(len(mg))
            w = 0.35
            b1 = ax.bar(x-w/2, mg["skill"], w, label="Skill Gain", color=ACCENT, edgecolor=DARK_BG)
            b2 = ax.bar(x+w/2, mg["placement"]*3, w, label="Placement (×3)", color=ACCENT3, edgecolor=DARK_BG)
            ax.set_xticks(x); ax.set_xticklabels(mg.index, color=TEXT_CLR)
            ax.set_title("Mode → Skill & Placement", color=TEXT_CLR, fontweight="bold")
            ax.legend(facecolor=CARD_BG, labelcolor=TEXT_CLR, fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    with c2:
        if C_PROG_TYPE and C_ROI:
            ptg = dff.groupby(C_PROG_TYPE)[C_ROI].mean().dropna().sort_values(ascending=False)
            fig, ax = dark_fig(6, 4)
            bar_chart(ax, ptg.index.astype(str), ptg.values,
                      "Program Type → Avg ROI", "Avg ROI",
                      colors=PALETTE[:len(ptg)])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("""
    <div class="insight">
    🌐 <b>Hybrid = Best Learning Model</b> — combines flexibility of online with depth of offline engagement.<br>
    🤖 <b>GenAI programs</b> show the highest ROI among all program types — reflecting strong market demand.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 6 – SKILL IMPROVEMENT
# ══════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="sec-hdr">Skill Improvement Deep Dive</div>', unsafe_allow_html=True)

    if C_SKILL_B and C_SKILL_A:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = dark_fig(5, 3.5)
            ax.hist(dff[C_SKILL_B].dropna(), bins=20, alpha=0.7, color=ACCENT2, label="Before", edgecolor=DARK_BG)
            ax.hist(dff[C_SKILL_A].dropna(), bins=20, alpha=0.7, color=ACCENT3, label="After",  edgecolor=DARK_BG)
            ax.set_title("Skill Level Before vs After", color=TEXT_CLR, fontweight="bold")
            ax.set_xlabel("Skill Level (0–3)", color=TEXT_CLR)
            ax.legend(facecolor=CARD_BG, labelcolor=TEXT_CLR)
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            if C_SKILL_IMP:
                imp_grp = dff.groupby(C_SKILL_IMP)["placed_flag"].mean().dropna()
                fig, ax = dark_fig(5, 3.5)
                bar_chart(ax, imp_grp.index.astype(str), imp_grp.values*100,
                          "Skill Improvement → Placement Rate (%)", "Placement %",
                          colors=PALETTE[:len(imp_grp)])
                fig.tight_layout(); st.pyplot(fig); plt.close()

    if C_TIER and C_SKILL_IMP:
        st.markdown('<div class="sec-hdr">College Tier vs Skill Gain</div>', unsafe_allow_html=True)
        tg2 = dff.groupby(C_TIER)[C_SKILL_IMP].mean().dropna().sort_values()
        fig, ax = dark_fig(10, 3.5)
        bar_chart(ax, tg2.index.astype(str), tg2.values,
                  "College Tier → Avg Skill Improvement", "Avg Skill Gain",
                  colors=PALETTE[:len(tg2)])
        fig.tight_layout(); st.pyplot(fig); plt.close()

    if C_CGPA and C_SKILL_IMP:
        fig, ax = dark_fig(11, 3.5)
        ax.scatter(dff[C_CGPA], dff[C_SKILL_IMP],
                   alpha=0.2, s=8, color=ACCENT2, edgecolors="none")
        z = np.polyfit(dff[[C_CGPA, C_SKILL_IMP]].dropna()[C_CGPA],
                       dff[[C_CGPA, C_SKILL_IMP]].dropna()[C_SKILL_IMP], 1)
        p = np.poly1d(z)
        xs = np.linspace(dff[C_CGPA].min(), dff[C_CGPA].max(), 100)
        ax.plot(xs, p(xs), color=ACCENT, linewidth=2, label="Trend")
        ax.set_xlabel("CGPA", color=TEXT_CLR)
        ax.set_ylabel("Skill Improvement", color=TEXT_CLR)
        ax.set_title("CGPA vs Skill Improvement", color=TEXT_CLR, fontweight="bold")
        ax.legend(facecolor=CARD_BG, labelcolor=TEXT_CLR)
        fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════
#  TAB 7 – ROI & VALUE
# ══════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="sec-hdr">ROI, Salary & Value Metrics</div>', unsafe_allow_html=True)

    # Advanced metrics
    if C_PROJECTS and C_TOOLS and C_DURATION:
        dff["learning_density"] = (dff[C_PROJECTS] + dff[C_TOOLS]) / dff[C_DURATION].replace(0, np.nan)
    if C_PROJECTS and C_TOOLS and C_FAC_EXP and C_COST:
        dff["value_score"] = (dff[C_PROJECTS] + dff[C_TOOLS] + dff[C_FAC_EXP]) / (dff[C_COST]/100000).replace(0, np.nan)
    if C_SKILL_IMP and C_ROI:
        dff["effectiveness"] = dff[C_SKILL_IMP] + dff["placed_flag"] + dff[C_ROI]

    c1, c2, c3 = st.columns(3)
    def metric_hist(col, label, color, container):
        with container:
            if col in dff.columns:
                fig, ax = dark_fig(4, 3)
                ax.hist(dff[col].dropna(), bins=40, color=color,
                        edgecolor=DARK_BG, linewidth=0.3, alpha=0.9)
                ax.axvline(dff[col].median(), color="#fff",
                           linestyle="--", linewidth=1.2, label="Median")
                ax.set_title(label, color=TEXT_CLR, fontsize=10, fontweight="bold")
                ax.legend(facecolor=CARD_BG, labelcolor=TEXT_CLR, fontsize=8)
                fig.tight_layout(); st.pyplot(fig); plt.close()

    metric_hist("learning_density",  "Learning Density",   ACCENT,  c1)
    metric_hist("value_score",       "Value Score",        ACCENT2, c2)
    metric_hist("effectiveness",     "Effectiveness Score",ACCENT3, c3)

    if C_SALARY and C_COST:
        st.markdown('<div class="sec-hdr">Salary LPA Distribution by Program Type</div>', unsafe_allow_html=True)
        if C_PROG_TYPE:
            types_list = dff[C_PROG_TYPE].dropna().unique()
            fig, ax = dark_fig(11, 4)
            data_to_plot = [dff[dff[C_PROG_TYPE]==t][C_SALARY].dropna().values for t in types_list]
            bp = ax.boxplot(data_to_plot, patch_artist=True,
                            medianprops=dict(color="#fff", linewidth=2),
                            whiskerprops=dict(color=TEXT_CLR),
                            capprops=dict(color=TEXT_CLR),
                            flierprops=dict(markerfacecolor=ACCENT, markersize=3, alpha=0.4))
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            ax.set_xticklabels(types_list, color=TEXT_CLR, rotation=15, ha="right")
            ax.set_ylabel("Salary LPA", color=TEXT_CLR)
            ax.set_title("Salary LPA by Program Type", color=TEXT_CLR, fontweight="bold")
            fig.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════
#  TAB 8 – SEGMENTATION
# ══════════════════════════════════════════════
with tabs[8]:
    st.markdown('<div class="sec-hdr">Program Segmentation</div>', unsafe_allow_html=True)

    seg_cols_present = all(c for c in [C_DURATION, C_PROJECTS, C_COST, C_ROI])
    if seg_cols_present:
        def segment(row):
            dur = row[C_DURATION] if C_DURATION else 0
            proj = row[C_PROJECTS] if C_PROJECTS else 0
            cost = row[C_COST] if C_COST else 0
            roi = row[C_ROI] if C_ROI else 0
            if 6 <= dur <= 12 and proj >= 5 and roi >= 2:
                return "🔥 High Value"
            elif cost > 200000 and proj < 3:
                return "⚠️ Overpriced"
            elif dur < 3 and (C_TOOLS and row.get(C_TOOLS, 0) < 3):
                return "📉 Low Depth"
            elif (C_SUBJECTS and row.get(C_SUBJECTS, 0) > 10) and roi < 1:
                return "🔴 Heavy Inefficient"
            else:
                return "🟡 Average"

        dff["segment"] = dff.apply(segment, axis=1)
        seg_counts = dff["segment"].value_counts()

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = dark_fig(5, 4)
            colors_seg = [ACCENT3, ACCENT, "#4a4a8a", "#e94560", ACCENT2]
            ax.pie(seg_counts.values, labels=seg_counts.index,
                   colors=colors_seg[:len(seg_counts)],
                   autopct="%1.1f%%", startangle=140,
                   textprops={"color": TEXT_CLR, "fontsize": 9},
                   wedgeprops={"edgecolor": DARK_BG})
            ax.set_title("Program Segments", color=TEXT_CLR, fontweight="bold")
            fig.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            st.markdown("#### Segment Summary")
            seg_summary = dff.groupby("segment").agg(
                Count=("segment","count"),
                Avg_ROI=(C_ROI, "mean"),
                Avg_Salary=(C_SALARY, "mean") if C_SALARY else (C_ROI,"count"),
                Placement_Rate=("placed_flag","mean")
            ).round(2)
            seg_summary["Placement_Rate"] = (seg_summary["Placement_Rate"]*100).round(1).astype(str) + "%"
            st.dataframe(seg_summary.style.background_gradient(cmap="RdYlGn", subset=["Avg_ROI"]),
                         use_container_width=True)

        st.markdown("""
        <div class="insight">
        🔥 <b>High Value</b>: 6–12 months | 5+ projects | Strong ROI → Best programs to enroll<br>
        ⚠️ <b>Overpriced</b>: High cost, low projects → Avoid at all costs<br>
        📉 <b>Low Depth</b>: Short duration, few tools → Low real-world impact<br>
        🔴 <b>Heavy Inefficient</b>: Too many subjects, poor outcomes → Poor curriculum design
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("Need Duration, Projects, Cost, ROI columns for segmentation.")

# ══════════════════════════════════════════════
#  TAB 9 – KEY INSIGHTS
# ══════════════════════════════════════════════
with tabs[9]:
    st.markdown('<div class="sec-hdr">🔑 Key Insights & Strategic Intelligence</div>', unsafe_allow_html=True)

    insights = [
        ("🎯", "Insight 1", "Students focus on placement, not learning depth",
         "Programs must re-orient marketing toward skill depth, projects, and real-world tools — not just placement numbers."),
        ("🛠", "Insight 2", "Projects + Tools > Subjects",
         "Hands-on projects and tool exposure are the strongest predictors of both skill gain and salary outcomes."),
        ("👨‍🏫", "Insight 3", "Faculty industry experience is critical",
         "Faculty with real-world company experience produce graduates who earn more and place faster."),
        ("⏱", "Insight 4", "Duration sweet spot = 6–12 months",
         "Programs shorter than 3 months lack depth. Programs longer than 12 months hit a learning plateau."),
        ("🌐", "Insight 5", "Hybrid learning is most effective",
         "Hybrid programs combining online flexibility with offline depth outperform pure-online and pure-offline models."),
        ("💰", "Insight 6", "Mid-range cost (₹50K–2L) gives best ROI",
         "Expensive programs don't guarantee better outcomes. Over-investment in course fees reduces net ROI significantly."),
    ]

    for icon, label, title, desc in insights:
        st.markdown(f"""
        <div style="background:#1a1a3e;border:1px solid #3a3a7a;border-radius:12px;
                    padding:18px 22px;margin:10px 0;display:flex;gap:18px;align-items:flex-start;">
          <div style="font-size:2rem;line-height:1;">{icon}</div>
          <div>
            <div style="color:#a0b4d6;font-size:.75rem;font-weight:700;
                        letter-spacing:.1em;text-transform:uppercase;">{label}</div>
            <div style="color:#fff;font-size:1.05rem;font-weight:600;margin:4px 0 6px;">{title}</div>
            <div style="color:#8090b0;font-size:.9rem;line-height:1.5;">{desc}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Strategic Shift</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:#2a1a1a;border:1px solid #e94560;border-radius:12px;padding:20px;">
        <div style="color:#e94560;font-size:1.1rem;font-weight:700;margin-bottom:12px;">❌ Current Mindset</div>
        <div style="color:#c0a0a0;font-size:.95rem;">"Will I get placement?"</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#1a2a1a;border:1px solid #34d399;border-radius:12px;padding:20px;">
        <div style="color:#34d399;font-size:1.1rem;font-weight:700;margin-bottom:12px;">✅ New Mindset</div>
        <div style="color:#a0c0a0;font-size:.95rem;line-height:1.8;">
        Does this program have:<br>
        ✔ Real projects (5+)<br>
        ✔ Tool exposure (8+)<br>
        ✔ Industry-background faculty<br>
        ✔ Optimal duration (6–12 months)<br>
        ✔ Strong ROI (>2x)
        </div>
        </div>""", unsafe_allow_html=True)

    # Master formula
    st.markdown("""
    <div style="background:linear-gradient(135deg,#e94560 0%,#302b63 100%);
                border-radius:16px;padding:28px 32px;margin-top:24px;text-align:center;">
      <div style="font-family:'Syne',sans-serif;font-size:1.6rem;color:#fff;font-weight:800;margin-bottom:8px;">
        🏆 Best Program Formula
      </div>
      <div style="color:rgba(255,255,255,.85);font-size:1.1rem;">
        <b>Projects + Tools + Faculty + Duration + ROI</b>
      </div>
      <div style="color:rgba(255,255,255,.6);font-size:.9rem;margin-top:8px;">
        ❌ Expensive ≠ Good &nbsp;|&nbsp; ✅ Effective = Hands-on + Real-world + Optimised Duration
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 10 – DATA EXPLORER
# ══════════════════════════════════════════════
with tabs[10]:
    st.markdown('<div class="sec-hdr">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown(f"**{len(dff):,} records** after filters | **{len(dff.columns)} columns**")

    search = st.text_input("🔍 Search program name", "")
    view_df = dff.copy()
    if search and C_PROG_NAME:
        view_df = view_df[view_df[C_PROG_NAME].astype(str).str.lower().str.contains(search.lower())]

    st.dataframe(
        view_df.head(500).reset_index(drop=True),
        use_container_width=True,
        height=420
    )

    st.markdown("#### Quick Stats")
    num_df = dff.select_dtypes(include=np.number)
    st.dataframe(num_df.describe().T.round(2), use_container_width=True)

    csv_out = dff.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data", csv_out,
                       "pragyanai_filtered.csv", "text/csv")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:40px;padding:20px;
            color:#4a5a8a;font-size:.82rem;border-top:1px solid #2a2a4a;">
  🎯 <b>PragyanAI Program Intelligence Engine</b> &nbsp;|&nbsp;
  Built with Python · NumPy · Pandas · Streamlit · Matplotlib
</div>""", unsafe_allow_html=True)
