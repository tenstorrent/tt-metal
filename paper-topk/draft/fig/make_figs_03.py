# Figures owned by sections/03-design-space.tex (brief 03).
# EVIDENCE DISCIPLINE: every number below traces to
#   paper-topk/evidence/paper/evidence.md rows C1-1..C1-5 (which point at
#   paper-topk/evidence/validate/cgtceq-debug.md and the risc_scan_bench /
#   cgtceq_perf commit-message numbers) and RADIX_BUCKET_GPU.md section 7.
# No number here is invented; per-value provenance is in inline comments.
#
# Style: Tufte rules per paper-topk/draft/fig/README.md +
# /home/nachiket/writing-style/nachiket_writing_style.md (no grid, thin
# spines, direct annotation, frameless legend above, flat 2D).
#
# Colors (CVD-safe Okabe-Ito pair + neutral gray; identity is doubly
# encoded by direct text labels, never color alone):
#   counting        -> #0072B2 (blue)
#   materialization -> #E69F00 (orange)
#   floor/incumbent -> #666666 (gray)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

C_COUNT = "#0072B2"
C_MAT = "#E69F00"
C_FLOOR = "#666666"
RED_BG = "#FFE5E5"

plt.rcParams.update(
    {
        "font.size": 7.5,
        "font.family": "sans-serif",
        "axes.linewidth": 0.5,
        "pdf.fonttype": 42,
    }
)


def tufte(ax):
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.5)


# ----------------------------------------------------------------------
# Fig D1 - the Q1/Q2/Q3 design-space map (RADIX_BUCKET_GPU.md section 7.1)
# Conceptual figure: cell placements are the (A)-(F) taxonomy; the two
# quantitative annotations (~2 cyc/elem bitonic pass, 81 cyc rendezvous)
# trace to RBG 7.1/7.2 and evidence C1-3.
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.axis("off")

# quadrant separators (hairline)
ax.plot([1, 1], [0, 2], color="#bbbbbb", lw=0.5, zorder=1)
ax.plot([0, 2], [1, 1], color="#bbbbbb", lw=0.5, zorder=1)

# axis labels (Q1 across the bottom, Q2 up the left side)
ax.text(0.5, -0.07, "(A) comparison networks", ha="center", va="top", fontsize=7)
ax.text(1.5, -0.07, "(B) counting / partitioning", ha="center", va="top", fontsize=7)
ax.text(1.0, -0.22, "Q1 — winner identification", ha="center", va="top",
        fontsize=7.5, style="italic")
ax.text(-0.05, 0.5, "(C) full-data passes", ha="right", va="center",
        rotation=90, fontsize=7)
ax.text(-0.05, 1.5, "(D) shrinking candidates", ha="right", va="center",
        rotation=90, fontsize=7)
ax.text(-0.2, 1.0, "Q2 — data touched per pass", ha="right", va="center",
        rotation=90, fontsize=7.5, style="italic")


def cell(x, y, lines, fc, ec, ls="-", tc="black"):
    box = FancyBboxPatch(
        (x - 0.44, y - 0.30), 0.88, 0.60,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fc, edgecolor=ec, linewidth=0.8, linestyle=ls, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x, y, lines, ha="center", va="center", fontsize=6.6, color=tc,
            zorder=3, linespacing=1.35)


# (A,C): shipping bitonic engine — RBG 7.1 "(A)+(C)+(E) ... ~2 cyc/elem"
cell(0.5, 0.5,
     "shipping bitonic\n(A)+(C)+(E)\n$\\approx$2 cyc/elem every pass,\nzero decisions",
     "#ebebeb", C_FLOOR)
# (B,D): GPU radix select — RBG 7.1 "(B)+(D)+cheap-(F)"
cell(1.5, 1.5,
     "GPU radix select\n(B)+(D)+cheap (F)\natomic histograms\n+ scatter compaction",
     "#dceaf4", C_COUNT)
# (B,C): the degenerate Blackhole corner — RBG 7.2 close; 81 cyc = C1-3
cell(1.5, 0.5,
     "Blackhole corner (§3)\n(B)+(C)+(F)\nthreshold bisection;\n81-cyc rendezvous",
     "#fdeed3", C_MAT)
# (A,D): chunk-skip forward pointer (section 4 builds it)
cell(0.5, 1.5,
     "chunk-skip (§4)\n(A)+(D)+(F)\nsound skip,\nno scatter needed",
     "white", C_FLOOR, ls="--")

# the blocking arrow: no scatter => (D) unreachable from (B)
arrow = FancyArrowPatch(
    (1.5, 1.18), (1.5, 0.83),
    arrowstyle="-|>", mutation_scale=8, color="#B03030", lw=1.0, zorder=4,
)
ax.add_patch(arrow)
ax.text(1.53, 1.005, "no scatter:\n(D) unreachable", ha="left", va="center",
        fontsize=6.2, color="#B03030", zorder=4)

fig.subplots_adjust(left=0.13, right=0.99, top=0.99, bottom=0.13)
fig.savefig("/home/nachiket/tt-metal/paper-topk/draft/fig/fig-d1-design-space.pdf")
plt.close(fig)

# ----------------------------------------------------------------------
# Fig D2 - engine shootout: measured per-element mechanism costs.
# All values in cycles, slope-measured (no us conversion -> no clock
# caveat needed on this figure).  Provenance per bar:
#   SFPU 1-bit count 2.0 cyc/vec exact  -> C1-1 (cgtceq_perf rate arm);
#     /32 lanes = 0.0625 cyc/elem (arithmetic normalization only)
#   compressed-stream consumer 0.63 cyc/orig-elem -> C1-5 (risc_scan_bench)
#   RISC pure-load floor 3.02 cyc/elem  -> C1-2 (risc_scan_bench)
#   RISC 256-bin histogram 7.56-7.90    -> C1-2
#   dense RISC emit 12.97-17.81         -> C1-5
#   L1-resident histogram 15.5-19.9     -> C1-2
# Reference lines: compaction bar ~0.5 cyc/elem (C1-5), bitonic leaf
# ~2 cyc/elem (RBG 7.1).  Decision cost 81 cyc (C1-3) annotated as text.
# ----------------------------------------------------------------------
rows = [
    # (label, lo, hi or None, color, value-label)
    ("SFPU 1-bit count\n(2.0 cyc/vec $\\div$ 32)", 0.0625, None, C_COUNT, "0.063"),
    ("compressed-stream\nconsumer", 0.63, None, C_MAT, "0.63"),
    ("RISC pure-load floor", 3.02, None, C_FLOOR, "3.02"),
    ("RISC 256-bin\nhistogram", 7.56, 7.90, C_COUNT, "7.56–7.90"),
    ("dense RISC emit", 12.97, 17.81, C_MAT, "12.97–17.81"),
    ("L1-resident\nhistogram", 15.5, 19.9, C_COUNT, "15.5–19.9"),
]

fig, ax = plt.subplots(figsize=(3.5, 2.5))
ypos = list(range(len(rows)))[::-1]
NTOP = len(rows) + 0.7  # headroom band for reference-line annotations

# red band: costlier than one bitonic leaf pass (>2 cyc/elem)
ax.add_patch(Rectangle((2.0, -0.6), 90 - 2.0, NTOP + 0.6, facecolor=RED_BG,
                       edgecolor="none", alpha=0.5, zorder=0))

for (label, lo, hi, color, vlabel), y in zip(rows, ypos):
    ax.barh(y, lo, height=0.55, color=color, edgecolor="none", zorder=2)
    if hi is not None:  # measured min-max range as a lighter extension
        ax.barh(y, hi - lo, left=lo, height=0.55, color=color, alpha=0.45,
                edgecolor="none", zorder=2)
    xend = hi if hi is not None else lo
    ax.text(xend * 1.15, y, vlabel, va="center", ha="left", fontsize=6.8,
            zorder=3)

# reference lines, directly annotated (anchored away from each other)
ax.axvline(0.5, ymax=0.99, color="#B03030", lw=0.7, ls=":", zorder=1)
ax.text(0.45, len(rows) - 0.05, "compaction bar $\\approx$0.5",
        ha="right", va="bottom", fontsize=6.2, color="#B03030")
ax.axvline(2.0, ymax=0.99, color="#333333", lw=0.7, ls="--", zorder=1)
ax.text(2.2, len(rows) - 0.05, "bitonic leaf pass $\\approx$2",
        ha="left", va="bottom", fontsize=6.2, color="#333333")

ax.set_ylim(-0.6, NTOP)
ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows], fontsize=6.6)
ax.set_xscale("log")
ax.set_xlim(0.04, 150)
ax.set_xticks([0.063, 0.5, 2, 10, 20])
ax.set_xticklabels(["0.063", "0.5", "2", "10", "20"])
ax.set_xlabel("cycles per element (log scale)", fontsize=7)
ax.tick_params(width=0.5, labelsize=6.8)
tufte(ax)

# frameless legend above the plot (role identity; labels double-encode)
handles = [
    plt.Rectangle((0, 0), 1, 1, color=C_COUNT),
    plt.Rectangle((0, 0), 1, 1, color=C_MAT),
    plt.Rectangle((0, 0), 1, 1, color=C_FLOOR),
]
ax.legend(handles, ["counting", "materialization", "floor"],
          loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False,
          fontsize=6.8, ncol=3, columnspacing=1.2, handlelength=1.0,
          handleheight=0.8)

# the decision-cost annotation (not a per-element quantity)
fig.text(0.02, 0.015,
         "one data-dependent decision: 81 cyc (fold + sync + MMIO read)",
         fontsize=6.2, ha="left", va="bottom", style="italic")

fig.subplots_adjust(left=0.30, right=0.98, top=0.90, bottom=0.24)
fig.savefig("/home/nachiket/tt-metal/paper-topk/draft/fig/fig-d2-engine-shootout.pdf")
plt.close(fig)

print("wrote fig-d1-design-space.pdf, fig-d2-engine-shootout.pdf")
