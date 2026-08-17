#!/usr/bin/env python3
"""Regenerate the three why-we-win exhibits (Tufte pass, grayscale, Times-
matched fonts):

  fig-s1-operator.pdf   -- hierarchical-selection dataflow schematic (F1)
  fig-e1-pscaling.pdf   -- cost(P) model vs measured P-sweep (F2, money fig)
  fig-s2-skip-law.pdf   -- chunk-skip law Eq. 2 + measured A/B callouts (F3)

Every number is read from committed artifacts -- never typed from memory:
  P-sweep points : tests/.../reduction/baselines/comp3/psweep4_full.csv
  chosen P*      : tests/.../reduction/baselines/comp3/competition_table.csv
  merge units    : forecast.md section 3 constants (1.46 / 5.51 us), the same
                   two constants section 4-B of the paper quotes
  skip A/B       : paper-topk/evidence/tileskip/{baseline,skipgated}.csv
  skip law       : Eq. 2 computed exactly (log-gamma) at integer c -- no
                   sampled skip-rate points exist on device (paper 6-D)

Figures are designed AT their final rendered width (inches below == print
inches), so the font sizes below are the printed font sizes.
Style: draft/fig/README.md Tufte rules + grayscale-first (paper prints gray).
"""
import csv
import math
import statistics as st
from math import lgamma

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = "/home/nachiket/tt-metal"
BASE = f"{REPO}/tests/ttnn/unit_tests/operations/reduction/baselines"
TILESKIP = f"{REPO}/paper-topk/evidence/tileskip"
FIGDIR = f"{REPO}/paper-topk/draft/fig"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # Times-compatible math
        "pdf.fonttype": 42,
        "axes.linewidth": 0.5,
    }
)

INK = "#000000"
GRAY45 = "#6e6e6e"
GRAY62 = "#9e9e9e"
LIGHT = "#e8e8e8"


def tufte(ax):
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.5)
    ax.tick_params(width=0.5, labelsize=7)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# =====================================================================
# F1 -- fig-s1-operator: hierarchical selection dataflow (schematic)
# Tournament-bracket tree: leaves -> level nodes -> root, straight edges.
# =====================================================================
fig, ax = plt.subplots(figsize=(3.05, 2.45))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# --- row strip: one row = C chunks of M ---
ROW_Y, ROW_H, ROW_X0, ROW_X1 = 88, 6, 4, 60
NCH = 16
ax.add_patch(Rectangle((ROW_X0, ROW_Y), ROW_X1 - ROW_X0, ROW_H, fill=False, ec=INK, lw=0.7))
for i in range(1, NCH):
    x = ROW_X0 + (ROW_X1 - ROW_X0) * i / NCH
    ax.plot([x, x], [ROW_Y, ROW_Y + ROW_H], color=INK, lw=0.35)
ax.text(
    ROW_X0,
    ROW_Y + ROW_H + 2.5,
    "one row: $N$ elements $=C$ chunks of $M$",
    fontsize=7,
    ha="left",
    va="bottom",
    color=INK,
)

# --- P=4 slices: shaded spans in the strip + arrows down to leaf windows
P_SHOWN = 4
LEAF_Y, LEAF_H, LEAF_W = 66, 5.5, 10
leaf_x = []
for p in range(P_SHOWN):
    span0 = ROW_X0 + (ROW_X1 - ROW_X0) * p / P_SHOWN
    span1 = ROW_X0 + (ROW_X1 - ROW_X0) * (p + 1) / P_SHOWN
    if p % 2 == 1:  # alternate shading marks slice ownership
        ax.add_patch(Rectangle((span0, ROW_Y), span1 - span0, ROW_H, fill=True, fc=LIGHT, ec="none", zorder=0))
    cx = 0.5 * (span0 + span1)
    leaf_x.append(cx)
    ax.annotate(
        "",
        xy=(cx, LEAF_Y + LEAF_H + 0.7),
        xytext=(cx, ROW_Y - 0.7),
        arrowprops=dict(arrowstyle="->", color=GRAY45, lw=0.6),
    )
    ax.add_patch(Rectangle((cx - LEAF_W / 2, LEAF_Y), LEAF_W, LEAF_H, fill=True, fc=LIGHT, ec=INK, lw=0.7))
ax.text(
    64,
    (ROW_Y + LEAF_Y) / 2 + 4,
    "leaf: each of $P$ slices streams\n$\\lceil C/P\\rceil$ chunks"
    " (sort, merge,\nrebuild) into an $M$-wide sorted\nwindow"
    " $\\approx 2\\lceil C/P\\rceil$ merge units",
    fontsize=6.4,
    ha="left",
    va="center",
    color=INK,
)

# --- merge tree: tournament bracket, straight edges ---
LV1_Y, LV2_Y = 48, 30
lv1_x = [(leaf_x[0] + leaf_x[1]) / 2, (leaf_x[2] + leaf_x[3]) / 2]
for i, nx in enumerate(lv1_x):
    for src in (leaf_x[2 * i], leaf_x[2 * i + 1]):
        ax.annotate(
            "", xy=(nx, LV1_Y + 1.2), xytext=(src, LEAF_Y - 0.7), arrowprops=dict(arrowstyle="->", color=INK, lw=0.7)
        )
    ax.plot([nx], [LV1_Y], "s", color=INK, ms=3.4)
root_x = sum(lv1_x) / 2
for nx in lv1_x:
    ax.annotate(
        "", xy=(root_x, LV2_Y + 1.2), xytext=(nx, LV1_Y - 1.2), arrowprops=dict(arrowstyle="->", color=INK, lw=0.7)
    )
ax.plot([root_x], [LV2_Y], "s", color=INK, ms=3.4)
ax.text(leaf_x[0] - 6.5, LV1_Y + 4.5, "level 1", fontsize=6.2, color=GRAY45, ha="right")
ax.text(lv1_x[0] - 5.5, LV2_Y + 4.5, "level 2", fontsize=6.2, color=GRAY45, ha="right")
# one edge annotated = the NoC traffic statement
ax.annotate(
    "each edge: ONE $M$-wide window\n(values + indices) over the"
    " NoC;\n$P{-}1$ transfers total --- losers\nnever leave their"
    " slice",
    xy=((lv1_x[1] + root_x) / 2 + 1, (LV1_Y + LV2_Y) / 2),
    xytext=(64, 34),
    fontsize=6.4,
    ha="left",
    va="center",
    color=INK,
    arrowprops=dict(arrowstyle="-", color=GRAY45, lw=0.5, shrinkA=2, shrinkB=2),
)

# --- root emit ---
ax.annotate("", xy=(root_x, 18), xytext=(root_x, LV2_Y - 1.5), arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
ax.text(root_x - 3.0, 21.5, "root: top-$K$\n(values + indices)", fontsize=6.6, ha="right", va="center", color=INK)

# --- embedding panel: 13x10 grid, 8x4 rectangle, bottom-right ---
GX0, GY0, CELL = 74, 1.5, 1.85
for gx in range(13):
    for gy in range(10):
        ax.plot(GX0 + gx * CELL, GY0 + gy * CELL, ".", color=GRAY62, ms=1.0)
ax.add_patch(
    Rectangle(
        (GX0 - CELL * 0.4, GY0 + 6 * CELL - CELL * 0.4),
        8 * CELL - CELL * 0.2,
        4 * CELL - CELL * 0.2,
        fill=True,
        fc="#cfcfcf",
        ec=INK,
        lw=0.7,
        alpha=0.85,
        zorder=1,
    )
)
ax.text(
    GX0 - 3,
    GY0 + 4.8 * CELL,
    "placement: cost-optimal\n$a{\\times}b$ rectangle ($8{\\times}4$,\n"
    "$P{=}32$) on the $13{\\times}10$\nworker grid",
    fontsize=6.2,
    ha="right",
    va="center",
    color=INK,
)

fig.savefig(f"{FIGDIR}/fig-s1-operator.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("F1 written")

# =====================================================================
# F2 -- fig-e1-pscaling: cost model vs measured P-sweep (money figure)
# =====================================================================
rows = read_csv(f"{BASE}/comp3/psweep4_full.csv")
curves = {}
for r in rows:
    key = (int(r["k"]), int(r["W"]))
    curves.setdefault(key, []).append((int(r["cores"]), float(r["us"])))
for key in curves:
    curves[key].sort()

# calibrated model (forecast.md section 3; identical to the pre-existing
# make-06-evaluation-figs.py overlay): unit * (leaf_units(cmax) + levels),
# leaf_units(c) = 1 + 2(c-1)  [chunk 0 costs one unit, later chunks two]
UNIT = {(2048, 65536): 5.51, (512, 262144): 1.46}
CHUNKS = {(2048, 65536): 32, (512, 262144): 512}


def model_us(key, P):
    cmax = math.ceil(CHUNKS[key] / P)
    return UNIT[key] * ((1 + 2 * (cmax - 1)) + math.ceil(math.log2(P)))


comp = read_csv(f"{BASE}/comp3/competition_table.csv")
chosen = {}
for r in comp:
    key = (int(r["k"]), int(r["W"]))
    if key in curves:
        chosen[key] = (int(r["op_cores"]), float(r["op_us"]))

fig, ax = plt.subplots(figsize=(2.9, 2.15))
tufte(ax)

series = [((2048, 65536), INK, "o"), ((512, 262144), GRAY45, "s")]
report = []
for key, color, mk in series:
    P = [p for p, _ in curves[key]]
    us = [u for _, u in curves[key]]
    mus = [model_us(key, p) for p in P]
    ax.plot(P, mus, "--", color=GRAY62, lw=0.8, zorder=2)
    ax.plot(P, us, mk, color=color, ms=3.1, lw=0, zorder=4, mfc=color if mk == "o" else "white", mew=0.8)
    for p, u, m in zip(P, us, mus):
        report.append((key, p, u, round(m, 1), round(100 * (m - u) / u, 1)))

# direct labels (kept clear of curves)
ax.annotate("$k$=2048,\n$N$=65,536", xy=(2, 183), xytext=(2.02, 58), fontsize=6.4, color=INK, ha="left", va="center")
ax.annotate("$k$=512, $N$=262,144", xy=(4, 284), xytext=(4.8, 500), fontsize=6.4, color=GRAY45, ha="left")
ax.annotate(
    "cost model, no fitting: measured\nunit $\\times$ Eq. 1"
    " (conservative on\ndeep-serial tails: "
    f"+{100*(model_us((512,262144),2)/dict(curves[(512,262144)])[2]-1):.0f}% at $P$=2)",
    xy=(2.02, 1750),
    fontsize=6.0,
    color=GRAY62,
    ha="left",
    va="top",
    style="italic",
)

# ceil-term bump (model predicts it)
ax.annotate(
    "$\\lceil C/P\\rceil$ bump,\npredicted",
    xy=(24, 46),
    xytext=(23, 150),
    fontsize=6.2,
    color=INK,
    ha="center",
    arrowprops=dict(arrowstyle="-", color=INK, lw=0.5, shrinkA=1, shrinkB=2),
)

# chosen-P* rings + labels, deviation computed from data (never typed)
for (key, color, mk), (tx, ty, ha) in zip(series, [(4.3, 26, "left"), (97, 70, "right")]):
    pstar, opus = chosen[key]
    meas = dict(curves[key])[pstar]
    dev = abs(100 * (model_us(key, pstar) - meas) / meas)
    ax.plot([pstar], [meas], mk, color=color, ms=5.8, mfc="none", mew=0.9, zorder=5)
    ax.annotate(
        f"chosen $P^{{*}}$={pstar}:\n{meas:.1f} µs" f" ({dev:.1f}% off)",
        xy=(pstar, meas),
        xytext=(tx, ty),
        fontsize=6.2,
        color=color,
        ha=ha,
        va="center",
        arrowprops=dict(arrowstyle="-", color=color, lw=0.4, shrinkA=1, shrinkB=4),
    )

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks([2, 4, 8, 16, 32, 64, 104])
ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "104"])
ax.set_yticks([20, 50, 100, 200, 500, 1000])
ax.set_yticklabels(["20", "50", "100", "200", "500", "1000"])
ax.set_xlim(1.85, 130)
ax.set_ylim(16.5, 1900)
ax.minorticks_off()
ax.set_xlabel("cores $P$  (factory floor: $P\\geq 2$)", fontsize=7.5)
ax.set_ylabel("device kernel time (µs)", fontsize=7.5)
fig.tight_layout(pad=0.25)
fig.savefig(f"{FIGDIR}/fig-e1-pscaling.pdf")
plt.close(fig)
print("F2 written; model fit (key, P, meas, model, delta%):")
for row in report:
    print("  ", row)

# =====================================================================
# F3 -- fig-s2-skip-law: Eq. 2 curves + measured end-to-end callouts
# =====================================================================
M = 512  # LLK window for both regimes shown (K=32 -> M=512; K=512 -> M=512)


def exact_p(K, c):
    def lc(n, k):
        return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)

    return math.exp(lc(c * M, K) - lc((c + 1) * M, K))


def medians(path):
    per = {}
    for r in read_csv(path):
        per.setdefault(r["cell"], []).append(int(r["median_ns"]))
    return {c: st.median(v) / 1000.0 for c, v in per.items()}


b = medians(f"{TILESKIP}/baseline.csv")
g = medians(f"{TILESKIP}/skipgated.csv")
d2 = 100 * (g["r2_n65536_k32"] - b["r2_n65536_k32"]) / b["r2_n65536_k32"]
d8 = 100 * (g["r8_n65536_k32"] - b["r8_n65536_k32"]) / b["r8_n65536_k32"]
print(f"F3 measured deltas: rows=2 {d2:+.1f}%, rows=8 {d8:+.1f}%")

fig, ax = plt.subplots(figsize=(2.9, 2.1))
tufte(ax)
FLOOR = 1e-9
cints = list(range(1, 257))
for K, color in [(32, INK), (512, GRAY45)]:
    xs = [c for c in cints if exact_p(K, c) >= FLOOR]
    ys = [exact_p(K, c) for c in xs]
    ax.plot(xs, ys, "-", color=color, lw=1.0, zorder=3)
    xa = [c for c in cints if math.exp(-K / (c + 1)) >= FLOOR]
    ax.plot(xa, [math.exp(-K / (c + 1)) for c in xa], ":", color=color, lw=0.8, zorder=2)

# regime shading: column-parallel streams are 1-5 chunks (no payoff)
ax.axvspan(1, 5, color="#dcdcdc", alpha=0.55, lw=0, zorder=1)
ax.text(
    2.25,
    4e-7,
    "column-\nparallel\n$\\lceil C/P\\rceil$=1–5:\nno" " payoff\n($K$=512 at $c$=1:\n$2{\\times}10^{-307}$)",
    fontsize=5.8,
    color=INK,
    ha="center",
    va="center",
)
ax.text(17, 3e-8, "row-parallel: live", fontsize=6.0, color=GRAY45, ha="left")

# gates (labels kept low, clear of the curves)
for K, color in [(32, INK), (512, GRAY45)]:
    cgate = max(2, K // 4)
    ax.axvline(cgate, color=color, lw=0.6, ls="--", zorder=2)
ax.text(8 * 1.10, 1.1e-8, "gate $\\max(2,K/4)$, $K$=32", fontsize=5.8, color=INK, ha="left", rotation=90, va="bottom")
ax.text(128 * 0.90, 1.1e-8, "gate, $K$=512", fontsize=5.8, color=GRAY45, ha="right", rotation=90, va="bottom")

# anchor points, computed here (the prose quotes the same three)
for K, c, color, dx, dy, txt in [
    (32, 32, INK, 0.78, 2.6, None),
    (32, 128, INK, 0.60, 2.2, None),
    (512, 128, GRAY45, 1.12, 0.75, None),
]:
    y = exact_p(K, c)
    ax.plot([c], [y], "o", color=color, ms=2.6, zorder=4)
    lbl = txt or (f"{100*y:.0f}%" if y > 0.05 else f"{100*y:.1f}%")
    ax.annotate(lbl, xy=(c, y), xytext=(c * dx, y * dy), fontsize=6.3, color=color)

# curve identity labels
ax.text(7.2, 0.42, "$K$=32", fontsize=6.8, color=INK, ha="right")
ax.text(52, 8e-7, "$K$=512", fontsize=6.8, color=GRAY45, ha="left")
ax.text(1.13, 0.055, "exact (solid),\n$e^{-K/(c+1)}$ (dotted)", fontsize=5.9, color=INK, style="italic", ha="left")

# measured end-to-end callout (the only silicon numbers on this plot)
ax.text(
    272,
    1.1e-5,
    "$K$=32 row-parallel, measured\nend-to-end at $C$=128:\n"
    f"\u2212{abs(d2):.1f}% (rows=2), \u2212{abs(d8):.1f}% (rows=8)",
    fontsize=6.0,
    color=INK,
    ha="right",
    va="center",
)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256])
ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256"])
ax.set_xlim(1, 280)
ax.set_ylim(FLOOR, 4)
ax.set_yticks([1, 1e-2, 1e-4, 1e-6, 1e-8])
ax.set_yticklabels(["$10^{0}$", "$10^{-2}$", "$10^{-4}$", "$10^{-6}$", "$10^{-8}$"])
ax.minorticks_off()
ax.set_xlabel("stream position $c$ (chunks already seen)", fontsize=7.5)
ax.set_ylabel("$\\Pr[\\mathrm{skip\\ chunk}\\ c{+}1]$", fontsize=7.5)
fig.tight_layout(pad=0.25)
fig.savefig(f"{FIGDIR}/fig-s2-skip-law.pdf")
plt.close(fig)
print("F3 written")
print("Done.")
