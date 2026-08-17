#!/usr/bin/env python3
"""Generate Fig E1 (P-scaling) and Fig E2 (chunk-skip A/B) PDFs plus
booktabs table bodies for section 06-evaluation, all read from committed
CSVs / evidence-pack artifacts. Tufte rules per draft/fig/README.md."""
import csv
import math
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/nachiket/tt-metal"
BASE = f"{REPO}/tests/ttnn/unit_tests/operations/reduction/baselines"
TILESKIP = f"{REPO}/paper-topk/evidence/tileskip"
FIGDIR = f"{REPO}/paper-topk/draft/fig"
OUT = os.path.dirname(os.path.abspath(__file__))

# ---- consistent arm colors across ALL figures (fig/README rule) ----
C_OP = "#1f77b4"        # measured op arm (baseline in Fig E2 = op w/o skip)
C_OP2 = "#9467bd"       # second measured op cell in Fig E1
C_UNGATED = "#ff7f0e"   # skip, ungated
C_GATED = "#2ca02c"     # skip, gated (final)
C_MODEL = "#888888"     # analytic cost model overlay


def tufte(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(width=0.5)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# =====================================================================
# Fig E1 — P-scaling with cost-model overlay
# =====================================================================
rows = read_csv(f"{BASE}/comp3/psweep4_full.csv")
curves = {}
for r in rows:
    key = (int(r["k"]), int(r["W"]))
    curves.setdefault(key, []).append((int(r["cores"]), float(r["us"])))
for key in curves:
    curves[key].sort()

# cost model: measured = unit * (leaf_units(cmax) + levels)
# leaf_units(c) = 1 + 2(c-1); levels = ceil(log2 P); C = N / llk_k
# unit constants (Tracy-calibrated, forecast.md section 3):
UNIT = {(2048, 65536): 5.51, (512, 262144): 1.46}
CHUNKS = {(2048, 65536): 32, (512, 262144): 512}


def model_us(key, P):
    C = CHUNKS[key]
    cmax = math.ceil(C / P)
    leaf = 1 + 2 * (cmax - 1)
    levels = math.ceil(math.log2(P))
    return UNIT[key] * (leaf + levels)


fig, ax = plt.subplots(figsize=(3.6, 2.6))
tufte(ax)

fit_report = []
for key, color, label, lx in [
    ((2048, 65536), C_OP, "k=2048, N=65{,}536", None),
    ((512, 262144), C_OP2, "k=512, N=262{,}144", None),
]:
    P = [p for p, _ in curves[key]]
    us = [u for _, u in curves[key]]
    ax.plot(P, us, "-o", color=color, lw=1.2, ms=3.5, zorder=3)
    mus = [model_us(key, p) for p in P]
    ax.plot(P, mus, "--", color=C_MODEL, lw=0.9, zorder=2)
    for p, u, m in zip(P, us, mus):
        fit_report.append((key, p, u, round(m, 1), round(100 * (m - u) / u, 1)))

# direct labels (no legend)
ax.annotate("k=2048, N=65,536", xy=(2, 183), xytext=(2.05, 92),
            fontsize=7, color=C_OP, ha="left")
ax.annotate("k=512, N=262,144", xy=(4, 284), xytext=(3.6, 430),
            fontsize=7, color=C_OP2, ha="left")
ax.annotate("cost model", xy=(2.3, 700), fontsize=7, color=C_MODEL,
            style="italic", ha="left")
# the ceil-term bump at P=24 (model predicts it)
ax.annotate("⌈C/P⌉ bump\n(model predicts it)",
            xy=(24, 44.5), xytext=(17, 110),
            fontsize=6.5, color=C_OP, ha="center",
            arrowprops=dict(arrowstyle="-", color=C_OP, lw=0.5,
                            shrinkA=2, shrinkB=2))

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks([2, 4, 8, 16, 32, 64, 104])
ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "104"], fontsize=7)
ax.set_yticks([20, 50, 100, 200, 500])
ax.set_yticklabels(["20", "50", "100", "200", "500"], fontsize=7)
ax.set_ylim(16, 950)
ax.minorticks_off()
ax.set_xlabel("cores P", fontsize=8)
ax.set_ylabel("device kernel time (µs)", fontsize=8)
fig.tight_layout(pad=0.3)
fig.savefig(f"{FIGDIR}/fig-e1-pscaling.pdf")
plt.close(fig)

print("== Fig E1 model fit (key, P, measured, model, delta%) ==")
for row in fit_report:
    print(row)

# =====================================================================
# Fig E2 — chunk-skip A/B bars (normalized to baseline, direct labels)
# =====================================================================
CELLS = ["r2_n65536_k32", "r2_n65536_k512", "r8_n65536_k32",
         "r640_n51200_k1536", "r2_n102400_k1536v"]
LABELS = ["rows=2\nk=32\n279 µs", "rows=2\nk=512\n279 µs",
          "rows=8\nk=32\n279 µs", "rows=640\nk=1536\n1,377 µs",
          "valid=56,320\nk=1536\n312 µs"]


def medians(path):
    per = {}
    for r in read_csv(path):
        per.setdefault(r["cell"], []).append(int(r["median_ns"]))
    return {c: st.median(v) / 1000.0 for c, v in per.items()}  # us


base = medians(f"{TILESKIP}/baseline.csv")
ungated = medians(f"{TILESKIP}/skipon.csv")
gated = medians(f"{TILESKIP}/skipgated.csv")

print("\n== Fig E2 cell medians (us) ==")
for c in CELLS:
    print(c, round(base[c], 2), round(ungated[c], 2), round(gated[c], 2),
          f"gated delta {100*(gated[c]-base[c])/base[c]:+.2f}%")

fig, ax = plt.subplots(figsize=(3.6, 2.4))
tufte(ax)
x = range(len(CELLS))
w = 0.27
for off, arm, color in [(-w, base, C_OP), (0, ungated, C_UNGATED),
                        (w, gated, C_GATED)]:
    vals = [arm[c] / base[c] for c in CELLS]
    ax.bar([i + off for i in x], vals, width=w * 0.92, color=color,
           edgecolor="none")

# direct annotations: gated delta % centered over the gated bar
# (deltas computed from the 0.01-us-rounded medians, matching the
#  implementation report's table exactly)
for i, c in enumerate(CELLS):
    d = 100 * (round(gated[c], 2) - round(base[c], 2)) / round(base[c], 2)
    txt = f"{d:+.2f}%" if abs(d) < 1 else f"−{abs(d):.1f}%"
    ax.annotate(txt, xy=(i + w, gated[c] / base[c]), xytext=(0, 2),
                textcoords="offset points", ha="center", fontsize=6.5,
                color=C_GATED)

# frame-free legend above the plot (fig/README rule)
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=C_OP, label="baseline"),
           mpatches.Patch(color=C_UNGATED, label="skip, ungated"),
           mpatches.Patch(color=C_GATED, label="skip, gated")]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
          frameon=False, fontsize=7, ncol=3, columnspacing=1.2,
          handlelength=1.0)

ax.set_xticks(list(x))
ax.set_xticklabels(LABELS, fontsize=6.5)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=7)
ax.set_ylim(0, 1.12)
ax.set_ylabel("runtime relative to baseline", fontsize=8)
fig.tight_layout(pad=0.3)
fig.savefig(f"{FIGDIR}/fig-e2-skip-ab.pdf")
plt.close(fig)

# =====================================================================
# Tab E1 — competition table body
# =====================================================================
comp = read_csv(f"{BASE}/comp3/competition_table.csv")


def us_fmt(v, nd=1):
    f = float(v)
    return f"{f:,.{nd}f}"


def ms_fmt(v):
    return f"{float(v)/1000.0:,.1f}"


lines = []
prev_k = None
for r in comp:
    k, W = r["k"], int(r["W"])
    kcell = k if k != prev_k else ""
    prev_k = k
    sp_op = round(float(r["speedup_prebranch_over_op"].replace("x", "")))
    sp_rt = round(float(r["speedup_prebranch_over_routed"].replace("x", "")))
    mark = "$^\\ddagger$" if r["blaze_us"] else ""
    lines.append(
        f"{kcell} & {W:,} & {ms_fmt(r['prebranch_us'])} & "
        f"{ms_fmt(r['stocknow_us'])} & "
        f"{us_fmt(r['routed_us'],1)} ({r['routed_cores']}) & "
        f"{us_fmt(r['op_us'],1)}{mark} ({r['op_cores']}) & "
        f"{us_fmt(r['opstock_us'],1)} & "
        f"{float(r['roofline_us']):.2f} & "
        f"{float(r['gap_op_vs_roofline']):.1f}$\\times$ & "
        f"{sp_rt:,}$\\times$ & {sp_op:,}$\\times$ \\\\"
    )
with open(f"{OUT}/tab_e1_body.tex", "w") as f:
    f.write("\n".join(lines) + "\n")

# =====================================================================
# Tab E2 — small-k routing before/after
# =====================================================================
routed = {(int(r["k"]), int(r["W"])): (float(r["routed_us"]),
                                       int(r["routed_cores"]))
          for r in read_csv(f"{BASE}/smallk_routefix/routed_after.csv")}
before = {}
for r in read_csv(f"{BASE}/smallk_routefix/stock_nonpow2.csv"):
    before[(int(r["k"]), int(r["n"]))] = (
        float(r["baseline_ns_mean"]) / 1000.0,
        float(r["baseline_ns_std"]) / 1000.0)
for r in read_csv(f"{BASE}/scope51/canonical_sweep.csv"):
    key = (int(r["k"]), int(r["n"]))
    if int(r["n"]) in (65534, 65536) and key[0] in (8, 32, 64):
        before[key] = (float(r["baseline_ns_mean"]) / 1000.0,
                       float(r["baseline_ns_std"]) / 1000.0)

lines = []
speeds = []
prev_k = None
for (k, W) in sorted(routed):
    if (k, W) not in before:
        continue
    b, bstd = before[(k, W)]
    a, cores = routed[(k, W)]
    s = b / a
    speeds.append(((k, W), round(s, 1)))
    kcell = str(k) if k != prev_k else ""
    prev_k = k
    bf = f"{b:,.0f}" if b >= 1000 else f"{b:.0f}"
    sf = f"{s:.1f}" if s < 100 else f"{s:,.0f}"
    lines.append(f"{kcell} & {W:,} & {bf} & {a:.2f} ({cores}) & "
                 f"{sf}$\\times$ \\\\")
with open(f"{OUT}/tab_e2_body.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n== Tab E2 speedups ==")
for s in speeds:
    print(s)

# =====================================================================
# Tab E3 — scenarios
# =====================================================================
scen = read_csv(f"{BASE}/comp3/scenarios1_table.csv")
print("\n== scenarios ==")
for r in scen:
    print(r["scenario"], r["rows"], r["n"], r["k"], "today", r["today_us"],
          "routed", r["routed_us"], "op", r["op_us"], r["op_cores"],
          r["speedup_today_over_op"])

print("\nDone. PDFs in", FIGDIR)
