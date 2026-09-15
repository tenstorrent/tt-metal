#!/usr/bin/env python
"""Page A figures, part 2 (DRAM law, causal vs non-causal, chain roles, spans, grid, step swimlane, zone method).

Read-only on inputs under handoff/revamp/{bh,data/bh_zones}; writes PNGs to handoff/revamp/figs only.
Interpreter: $POLARIS/.venv/bin/python (matplotlib 3.11, numpy 2.5).
The anchor decomposition figures (a_zone_tax, a_anchor_*, a_reader_breakdown, a_ablation_*, ...) are NOT made here.
"""
import csv
import os
import statistics
import textwrap
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

# Paths per PORTABLE_CONTRACT.md: ROOT is handoff/revamp (the directory above this file's),
# WORK the SDPA root above that; both are environment overrides.
ROOT = os.environ.get("HANDOFF", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "bh_zones")
OUT = os.path.join(ROOT, "figs")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, BG = "#0b0b0b", "#fcfcfb"
GRAY = "#b9b9b5"
MUTED = "#5c5c58"
LIGHT = "#ececea"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.color": "#e4e4e0",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.family": "DejaVu Sans",
    }
)
W_IN = 1600 / 150  # 1600 px at 150 dpi
MHZ = 1350.0
CORES = 110
BYTES_PER_KTILE_BFP8 = 8704  # (16 K + 16 V tiles) x 1088 B per k128 step = 34,816 B, 4 k-tiles per step
BYTES_PER_KTILE_BF16 = 16384  # 65,536 B per k128 step (a bf16 tile is 2,048 B; review/sdpa_refit_review.md R1)
FMT = matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:,.0f}")


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def heading(fig, title, subtitle=None, y=0.975, title_w=92, sub_w=138):
    t = "\n".join(textwrap.wrap(title, title_w))
    fig.text(0.012, y, t, fontsize=13, fontweight="bold", va="top", ha="left", color=INK)
    if subtitle:
        n = t.count("\n") + 1
        st = "\n".join(textwrap.wrap(subtitle, sub_w))
        fig.text(0.012, y - 0.033 * n - 0.01, st, fontsize=9.5, va="top", ha="left", color=MUTED)


def footer(fig, text):
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(text, 165)), fontsize=8, color=MUTED, ha="left", va="bottom")


# ---------------------------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------------------------
def read_csv(name):
    with open(os.path.join(D, name)) as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def wall_mean(tag_runs):
    """Mean device wall over invocations 1 and 2 (first invocation discarded), from <tag>_runs.csv."""
    rows = read_csv(tag_runs)
    return statistics.mean(float(r["wall_dev_cycles"]) for r in rows if r["run_idx"] in ("1", "2"))


def wall_core(tag_runs, run_idx="1"):
    rows = read_csv(tag_runs)
    r = [r for r in rows if r["run_idx"] == run_idx][0]
    return int(r["wall_core_x"]), int(r["wall_core_y"])


def trisc1_spans(tag_cores):
    """Per-core TRISC_1 KERNEL span, mean of invocations 1 and 2; dict (x, y) -> cycles."""
    acc = defaultdict(list)
    for r in read_csv(tag_cores):
        if r["risc"] == "TRISC_1" and r["run_idx"] in ("1", "2"):
            acc[(int(r["core_x"]), int(r["core_y"]))].append(float(r["kernel_dur"]))
    return {k: statistics.mean(v) for k, v in acc.items()}


def reader_counts(tag_cores, col="R_K_READ_N", run_idx="1"):
    out = {}
    for r in read_csv(tag_cores):
        if r["risc"] == "NCRISC" and r["run_idx"] == run_idx:
            out[(int(r["core_x"]), int(r["core_y"]))] = (float(r[col]), float(r["R_KCHUNK_N"]))
    return out


def decomp_parts(tag):
    """Corrected zone sums per (risc, part), mean over iterations 1 and 2, from decomp_<tag>.csv (wall core)."""
    acc = defaultdict(list)
    cnt = defaultdict(list)
    for r in read_csv(f"decomp_{tag}.csv"):
        if r["iter"] in ("1", "2") and r["cycles_corr"] != "":
            acc[(r["risc"], r["part"])].append(float(r["cycles_corr"]))
            if r["count"] != "":
                cnt[(r["risc"], r["part"])].append(float(r["count"]))
    parts = {k: statistics.mean(v) for k, v in acc.items()}
    counts = {k: statistics.mean(v) for k, v in cnt.items()}
    return parts, counts


GRID_X = [1, 2, 3, 4, 5, 6, 7, 11, 13, 14, 15]  # profiler core_x of the 110 workers (physical NoC coordinates)
GRID_Y = list(range(2, 12))  # profiler core_y


# ---------------------------------------------------------------------------------------------
# 1. a_dram_law.png
# ---------------------------------------------------------------------------------------------
def fig_dram_law():
    rate = {r["tag"]: r for r in read_csv("dram_rate_table.csv")}
    base = float(rate["t21_causal_q128k128"]["per_ktile_cycles"])
    base_k256 = float(rate["t21_causal_q128k256"]["per_ktile_cycles"])
    base_k512 = float(rate["t21_causal_q128k512"]["per_ktile_cycles"])
    a4 = float(rate["t22_abl_a4_causal_q128k128"]["per_ktile_cycles"])
    a4_k256 = float(rate["t22_abl_a4_causal_q128k256"]["per_ktile_cycles"])
    a4_k512 = float(rate["t22_abl_a4_causal_q128k512"]["per_ktile_cycles"])
    kvbf16 = wall_mean("t23_causal_q128k128_kvbf16_zoff_runs.csv") / 165 / 4
    allbf16 = wall_mean("t23_causal_q128k128_allbf16_zoff_runs.csv") / 165 / 4
    # two point law (base bfp8, K/V bf16) on the raw wall per k-tile: 695 + 0.3663; the model fits the same two walls net of
    # the 2,900-cycle wall_fixed_cycles and gets 691 + 0.3662 (model/refit_notes.md s2; 1,070 + 0.3232 on the 2,176 B tile before R1)
    slope = (kvbf16 - base) / (BYTES_PER_KTILE_BF16 - BYTES_PER_KTILE_BFP8)
    icpt = base - slope * BYTES_PER_KTILE_BFP8
    floor = 915585.6 / 165 / 4  # wall-core model floor per k-tile (RT T3: 915,585.6 cycles over 165 steps x 4 k-tiles)
    print(
        f"dram law: intercept {icpt:.0f} slope {slope:.4f}; points base {base:.0f} kvbf16 {kvbf16:.0f} allbf16 {allbf16:.0f} a4 {a4:.0f} floor {floor:.0f}"
    )

    fig, ax = plt.subplots(figsize=(W_IN, 7.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.265)
    xs = np.linspace(0, 20000, 50)
    ax.plot(
        xs,
        icpt + slope * xs,
        color=INK,
        lw=1.8,
        label=f"two point law {icpt:,.0f} + {slope:.4f} x bytes per k-tile through the two filled blue points (INFERRED);\nnet of the 2,900-cycle fixed launch cost the model's fit on the same two walls is 691 + 0.3662 (model/refit_notes.md s2)",
    )
    ax.axhline(
        floor,
        color=ORANGE,
        lw=1.4,
        ls=(0, (5, 3)),
        label=f"compute floor per k-tile on the wall core, {floor:,.0f} cycles (MODEL, predict() floor x 165 steps)",
    )
    ax.axhline(icpt, color=MUTED, lw=0.9, ls=(0, (1, 3)))
    ax.text(
        19800,
        icpt - 110,
        f"law intercept {icpt:,.0f}: fixed per k-tile cost not covered by the stream",
        ha="right",
        va="top",
        fontsize=8.8,
        color=MUTED,
    )
    ax.scatter([BYTES_PER_KTILE_BFP8], [base], s=110, color=BLUE, edgecolor=BG, linewidth=1.2, zorder=6)
    ax.scatter([BYTES_PER_KTILE_BF16], [kvbf16], s=110, color=BLUE, edgecolor=BG, linewidth=1.2, zorder=7)
    ax.scatter([BYTES_PER_KTILE_BF16], [allbf16], s=90, facecolor=BG, edgecolor=BLUE, linewidth=1.8, zorder=6)
    ax.scatter([0], [a4], s=110, color=RED, edgecolor=BG, linewidth=1.2, zorder=6)
    ax.scatter(
        [BYTES_PER_KTILE_BFP8] * 2,
        [base_k256, base_k512],
        s=45,
        facecolor=BG,
        edgecolor=BLUE,
        linewidth=1.4,
        zorder=5,
        marker="s",
    )
    ax.scatter([0] * 2, [a4_k256, a4_k512], s=45, facecolor=BG, edgecolor=RED, linewidth=1.4, zorder=5, marker="s")
    ax.annotate(
        f"bfp8 K/V: {base:,.0f} cycles per k-tile at\n{BYTES_PER_KTILE_BFP8:,} B per k-tile per core (34,816 B per k128 step)",
        xy=(BYTES_PER_KTILE_BFP8, base),
        xytext=(9300, 2150),
        fontsize=9.2,
        color=BLUE,
        va="top",
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8),
    )
    ax.text(
        BYTES_PER_KTILE_BFP8 + 250,
        3700,
        f"squares: k256 {base_k256:,.0f}, k512 {base_k512:,.0f}\n(same bytes per k-tile, flat within 2.4 percent)",
        fontsize=8.6,
        color=BLUE,
        va="top",
    )
    ax.annotate(
        f"K and V bf16, Q bfp8: {kvbf16:,.0f} cycles per k-tile at\n{BYTES_PER_KTILE_BF16:,} B per k-tile (65,536 B per step, 2,048 B tiles)",
        xy=(BYTES_PER_KTILE_BF16, kvbf16),
        xytext=(9400, 7850),
        fontsize=9.2,
        color=BLUE,
        va="top",
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8),
    )
    ax.annotate(
        f"all bf16 (Q, K, V, out; 2 reads in flight): {allbf16:,.0f}",
        xy=(BYTES_PER_KTILE_BF16, allbf16),
        xytext=(12200, 4500),
        fontsize=9.2,
        color=BLUE,
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8),
    )
    ax.annotate(
        f"A4 reader stub (K/V reads replaced by reserve and push): {a4:,.0f}\n= free-reader reference: floor plus compute front end",
        xy=(0, a4),
        xytext=(1300, 3250),
        fontsize=9.2,
        color=RED,
        va="top",
        arrowprops=dict(arrowstyle="-", color=RED, lw=0.8),
    )
    ax.text(350, 1560, f"A4 squares: k256 {a4_k256:,.0f}, k512 {a4_k512:,.0f}", fontsize=8.6, color=RED, va="center")
    ax.set_xlim(-600, 20000)
    ax.set_ylim(0, 8000)
    ax.set_xlabel("K plus V bytes per k-tile per core (4 k-tiles per k128 step: 34,816 B bfp8, 65,536 B bf16 per step)")
    ax.set_ylabel("wall cycles per k-tile on the wall core (wall / steps / 4)")
    ax.xaxis.set_major_formatter(FMT)
    ax.yaxis.set_major_formatter(FMT)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=BLUE,
            ls="",
            ms=9,
            label="MEASURED wall per k-tile; bytes DERIVED from the source geometry (filled: K/V bf16; hollow: all bf16; squares: k256, k512)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color=RED,
            ls="",
            ms=9,
            label="MEASURED, A4 reader-stub ablation (zero K/V bytes streamed)",
        ),
    ]
    h2, l2 = ax.get_legend_handles_labels()
    ax.legend(handles=handles + h2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=9)
    heading(
        fig,
        "DRAM streaming law: wall cycles per k-tile against K plus V bytes per k-tile per core (causal, 110 cores)",
        "Causal S 4096, nh 32, nkv 8, head_dim 128, q_chunk 128, k_chunk 128 (squares: k256, k512), HiFi2, exp approx, bfp8 Q and out unless stated; "
        "zones off, mean of invocations 1 and 2, 165 steps on the wall core. 1.88x the K/V bytes (bf16 K and V, 2,048 B tiles) raise the wall per k-tile 1.72x. The intercept is about 700 cycles "
        "below the 1,387 floor, so that much of the floor overlaps the stream; the A4 point (2,222) is the compute front end with a free reader.",
    )
    footer(
        fig,
        "Data: data/bh_zones/dram_rate_table.csv (base, k256, k512, A4), t23_causal_q128k128_kvbf16_zoff_runs.csv, t23_causal_q128k128_allbf16_zoff_runs.csv; law and floor: bh/zone_decomposition.md 5.2 and RT T3 (data/bh_zones/report_tables.md)",
    )
    save(fig, "a_dram_law.png")


# ---------------------------------------------------------------------------------------------
# 2. a_dram_rate_vs_config.png
# ---------------------------------------------------------------------------------------------
def fig_dram_rate():
    rate = {r["tag"]: r for r in read_csv("dram_rate_table.csv")}

    def from_table(tag):
        r = rate[tag]
        return float(r["GBps"]), float(r["step_cycles"]), int(r["bytes_per_step_core"])

    def from_runs(runs, steps, cores, bytes_per_step):
        w = wall_mean(runs)
        step = w / steps
        bpc = bytes_per_step * cores / step
        return bpc * MHZ / 1000.0, step, bytes_per_step

    cfgs = [
        ("base\nbfp8, 110 cores", from_table("t21_causal_q128k128"), 4, 165),
        ("kvbf16\nK, V bf16, Q bfp8", from_runs("t23_causal_q128k128_kvbf16_zoff_runs.csv", 165, 110, 65536), 4, 165),
        ("allbf16\nQ, K, V, out bf16", from_runs("t23_causal_q128k128_allbf16_zoff_runs.csv", 165, 110, 65536), 2, 165),
        ("nkv1\nall cores one KV head", from_runs("t23_causal_q128k128_nkv1_zoff_runs.csv", 165, 110, 34816), 4, 165),
        ("nkv32\nno GQA sharing", from_runs("t23_causal_q128k128_nkv32_zoff_runs.csv", 165, 110, 34816), 4, 165),
        ("grid8x8\n64 cores", from_runs("t23_causal_q128k128_grid8x8_zoff_runs.csv", 264, 64, 34816), 8, 264),
        ("A4b\n64 reads in flight", from_table("t22_abl_a4b_causal_q128k128"), 64, 165),
    ]
    for name, (g, step, b), thr, st in cfgs:
        print(f"rate {name.splitlines()[0]}: {g:.1f} GB/s, step {step:,.0f}, bytes {b}, in flight {thr}")
    fig, ax = plt.subplots(figsize=(W_IN, 7.2))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.79, bottom=0.2)
    xs = np.arange(len(cfgs))
    vals = [c[1][0] for c in cfgs]
    cols = [BLUE, BLUE, BLUE, BLUE, BLUE, ORANGE, RED]
    bars = ax.bar(xs, vals, width=0.72, color=cols, edgecolor=BG, linewidth=1.2)
    for x, (name, (g, step, b), thr, st), bar in zip(xs, cfgs, bars):
        ax.text(x, g + 6, f"{g:.1f} GB/s", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(
            x,
            g / 2,
            f"{b:,} B per\ncore per step\n\nstep {step:,.0f} cyc\n{st} steps\n{thr} in flight",
            ha="center",
            va="center",
            fontsize=8.2,
            color=BG,
        )
    ax.axhline(333, color=INK, lw=1.2, ls=(0, (5, 3)))
    ax.text(6.42, 333 + 5, "333 GB/s: sustained rate of the base run", ha="right", va="bottom", fontsize=9.2, color=INK)
    ax.axhline(512, color=MUTED, lw=1.2, ls=(0, (2, 3)))
    ax.text(
        6.42,
        512 + 5,
        "512 GB/s: Blackhole DRAM spec (8 x GDDR6 channels)",
        ha="right",
        va="bottom",
        fontsize=9.2,
        color=MUTED,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([c[0] for c in cfgs], fontsize=9.5)
    ax.set_ylabel("delivered chip K/V read rate (GB/s at 1.35 GHz)")
    ax.set_ylim(0, 570)
    ax.grid(axis="x", visible=False)
    handles = [
        Patch(color=BLUE, label="110 cores, 4 (or 2) reads in flight per core"),
        Patch(color=ORANGE, label="64 cores (grid 8x8, 8 reads in flight)"),
        Patch(color=RED, label="A4b barrier threshold ablation, 64 reads in flight"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
    heading(
        fig,
        "Delivered DRAM K/V read rate per configuration: bytes per step (source geometry) over the measured step time",
        "Causal S 4096, nh 32, nkv 8 unless stated, head_dim 128, q128 k128, HiFi2, exp approx; zones off, walls mean of invocations 1 and 2 (T2.3 runs 3 invocations, first discarded). "
        "Rate = bytes per core per step x active cores / step cycles, step = wall / steps on the wall core. nkv 1 and 32 equal the base within 0.1 percent; "
        "bf16 K/V (2,048 B tiles) lifts the rate to 363; 64 cores deliver 268 (neither a pure chip nor a pure per-core limit); 64 reads in flight lower it to 293. Walls MEASURED, rates INFERRED.",
    )
    footer(
        fig,
        "Data: data/bh_zones/dram_rate_table.csv (base, A4b), t23_causal_q128k128_{kvbf16,allbf16,nkv1,nkv32,grid8x8}_zoff_runs.csv; table and readings bh/zone_decomposition.md 5.2. "
        "512 GB/s is the Blackhole DRAM peak from the internal architecture brief.",
    )
    save(fig, "a_dram_rate_vs_config.png")


# ---------------------------------------------------------------------------------------------
# 3. a_causal_vs_noncausal_step.png
# ---------------------------------------------------------------------------------------------
def fig_causal_vs_noncausal():
    ca, cca = decomp_parts("t21_causal_q128k128")
    nc, cnc = decomp_parts("t21_noncausal_q128k128")
    steps_c = cca.get(("TRISC_1", "STEP"), 165.0)
    steps_n = cnc.get(("TRISC_1", "STEP"), 320.0)
    print("steps on wall core: causal", steps_c, "non-causal", steps_n)

    def g(parts, risc, names):
        return sum(parts.get((risc, n), 0.0) for n in names)

    rows = [
        ("K wait + V wait", ["K_WAIT", "V_WAIT"], "TRISC_0"),
        ("Q wait", ["Q_WAIT"], "TRISC_0"),
        ("K + V chunk read (barrier, reserve nested)", ["R_K_READ", "R_V_READ"], "NCRISC"),
        ("read barriers (inside the reads)", ["R_BARRIER"], "NCRISC"),
        ("un-zoned control (chain sems, forwarding)", ["R_CONTROL_IN_KCHUNK"], "NCRISC"),
        ("SUBEXP (sub, exp, pack, row sum)", ["SUBEXP"], "TRISC_2"),
        ("SALAD_EXP (column exp of max diff)", ["SALAD_EXP"], "TRISC_2"),
        ("REDUCE (row max)", ["REDUCE"], "TRISC_2"),
        ("MASK + MASK_DIAG (issue stall)", ["MASK", "MASK_DIAG"], "TRISC_2"),
        ("EXP_INIT (SFPU program load)", ["EXP_INIT"], "TRISC_2"),
        ("RECONFIG (12 sites per step)", ["RECONFIG"], "TRISC_1"),
        ("NORM (last chunk of a q chunk)", ["NORM"], "TRISC_1"),
    ]
    vals_c = [g(ca, risc, names) / steps_c for _, names, risc in rows]
    vals_n = [g(nc, risc, names) / steps_n for _, names, risc in rows]
    for r, vc, vn in zip(rows, vals_c, vals_n):
        print(f"  {r[0]:48s} {r[2]:8s} causal {vc:9,.1f}  noncausal {vn:9,.1f}")

    fig, ax = plt.subplots(figsize=(W_IN, 9.0))
    fig.subplots_adjust(left=0.44, right=0.975, top=0.845, bottom=0.15)
    ys = np.arange(len(rows))
    h = 0.38
    ax.barh(
        ys - h / 2,
        vals_c,
        height=h,
        color=BLUE,
        edgecolor=BG,
        label="causal: 165 steps on its wall core (step 15,531 cycles, zones off)",
    )
    ax.barh(
        ys + h / 2,
        vals_n,
        height=h,
        color=ORANGE,
        edgecolor=BG,
        label="non-causal: 320 steps on its wall core (step 9,014 cycles, zones off)",
    )
    xmax = max(max(vals_c), max(vals_n)) * 1.14
    for y, vc, vn in zip(ys, vals_c, vals_n):
        ax.text(vc + xmax * 0.006, y - h / 2, f"{vc:,.0f}", va="center", fontsize=9, color=INK)
        ax.text(vn + xmax * 0.006, y + h / 2, f"{vn:,.0f}", va="center", fontsize=9, color=INK)
    groups = [
        ("UNPACK\n(TRISC_0)", 0, 1),
        ("reader\n(NCRISC)", 2, 4),
        ("PACK (TRISC_2)", 5, 9),
        ("MATH (TRISC_1)", 10, 11),
    ]
    for gi, (lab, a, b) in enumerate(groups):
        if gi % 2 == 0:
            ax.axhspan(a - 0.5, b + 0.5, color=LIGHT, alpha=0.6, zorder=0, lw=0)
        ax.text(
            -0.69,
            (a + b) / 2,
            lab,
            transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
            ha="center",
            va="center",
            fontsize=9.6,
            color=INK,
            rotation=90,
            fontweight="bold",
        )
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.2)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(FMT)
    ax.set_xlabel("cycles per k-chunk step on the wall core (zone sum / steps, zones on)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, fontsize=9.2)
    ax.text(
        xmax * 0.99,
        6.5,
        "critical thread: causal = UNPACK, waiting on the reader's DRAM stream;\nnon-causal = PACK (the K/V chains leave about a quarter of the cores reading DRAM)",
        ha="right",
        va="center",
        fontsize=8.8,
        color=INK,
    )
    heading(
        fig,
        "Causal versus non-causal, per k-chunk step: the same compute parts, a different data path",
        "S 4096, nh 32, nkv 8, head_dim 128, q128 k128, bfp8, HiFi2, exp approx, 110 cores; wall-setting core of each run; 12 parts on their critical threads. "
        "Causal K and V waits cost 2,793 + 2,786 cycles per step against 95 + 102 for non-causal; the causal reader spends 12,426 cycles per step in read barriers against 4,034. "
        "The non-causal wall core injects on 256 of its 320 chunks and receives 64, and its reader spends 1,853 cycles per step in un-zoned chain control. "
        "MEASURED zone sums; the thread attribution is INFERRED from the waits and the ablations.",
    )
    footer(
        fig,
        "Data: data/bh_zones/decomp_t21_causal_q128k128.csv and decomp_t21_noncausal_q128k128.csv (RT T5 and T5b in data/bh_zones/report_tables.md); bh/zone_decomposition.md 4.5. "
        "Reader zones nest: R_BARRIER and R_RESERVE sit inside R_K_READ and R_V_READ.",
    )
    save(fig, "a_causal_vs_noncausal_step.png")


# ---------------------------------------------------------------------------------------------
# 4. a_chain_roles.png
# ---------------------------------------------------------------------------------------------
def grid_matrix(d, fn=lambda v: v):
    m = np.full((len(GRID_Y), len(GRID_X)), np.nan)
    for (x, y), v in d.items():
        m[GRID_Y.index(y), GRID_X.index(x)] = fn(v)
    return m


def style_grid_axes(ax):
    ax.set_xticks(range(len(GRID_X)))
    ax.set_xticklabels(GRID_X, fontsize=8.5)
    ax.set_yticks(range(len(GRID_Y)))
    ax.set_yticklabels(GRID_Y, fontsize=8.5)
    ax.set_xlabel("core_x (profiler, physical NoC coordinate)", fontsize=9)
    ax.set_ylabel("core_y", fontsize=9)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_chain_roles():
    nc = reader_counts("t21_noncausal_q128k128_zon_cores.csv")
    ca = reader_counts("t21_causal_q128k128_zon_cores.csv")
    assert (
        reader_counts("t21_noncausal_q128k128_zon_cores.csv", run_idx="2") == nc
    ), "chain roles differ between invocations"
    cmap = LinearSegmentedColormap.from_list("roles", [BG, BLUE], N=256)
    n_inj = sum(1 for v in nc.values() if v[0] > 0)
    n_rec = sum(1 for v in nc.values() if v[0] == 0)
    tot_nc = sum(v[0] for v in nc.values())
    tot_all = sum(v[1] for v in nc.values())
    tot_ca = sum(v[0] for v in ca.values())
    notes = {
        "nc": f"{n_inj} cores read K from DRAM on some or all of their chunks (chain injectors),\n{n_rec} receive everything over the NoC. Chip: {tot_nc:,.0f} of {tot_all:,.0f} k chunks from DRAM ({100 * tot_nc / tot_all:.0f} percent).",
        "ca": f"72 heavy cores read 165 chunks (5 pairs x 33), 38 light cores 132 (4 pairs x 33):\n{tot_ca:,.0f} DRAM chunk reads, 100 percent; NoC incoming counter zero (RT T7).",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_IN, 8.0), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.25))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.745, bottom=0.14)
    for ax, d, title, note in (
        (ax1, nc, "non-causal q128 k128: K chunks read from DRAM per core", notes["nc"]),
        (ax2, ca, "causal q128 k128: every K chunk is read from DRAM", notes["ca"]),
    ):
        frac = grid_matrix(d, lambda v: 100.0 * v[0] / v[1])
        ax.imshow(frac, cmap=cmap, vmin=0, vmax=100, aspect="equal")
        for (x, y), (n, tot) in d.items():
            i, j = GRID_Y.index(y), GRID_X.index(x)
            f = n / tot
            ax.text(
                j,
                i - 0.13,
                f"{n:.0f}",
                ha="center",
                va="center",
                fontsize=7.6,
                color=BG if f > 0.55 else INK,
                fontweight="bold" if n > 0 else "normal",
            )
            ax.text(
                j, i + 0.25, f"of {tot:.0f}", ha="center", va="center", fontsize=5.8, color=BG if f > 0.55 else MUTED
            )
        ax.set_title(title, fontsize=10.5, loc="left", pad=34)
        ax.text(0, 1.015, note, transform=ax.transAxes, fontsize=8.0, color=INK, va="bottom", ha="left")
        style_grid_axes(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 100))
    cb = fig.colorbar(sm, ax=[ax1, ax2], orientation="horizontal", fraction=0.035, pad=0.13, aspect=50)
    cb.set_label(
        "percent of the core's k chunks whose K tile block this core read from DRAM (R_K_READ_N / R_KCHUNK_N)",
        fontsize=9,
    )
    heading(
        fig,
        "Chain roles per core: reader K-read occurrences (R_K_READ_N) on the non-causal anchor against the causal anchor",
        "S 4096, nh 32, nkv 8, head_dim 128, q128 k128, bfp8, HiFi2, 110 cores, zones on, invocation 1 (invocation 2 identical). Cell text: DRAM K-chunk reads of the total k chunks on that core "
        "(320 or 288 non-causal, 165 or 132 causal). Non-causal: 83 chain receivers read nothing from DRAM, 27 injectors read 32 to 256 of their chunks; the reader's wall core injects on 256 of 320. "
        "Causal: every step reads from DRAM. MEASURED counts.",
    )
    footer(
        fig,
        "Data: data/bh_zones/t21_noncausal_q128k128_zon_cores.csv and t21_causal_q128k128_zon_cores.csv (NCRISC rows, R_K_READ_N, R_KCHUNK_N); bh/zone_decomposition.md 4.5 and 4.6 (RT T7 L1_0_NOC_RING0_INCOMING).",
    )
    save(fig, "a_chain_roles.png")


# ---------------------------------------------------------------------------------------------
# 5. a_span_histogram.png
# ---------------------------------------------------------------------------------------------
def fig_span_histogram():
    cases = [
        (
            "anchor: causal q128 k128 (512 pairs, 33 steps per pair)",
            "t21_causal_q128k128_zoff",
            "t21_causal_q128k128_zon",
            33,
            46,
        ),
        (
            "causal q256 k128 (256 pairs, 34 steps per pair)",
            "t21_causal_q256k128_zoff",
            "t21_causal_q256k128_zon",
            34,
            62,
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, 6.8), gridspec_kw=dict(wspace=0.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.79, bottom=0.14)
    for ax, (title, zoff, zon, steps_per_pair, ymax) in zip(axes, cases):
        spans = trisc1_spans(f"{zoff}_cores.csv")
        pairs = {k: v[0] / steps_per_pair for k, v in reader_counts(f"{zon}_cores.csv").items()}
        wall = wall_mean(f"{zoff}_runs.csv")
        pvals = sorted(set(round(p) for p in pairs.values()))
        assert len(pvals) == 2, pvals
        heavy, light = max(pvals), min(pvals)
        s_heavy = [spans[k] / 1e6 for k in spans if round(pairs[k]) == heavy]
        s_light = [spans[k] / 1e6 for k in spans if round(pairs[k]) == light]
        print(
            f"{zoff}: heavy {len(s_heavy)} cores x {heavy} pairs, light {len(s_light)} x {light}; wall {wall:,.0f}; "
            f"heavy span range {min(s_heavy):.3f}-{max(s_heavy):.3f} M, light {min(s_light):.3f}-{max(s_light):.3f} M"
        )
        bins = np.arange(1.0, 2.75, 0.04)
        ax.hist(
            [s_heavy, s_light],
            bins=bins,
            stacked=True,
            color=[BLUE, ORANGE],
            edgecolor=BG,
            linewidth=0.6,
            label=[
                f"{len(s_heavy)} heavy cores: {heavy} pairs = {heavy * 2} q chunks = {heavy * steps_per_pair} steps",
                f"{len(s_light)} light cores: {light} pairs = {light * 2} q chunks = {light * steps_per_pair} steps",
            ],
        )
        ax.axvline(wall / 1e6, ymin=0, ymax=0.74, color=INK, lw=1.4, ls=(0, (5, 3)))
        ax.set_ylim(0, ymax)
        ax.text(
            wall / 1e6 + 0.12,
            ymax * 0.755,
            f"device wall\n{wall:,.0f} cycles",
            ha="right",
            va="bottom",
            fontsize=9,
            color=INK,
        )
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.set_xlabel("per-core TRISC_1 KERNEL span (million cycles)")
        ax.set_ylabel("cores")
        ax.set_xlim(1.0, 2.75)
        ax.legend(loc="upper left" if "q256" not in zoff else "upper right", fontsize=9)
        if "q256" not in zoff:
            ax.annotate(
                "light cores finish at 1.17 to 1.72 M, well under 132 x 15,531 = 2.05 M:\nthe per-step time differs across cores (row gradient, see a_grid_and_pairs)",
                xy=(1.45, 5.5),
                xytext=(1.03, 26),
                fontsize=8.6,
                color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.7),
            )
    heading(
        fig,
        "Per-core TRISC_1 span histograms: the heavy and light pair populations and the device wall",
        "S 4096, nh 32, nkv 8, head_dim 128, k128, bfp8, HiFi2, exp approx, 110 cores, zones off, spans mean of invocations 1 and 2. Population membership from the zones-on twin run "
        "(R_K_READ_N / steps per pair). The factory pair-distributes q chunks (chunk i with chunk Q-1-i, equal work per pair); 512 pairs over 110 cores leave 72 cores with 5 pairs, "
        "256 pairs leave 36 with 3. The wall is set by a heavy core in both cases. MEASURED spans and walls.",
    )
    footer(
        fig,
        "Data: data/bh_zones/t21_causal_q128k128_zoff_cores.csv, t21_causal_q256k128_zoff_cores.csv (TRISC_1 kernel_dur), *_zoff_runs.csv (wall_dev_cycles), pair counts from "
        "t21_causal_{q128k128,q256k128}_zon_cores.csv (R_K_READ_N); pair rule bh/targets.md section 1.",
    )
    save(fig, "a_span_histogram.png")


# ---------------------------------------------------------------------------------------------
# 6. a_grid_and_pairs.png
# ---------------------------------------------------------------------------------------------
def fig_grid_and_pairs():
    spans = trisc1_spans("t21_causal_q128k128_zoff_cores.csv")
    pairs = {k: round(v[0] / 33) for k, v in reader_counts("t21_causal_q128k128_zon_cores.csv").items()}
    wall = wall_mean("t21_causal_q128k128_zoff_runs.csv")
    wc1, wc2 = wall_core("t21_causal_q128k128_zoff_runs.csv", "1"), wall_core("t21_causal_q128k128_zoff_runs.csv", "2")
    n_heavy = sum(1 for p in pairs.values() if p == 5)
    print("heavy cores", n_heavy, "light", 110 - n_heavy, "wall cores", wc1, wc2)
    cmap = LinearSegmentedColormap.from_list("span", ["#dfe9f6", BLUE, "#0d2f5e"], N=256)
    vmin, vmax = 1.1, 2.6
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_IN, 8.0), gridspec_kw=dict(width_ratios=[1.0, 1.25], wspace=0.2))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.76, bottom=0.2)
    m = grid_matrix(spans, lambda v: v / 1e6)
    im = ax1.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    for (x, y), s in spans.items():
        i, j = GRID_Y.index(y), GRID_X.index(x)
        p = pairs[(x, y)]
        col = BG if s / 1e6 > 1.9 else INK
        ax1.text(j, i - 0.12, f"{p}p", ha="center", va="center", fontsize=8.4, color=col, fontweight="bold")
        ax1.text(j, i + 0.27, f"{s / 1e6:.2f}", ha="center", va="center", fontsize=6.4, color=col)
        if p == 5:
            ax1.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=ORANGE, lw=1.6))
    for wc in (wc1, wc2):
        i, j = GRID_Y.index(wc[1]), GRID_X.index(wc[0])
        ax1.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=RED, lw=2.6))
    style_grid_axes(ax1)
    ax1.set_title("pairs per core (text, orange frame = 5) over the span (fill)", fontsize=10.2, loc="left")
    cb = fig.colorbar(im, ax=ax1, orientation="horizontal", fraction=0.04, pad=0.1)
    cb.set_label("TRISC_1 span, million cycles", fontsize=9)
    order = [(x, y) for y in GRID_Y for x in GRID_X]
    idx = np.arange(len(order))
    vals = np.array([spans[k] / 1e6 for k in order])
    cols = [BLUE if pairs[k] == 5 else ORANGE for k in order]
    ax2.bar(idx, vals, width=0.85, color=cols, edgecolor="none")
    for ref, lab in (
        (165 * 15531 / 1e6, "165 steps x 15,531 (wall-core step) = 2.56 M"),
        (132 * 15531 / 1e6, "132 steps x 15,531 = 2.05 M"),
    ):
        ax2.axhline(ref, color=INK, lw=1.0, ls=(0, (5, 3)))
        ax2.text(109, ref + 0.025, lab, fontsize=8.8, color=INK, va="bottom", ha="right")
    for r, y in enumerate(GRID_Y):
        ax2.axvline(r * len(GRID_X) - 0.5, color=GRAY, lw=0.6)
        ax2.text(
            r * len(GRID_X) + len(GRID_X) / 2 - 0.5, 2.7, f"y={y}", ha="center", va="bottom", fontsize=8, color=MUTED
        )
    ax2.set_xlim(-0.5, len(order) - 0.5)
    ax2.set_ylim(0, 2.85)
    ax2.set_xlabel("core index in row-major profiler order (rows of 11 cores)")
    ax2.set_ylabel("TRISC_1 span (million cycles)")
    ax2.grid(axis="x", visible=False)
    ax2.set_title(
        f"{n_heavy} cores x 5 pairs, then {110 - n_heavy} x 4; spans fall with core_y", fontsize=10.2, loc="left"
    )
    ax2.legend(
        handles=[
            Patch(color=BLUE, label="5 pairs (10 q chunks, 165 steps)"),
            Patch(color=ORANGE, label="4 pairs (8 q chunks, 132 steps)"),
            Line2D([], [], color=RED, lw=2.6, label="wall-setting core (left panel)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        fontsize=9,
    )
    heading(
        fig,
        "The 110-core grid at the anchor: causal pair distribution (72 cores with 5 pairs, 38 with 4) and the per-core TRISC_1 span",
        "Causal S 4096, nh 32, nkv 8, head_dim 128, q128 k128, bfp8, HiFi2, exp approx, zones off. Pair rule (sdpa_program_factory.cpp global_q_pair_distribute): 1,024 q chunks = 512 pairs (i, 31 - i), "
        "each pair 33 k-chunk steps; the heaviest core gets (512 // 110) x 2 + 2 = 10 chunks. Pair counts MEASURED from the zones-on twin (R_K_READ_N / 33: 165 or 132); spans and wall MEASURED. "
        f"The device wall ({wall:,.0f} cycles) is set by a heavy core in core_y 3 or 4; heavy cores in lower rows finish earlier (2.05 to 2.42 M), light cores at 1.17 to 1.72 M.",
    )
    footer(
        fig,
        "Data: data/bh_zones/t21_causal_q128k128_zoff_cores.csv (TRISC_1 kernel_dur), t21_causal_q128k128_zoff_runs.csv (wall, wall core), t21_causal_q128k128_zon_cores.csv (R_K_READ_N); "
        "pair rule bh/targets.md sections 1 and 2 (Q_wallcore_kernel 10, 72 heavy cores).",
    )
    save(fig, "a_grid_and_pairs.png")


# ---------------------------------------------------------------------------------------------
# 7. a_step_timeline.png
# ---------------------------------------------------------------------------------------------
CAT_COL = {
    "wait": BLUE,
    "matmul": ORANGE,
    "softmax": AQUA,
    "frontend": RED,
    "control": GRAY,
    "sync": "#8f8f8b",
    "write": "#c9c9c4",
}
COMPUTE_ORDER = [  # one box per zone at its first program position, with category
    ("EXP_INIT", "frontend"),
    ("RESERVE_QKT", "sync"),
    ("K_WAIT", "wait"),
    ("Q_WAIT", "wait"),
    ("RECONFIG", "frontend"),
    ("QK_MM", "matmul"),
    ("SUBEXP", "softmax"),
    ("MASK", "frontend"),
    ("PUSH_HOLD", "sync"),
    ("REDUCE", "softmax"),
    ("POPS", "sync"),
    ("OUT_RESERVE", "sync"),
    ("QKTIM_WAIT", "wait"),
    ("V_WAIT", "wait"),
    ("PV_MM", "matmul"),
    ("PACK_DONE", "sync"),
    ("SALAD_EXP", "softmax"),
    ("SALAD_CORR", "softmax"),
    ("NORM", "softmax"),
    ("PUSHES", "sync"),
    ("UNZONED_IN_STEP", "control"),
    ("ZONE_TAX_IN_STEP", "control"),
]
SHORT = {
    "UNZONED_IN_STEP": "UNZONED",
    "ZONE_TAX_IN_STEP": "ZONE TAX",
    "SALAD_CORR": "SALAD_CORR",
    "SALAD_EXP": "SALAD_EXP",
}


def fig_step_timeline():
    parts, counts = decomp_parts("t21_causal_q128k128")
    steps = counts.get(("TRISC_1", "STEP"), 165.0)

    def per_step(risc, name):
        if name == "MASK":
            return (parts.get((risc, "MASK"), 0) + parts.get((risc, "MASK_DIAG"), 0)) / steps
        return parts.get((risc, name), 0.0) / steps

    rk, rq, rv = (parts[("NCRISC", n)] / steps for n in ("R_K_READ", "R_Q_READ", "R_V_READ"))
    rkc = parts[("NCRISC", "R_KCHUNK")] / steps
    rctl = rkc - rk - rq - rv
    rbar, rres = parts[("NCRISC", "R_BARRIER")] / steps, parts[("NCRISC", "R_RESERVE")] / steps
    ww, wwr = parts[("BRISC", "W_WAIT")] / steps, parts.get(("BRISC", "W_WRITE"), 0.0) / steps
    lanes = [
        (
            "reader\nNCRISC",
            4.55,
            [
                ("R_K_READ", rk, "wait"),
                ("R_Q_READ", rq, "wait"),
                ("R_V_READ", rv, "wait"),
                ("control", rctl, "control"),
            ],
        )
    ]
    for risc, lab, y in (
        ("TRISC_0", "UNPACK\nTRISC_0", 3.0),
        ("TRISC_1", "MATH\nTRISC_1", 2.0),
        ("TRISC_2", "PACK\nTRISC_2", 1.0),
    ):
        lanes.append((lab, y, [(n, per_step(risc, n), c) for n, c in COMPUTE_ORDER]))
    lanes.append(
        (
            "writer\nBRISC",
            0.0,
            [
                ("W_WAIT (cb_out wait_front: the writer does nothing per k chunk)", ww, "write"),
                ("W_WRITE", wwr, "sync"),
            ],
        )
    )
    for lab, _, boxes in lanes:
        print(
            f"  lane {lab.splitlines()[0]:7s} sum {sum(b[1] for b in boxes):9,.0f}: "
            + ", ".join(f"{n} {v:,.0f}" for n, v, c in boxes if v > 300)
        )

    fig, ax = plt.subplots(figsize=(W_IN, 9.9))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.81, bottom=0.225)
    LH = 0.56
    xmax = max(sum(b[1] for b in boxes) for _, _, boxes in lanes)
    box_pos = {}
    for lab, y, boxes in lanes:
        x = 0.0
        for n, v, c in boxes:
            dark = c in ("wait", "matmul", "softmax", "frontend")
            ax.add_patch(
                Rectangle(
                    (x, y - LH / 2),
                    v,
                    LH,
                    facecolor=CAT_COL[c],
                    edgecolor=BG,
                    linewidth=0.7,
                    hatch="////" if n.startswith("ZONE") else None,
                )
            )
            box_pos[(lab, n)] = (x, x + v, y)
            frac = v / xmax
            nm = SHORT.get(n, n)
            if frac >= 0.055:
                ax.text(
                    x + v / 2, y, f"{nm}\n{v:,.0f}", ha="center", va="center", fontsize=8.6, color=BG if dark else INK
                )
            elif frac >= 0.02:
                ax.text(
                    x + v / 2,
                    y,
                    f"{nm} {v:,.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.4,
                    rotation=90,
                    color=BG if dark else INK,
                )
            x += v
        ax.text(-150, y, lab, ha="right", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 160, y, f"sum\n{x:,.0f}", ha="left", va="center", fontsize=8.5, color=MUTED)
    # nested reader strip (barrier, reserve, issue+push sit inside the three reads)
    yr = lanes[0][1]
    ys0 = yr - LH / 2 - 0.2
    x0 = 0
    for n, v, c in (
        ("R_BARRIER (nested in the reads)", rbar, "wait"),
        ("R_RESERVE", rres, "sync"),
        ("issue + push", rk + rq + rv - rbar - rres, "control"),
    ):
        ax.add_patch(Rectangle((x0, ys0), v, 0.16, facecolor=CAT_COL[c], edgecolor=BG, linewidth=0.5, alpha=0.75))
        if v > 3000:
            ax.text(x0 + v / 2, ys0 + 0.08, f"{n} {v:,.0f}", ha="center", va="center", fontsize=7.8, color=INK)
        x0 += v
    ax.text(
        rk + rq + rv,
        ys0 - 0.05,
        f"then R_RESERVE {rres:,.0f} and issue + push {rk + rq + rv - rbar - rres:,.0f} (derived); positions schematic",
        ha="right",
        va="top",
        fontsize=7.6,
        color=MUTED,
    )
    ul = "UNPACK\nTRISC_0"

    def arrow(src, dst, label, col, rad, lab_xy):
        (sx0, sx1, sy), (dx0, dx1, dy) = box_pos[src], box_pos[dst]
        down = sy > dy
        ax.add_patch(
            FancyArrowPatch(
                (sx1, sy - LH / 2 if down else sy + LH / 2),
                ((dx0 + dx1) / 2, dy + LH / 2 if down else dy - LH / 2),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=13,
                lw=1.3,
                color=col,
                zorder=8,
            )
        )
        ax.text(
            lab_xy[0],
            lab_xy[1],
            label,
            fontsize=8.2,
            color=col,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=BG, edgecolor="none", alpha=0.9),
            zorder=9,
        )

    arrow(("reader\nNCRISC", "R_K_READ"), (ul, "K_WAIT"), "K chunk pushed (cb_kt_in)", INK, -0.2, (2900, 3.62))
    arrow(
        ("reader\nNCRISC", "R_Q_READ"),
        (ul, "Q_WAIT"),
        "Q subblocks pushed (first k chunk of a q chunk)",
        INK,
        0.15,
        (6200, 3.78),
    )
    arrow(("reader\nNCRISC", "R_V_READ"), (ul, "V_WAIT"), "V chunk pushed (cb_v_in)", INK, 0.2, (12300, 3.62))
    arrow(("PACK\nTRISC_2", "QK_MM"), (ul, "QKTIM_WAIT"), "qkt_im packed by PACK (cb_qkt_im)", RED, 0.3, (5000, 2.55))
    order_txt = (
        "phase 1: "
        + " > ".join(n for n, _ in COMPUTE_ORDER[:11])
        + "\nphase 2: "
        + " > ".join(n for n, _ in COMPUTE_ORDER[11:20])
        + "  |  then "
        + " > ".join(n for n, _ in COMPUTE_ORDER[20:])
    )
    ax.text(
        0,
        5.05,
        "compute program order (one box per zone at its first position; RECONFIG has 12 sites, SUBEXP 2 calls; PV_MM, QKTIM_WAIT and SALAD_EXP recur):\n"
        + order_txt,
        fontsize=7.4,
        color=MUTED,
        va="bottom",
    )
    ax.set_xlim(-1900, xmax * 1.09)
    ax.set_ylim(-0.55, 5.85)
    ax.set_yticks([])
    ax.set_xlabel(
        "cycles within one k-chunk step, wall core, per-step means (zone sum / 165 steps; the step envelope is 15,531 cycles zones off)"
    )
    ax.xaxis.set_major_formatter(FMT)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    handles = [
        Patch(color=CAT_COL["wait"], label="waits on data (cb_wait_front, reads, barriers)"),
        Patch(color=CAT_COL["matmul"], label="matmul zones (QK_MM, PV_MM)"),
        Patch(color=CAT_COL["softmax"], label="softmax and rescale (SUBEXP, REDUCE, SALAD_*, NORM)"),
        Patch(color=CAT_COL["frontend"], label="front end (EXP_INIT, RECONFIG, MASK)"),
        Patch(color=CAT_COL["sync"], label="small sync (reserves, pops, pushes, pack_done)"),
        Patch(facecolor=CAT_COL["control"], hatch="////", edgecolor=BG, label="un-zoned control; zone tax (hatched)"),
        Line2D([], [], color=INK, lw=1.3, label="dependency: reader push to UNPACK wait"),
        Line2D([], [], color=RED, lw=1.3, label="dependency: PACK to UNPACK (qkt_im)"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8.6)
    heading(
        fig,
        "One k-chunk step as a swimlane per thread: zones in program order, widths proportional to the anchor wall-core per-step zone sums",
        "Causal S 4096, nh 32, nkv 8, head_dim 128, q128 k128, bfp8, HiFi2, exp approx, 110 cores; wall core (2,3), zones on, mean of invocations 1 and 2, divided by 165 steps. "
        "Each thread's zones tile its own span (within 0.1 percent): UNPACK spends 5,937 cycles per step waiting for K, V and Q, MATH 9,814 inside the two matmuls, PACK 6,558 in "
        "MASK and EXP_INIT issue stalls. Arrows are the four data dependencies of the step. MEASURED zone sums; one box per zone is a schematic of program order, not a timing trace.",
    )
    footer(
        fig,
        "Data: data/bh_zones/decomp_t21_causal_q128k128.csv (RT T2 and T2b in data/bh_zones/report_tables.md); program order bh/sdpa_kernel_phase_map.md 1.1 to 1.3 and bh/zone_decomposition.md 2.2 (zone map).",
    )
    save(fig, "a_step_timeline.png")


# ---------------------------------------------------------------------------------------------
# 8. a_method_zones.png (schematic)
# ---------------------------------------------------------------------------------------------
COMPUTE_SLOTS = [  # (slot, name, meaningful threads: U unpack, M math, P pack)
    (21, "EXP_INIT", "P"),
    (3, "RESERVE_QKT", "P"),
    (1, "K_WAIT", "U"),
    (2, "Q_WAIT", "U"),
    (4, "RECONFIG", "UMP"),
    (5, "SUBEXP", "UMP"),
    (6, "EXP (in SUBEXP)", "P"),
    (7, "QK_MM", "UMP"),
    (8, "MASK", "PU"),
    (20, "MASK_DIAG", "PU"),
    (18, "PUSH_HOLD", "P"),
    (9, "REDUCE", "UMP"),
    (19, "POPS", "U"),
    (10, "OUT_RESERVE", "P"),
    (11, "QKTIM_WAIT", "U"),
    (12, "V_WAIT", "U"),
    (13, "PV_MM", "UMP"),
    (14, "PACK_DONE", "U"),
    (17, "NORM", "UMP"),
    (16, "SALAD_CORR", "UMP"),
    (15, "SALAD_EXP", "P"),
    (22, "PUSHES", "P"),
]


def fig_method_zones():
    fig, ax = plt.subplots(figsize=(W_IN, 11.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.835, bottom=0.075)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    cols = [
        ("BRISC", "writer", "#c9c9c4"),
        ("NCRISC", "reader", BLUE),
        ("TRISC_0", "UNPACK", AQUA),
        ("TRISC_1", "MATH", ORANGE),
        ("TRISC_2", "PACK", RED),
    ]
    cw, gap, x0 = 18.4, 1.6, 1.0
    key = {"TRISC_0": "U", "TRISC_1": "M", "TRISC_2": "P"}

    def rbox(x, y, w, h, ec, lw=1.4, fc=BG, ls="-"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.2,rounding_size=0.8",
                linewidth=lw,
                edgecolor=ec,
                facecolor=fc,
                linestyle=ls,
            )
        )

    for ci, (risc, role, col) in enumerate(cols):
        x = x0 + ci * (cw + gap)
        ax.text(x + cw / 2, 99.5, risc, ha="center", va="top", fontsize=11.5, fontweight="bold", color=INK)
        ax.text(x + cw / 2, 96.4, role, ha="center", va="top", fontsize=10, color=col if col != "#c9c9c4" else MUTED)
        rbox(x, 2, cw, 91.5, INK, 1.6)
        ax.text(x + 0.8, 92.6, "KERNEL (native profiler zone)", fontsize=8.2, color=INK, va="top")
        ax.text(x + 0.8, 90.0, "INIT_PRE_FIRST_QCHUNK", fontsize=7.6, color=MUTED, va="top")
        ax.text(x + 0.8, 5.4, "TAIL_AFTER_LAST_QCHUNK", fontsize=7.6, color=MUTED, va="bottom")
        qname = {"BRISC": "W_QCHUNK", "NCRISC": "R_QCHUNK"}.get(risc, "QCHUNK")
        rbox(x + 0.9, 8.5, cw - 1.8, 78.5, MUTED, 1.2, ls=(0, (4, 2)))
        ax.text(
            x + 1.6,
            86.4,
            f"{qname} raw zone, one per q chunk\n(DeviceZoneScopedN)",
            fontsize=6.8,
            color=MUTED,
            va="top",
        )
        if risc == "BRISC":
            rbox(x + 1.8, 14, cw - 3.6, 62, "#8f8f8b", 1.2)
            ax.text(
                x + cw / 2, 73.5, "W_DRAIN  slot 1", ha="center", va="top", fontsize=8.6, fontweight="bold", color=INK
            )
            ax.text(x + cw / 2, 70.0, "write_block_row_grouped", ha="center", va="top", fontsize=7.4, color=MUTED)
            rbox(x + 2.8, 20, cw - 5.6, 44, "#8f8f8b", 1.0)
            ax.text(
                x + cw / 2, 61.5, "W_WAIT  slot 0", ha="center", va="top", fontsize=8.6, fontweight="bold", color=INK
            )
            ax.text(
                x + cw / 2,
                57.5,
                "cb.wait_front on cb_out\n(99.96 percent of the\nkernel at the anchor)\n\nno per k chunk work:\nthe writer runs only on\nthe last k chunk of a\nq chunk (2 x wait, 8\nwrites, flush, pop)",
                ha="center",
                va="top",
                fontsize=7.4,
                color=INK,
            )
        elif risc == "NCRISC":
            rbox(x + 1.8, 12, cw - 3.6, 66, col, 1.3)
            ax.text(
                x + cw / 2, 77.0, "R_KCHUNK  slot 5", ha="center", va="top", fontsize=8.6, fontweight="bold", color=INK
            )
            ax.text(x + cw / 2, 73.8, "k loop body (one per k chunk)", ha="center", va="top", fontsize=7.4, color=MUTED)
            for yy, name, sub in (
                (56, "R_K_READ  slot 0", "read_chunk_with_padding K"),
                (39, "R_Q_READ  slot 2", "Q subblock loop (first k chunk)"),
                (22, "R_V_READ  slot 1", "read_chunk_with_padding V"),
            ):
                rbox(x + 2.8, yy, cw - 5.6, 15.5, col, 1.0)
                ax.text(x + cw / 2, yy + 14.6, name, ha="center", va="top", fontsize=8.2, fontweight="bold", color=INK)
                ax.text(x + cw / 2, yy + 11.6, sub, ha="center", va="top", fontsize=6.6, color=MUTED)
                for k, nm in enumerate(("R_RESERVE  slot 3", "R_BARRIER  slot 4")):
                    yb = yy + 5.2 - k * 4.3
                    rbox(x + 3.8, yb, cw - 7.6, 3.4, MUTED, 0.8)
                    ax.text(x + cw / 2, yb + 1.7, nm, ha="center", va="center", fontsize=6.4, color=INK)
            ax.text(
                x + cw / 2,
                20.4,
                "un-zoned: chain semaphores,\nforwarding (non-causal), address gen\nderived: R_ISSUE_AND_PUSH =\nreads - barrier - reserve",
                ha="center",
                va="top",
                fontsize=6.4,
                color=MUTED,
            )
        else:
            rbox(x + 1.8, 10.5, cw - 3.6, 68.5, col, 1.3)
            ax.text(x + cw / 2, 78.0, "STEP  slot 0", ha="center", va="top", fontsize=8.6, fontweight="bold", color=INK)
            ax.text(
                x + cw / 2, 74.9, "sdpa_inner_loop_step, one k chunk", ha="center", va="top", fontsize=6.6, color=MUTED
            )
            yy = 72.0
            for slot, name, thr in COMPUTE_SLOTS:
                mine = key[risc] in thr
                ax.text(
                    x + 3.0,
                    yy,
                    f"{slot:>2}",
                    ha="left",
                    va="top",
                    fontsize=7.3,
                    color=INK if mine else GRAY,
                    family="DejaVu Sans Mono",
                )
                ax.text(
                    x + 5.6,
                    yy,
                    name,
                    ha="left",
                    va="top",
                    fontsize=7.6,
                    color=INK if mine else GRAY,
                    fontweight="bold" if mine else "normal",
                )
                yy -= 2.55
            ax.text(
                x + cw / 2,
                13.6,
                "plus UNZONED_IN_STEP and\nZONE_TAX_IN_STEP (derived)",
                ha="center",
                va="top",
                fontsize=7.0,
                color=MUTED,
            )
    fig.text(
        0.5,
        0.048,
        "\n".join(
            textwrap.wrap(
                "Compute columns: bold ink = the thread on which the zone's time is meaningful (UNPACK: cb_wait_front spins; PACK: STALLWAIT and SFPU issue; MATH: own view; all three for matmul, "
                "reduce, SALAD, NORM, RECONFIG); grey = recorded on this thread but not attributed. Slot numbers are the SDPA_ZACC indices (24 per RISC, flushed at kernel end as NAME and NAME_N).",
                170,
            )
        ),
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=INK,
    )
    heading(
        fig,
        "Zone method: the five RISCs of one Tensix core, the accumulate zone slots per thread, and the KERNEL / QCHUNK / STEP nesting (schematic, not to scale)",
        "Accumulate zones (SDPA_ZACC, RAII wall-clock read at entry and exit, sums and counts per slot, 26 cycles per occurrence, nothing written to L1 until kernel end) inside one raw zone "
        "per q chunk (DeviceZoneScopedN, 52 to 57 cycles per open and close) inside the native KERNEL zone. Accounting: KERNEL = INIT + sum(QCHUNK) + BETWEEN + TAIL; QCHUNK = sum(STEP) + "
        "OUTSIDE_STEP; STEP = 19 inner parts + EXP_INIT + PUSHES + zone tax + un-zoned. The compute source runs on all three TRISCs, so every compute zone yields three records per core.",
        y=0.985,
    )
    footer(
        fig,
        "Source: bh/zone_decomposition.md 2.1 (instrumentation, zone tax table from data/bh_zones/t02_zone_tax_summary.csv), 2.2 (zone map with CS/RD/WR/DF line numbers), 2.3 (accounting); bh/zone_patch.diff.",
    )
    save(fig, "a_method_zones.png")


if __name__ == "__main__":
    fig_dram_law()
    fig_dram_rate()
    fig_causal_vs_noncausal()
    fig_chain_roles()
    fig_span_histogram()
    fig_grid_and_pairs()
    fig_step_timeline()
    fig_method_zones()
