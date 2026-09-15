#!/usr/bin/env python
"""TopK model fit figures (tm_ prefix) for handoff/revamp/figs.

Inputs (read-only): data/topk/topk_model_acceptance.csv (written by topk/topk_model_acceptance.py),
data/topk/results_L1_L2_L3_L4_L5.csv (main-tip R=1 rows) and the model itself (ttsim/perf/roofline_topk.py
in the checkout named by POLARIS, default the polaris checkout on branch mvlahovic/sdpa_revamp).
Outputs: figs/tm_signed_errors.png, figs/tm_li_law_main.png; run with figure names to make a subset.
Interpreter: $POLARIS/.venv/bin/python.
"""
import csv
import json
import math
import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Paths per PORTABLE_CONTRACT.md: ROOT is handoff/revamp (the directory above this file's),
# WORK the SDPA root above that; both are environment overrides.
ROOT = os.environ.get("HANDOFF", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORK = os.environ.get("SDPA_WORK", os.path.dirname(os.path.dirname(ROOT)))
OUT = os.path.join(ROOT, "figs")
sys.path.insert(0, os.environ.get("POLARIS", os.path.join(WORK, "polaris")))
from ttsim.perf import roofline_topk as rt  # noqa: E402

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, BG = "#0b0b0b", "#fcfcfb"
GRAY, MUTED = "#b9b9b5", "#5c5c58"
GHZ = 1.35

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
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.color": "#e4e4e0",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.family": "DejaVu Sans",
    }
)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------------------------------
# tm_signed_errors.png
# ----------------------------------------------------------------------------------------------

REGIME_ORDER = [
    ("large_indices", "topk_large_indices\n(direct)", BLUE),
    ("composite", "composite\n(prep + li + finish)", AQUA),
    ("generic_single_core", "stock single core", ORANGE),
    ("generic_multi_core", "stock multi core", RED),
    ("gate", "gates and\nmoe_grouped_topk", MUTED),
    ("indexer", "indexer_\nscore_dsa", INK),
]
MIN_GROUP = 58
GAP = 6


def fig_signed_errors():
    with open(os.path.join(ROOT, "data", "topk", "topk_model_acceptance.csv")) as fh:
        rows = [r for r in csv.DictReader(l for l in fh if not l.startswith("#"))]
    for r in rows:
        r["err"] = float(r["err_pct"])
        r["meas"] = float(r["measured_ns"])
    # one point per (cell, op); composite totals and call totals are kept, they are what a caller sees
    fig, ax = plt.subplots(figsize=(1600 / 150, 820 / 150))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.15)
    ax.axhspan(-10, 10, color="#e9f1fb", zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    for y in (-10, 10):
        ax.axhline(y, color=BLUE, lw=0.8, ls="--")
    x0 = 0
    ticks, labels = [], []
    n_total = n_in = 0
    worst = None
    for regime, label, color in REGIME_ORDER:
        sel = sorted([r for r in rows if r["regime"] == regime], key=lambda r: r["meas"])
        if not sel:
            continue
        xs = [x for x, _ in _group_positions(x0, len(sel))]
        for x, r in zip(xs, sel):
            sub = r["subset"]
            n_total += 1
            n_in += abs(r["err"]) <= 10
            if worst is None or abs(r["err"]) > abs(worst[1]["err"]):
                worst = (x, r)
            if sub == "holdout":
                ax.plot(x, r["err"], marker="o", ms=6.5, mfc=BG, mec=color, mew=1.6, ls="none", zorder=4)
            elif sub == "production":
                ax.plot(x, r["err"], marker="D", ms=5.5, color=color, ls="none", zorder=4)
            elif sub == "check":
                ax.plot(x, r["err"], marker="s", ms=3.6, color=color, alpha=0.55, ls="none", zorder=3)
            else:
                ax.plot(x, r["err"], marker=".", ms=5, color=color, alpha=0.7, ls="none", zorder=3)
        width = max(len(sel), MIN_GROUP)
        ticks.append(x0 + width / 2 - 0.5)
        labels.append(f"{label}\n{len(sel)} rows, rms {math.sqrt(sum(r['err'] ** 2 for r in sel) / len(sel)):.1f}%")
        x0 += width + GAP
        ax.axvline(x0 - GAP / 2 - 0.5, color="#d8d8d4", lw=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlim(-2, x0 - GAP + 1)
    lo = min(r["err"] for r in rows)
    hi = max(r["err"] for r in rows)
    ax.set_ylim(min(-16, lo - 3), max(16, hi + 3))
    ax.set_ylabel("signed error, predicted / measured - 1 (percent)")
    # annotate the misses beyond the band and the worst production/holdout row
    outside = [(x, r) for x, r in _positions(rows) if abs(r["err"]) > 10]
    for x, r in outside:
        ax.annotate(
            f"{r['cell_id'].split('-topk')[0].split('-li')[0]} {r['op']}\n{r['detail'][:34]}",
            (x, r["err"]),
            textcoords="offset points",
            xytext=(8, -4 if r["err"] > 0 else 4),
            fontsize=7.5,
            color=MUTED,
        )
    prodhold = [(x, r) for x, r in _positions(rows) if r["subset"] in ("production", "holdout")]
    wx, wr = max(prodhold, key=lambda t: abs(t[1]["err"]))
    ax.annotate(
        f"worst production/holdout: {wr['cell_id'].split('-topk')[0].split('-li')[0]} {wr['op']} {wr['err']:+.1f}%",
        (wx, wr["err"]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=8,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7),
    )
    handles = [
        Line2D([], [], marker="D", ls="none", color=INK, ms=5.5, label="production cell (G, L6)"),
        Line2D([], [], marker="o", ls="none", mfc=BG, mec=INK, mew=1.6, ms=6.5, label="holdout (never in a fit)"),
        Line2D([], [], marker=".", ls="none", color=INK, ms=6, label="fit cell"),
        Line2D([], [], marker="s", ls="none", color=INK, alpha=0.55, ms=3.6, label="check cell (L3, L5, B, C, F)"),
        Line2D([], [], color=BLUE, ls="--", lw=0.8, label="+-10 percent band"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=5, bbox_to_anchor=(0.0, 1.0), fontsize=8.5)
    n_ph = len(prodhold)
    n_ph_in = sum(1 for _, r in prodhold if abs(r["err"]) <= 10)
    fig.text(
        0.012,
        0.975,
        "TopK roofline on main tip 2dbd14bf632: signed error per campaign cell by regime, 10 percent band",
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )
    sub = (
        f"{n_in} of {n_total} (cell, op) rows within 10 percent; production and holdout rows {n_ph_in} of {n_ph}. "
        "Measured: tracy DEVICE KERNEL DURATION medians, p100a, tt-metal f3fbc8984f0 (origin/main 2dbd14bf632 merged), 2026-09-12 "
        "(data/topk/results_*.csv); indexer points from the calibration checkout. Predicted: ttsim/perf/roofline_topk.py, "
        "kernel_rev main (data/topk/topk_model_acceptance.csv)."
    )
    fig.text(0.012, 0.935, "\n".join(textwrap.wrap(sub, 150)), fontsize=8.5, va="top", ha="left", color=MUTED)
    save(fig, "tm_signed_errors.png")


def _group_positions(x0, n):
    """Points of a regime group spread over max(n, MIN_GROUP) slots so narrow groups keep their label room."""
    width = max(n, MIN_GROUP)
    if n == 1:
        return [(x0 + width / 2 - 0.5, 0)]
    step = (width - 1) / (n - 1)
    return [(x0 + i * step, i) for i in range(n)]


def _positions(rows):
    x0 = 0
    for regime, _label, _color in REGIME_ORDER:
        sel = sorted([r for r in rows if r["regime"] == regime], key=lambda r: r["meas"])
        if not sel:
            continue
        for (x, _i), r in zip(_group_positions(x0, len(sel)), sel):
            yield x, r
        x0 += max(len(sel), MIN_GROUP) + GAP


# ----------------------------------------------------------------------------------------------
# tm_li_law_main.png
# ----------------------------------------------------------------------------------------------

MODE_COLOR = {"FusedEndToEnd": BLUE, "Classic": ORANGE, "FusedSegmented": AQUA}


def _main_r1_points():
    pts = []
    with open(os.path.join(ROOT, "data", "topk", "results_L1_L2_L3_L4_L5.csv")) as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            p = json.loads(r["params"]) if r["params"] else {}
            if not p or int(p.get("R", 0)) != 1 or p.get("mem") == "l1" or "valid_length" in p:
                continue
            kp = rt.snap_k(int(p["K"]))
            pts.append(
                dict(
                    N=int(p["N"]),
                    K=int(p["K"]),
                    Kp=kp,
                    chunks=math.ceil(int(p["N"]) / kp),
                    mode=rt.body_mode(kp, int(p["N"])),
                    us=float(r["dev_kernel_ns_median"]) / 1000.0,
                    cls=r["cls"],
                )
            )
    return pts


def fig_li_law_main():
    main = _main_r1_points()
    fig, axes = plt.subplots(1, 3, figsize=(1600 / 150, 780 / 150), sharey=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.78, bottom=0.20, wspace=0.28)
    for ax, kp in zip(axes, (512, 1024, 2048)):
        m = [p for p in main if p["Kp"] == kp]
        chunk_grid = np.unique(np.concatenate([[1, 2, 4, 8, 16, 32, 33, 64, 128, 256], [p["chunks"] for p in m]]))
        # model lines at R=1 per body mode on main; the mode is decided by the physical width
        for mode, color in MODE_COLOR.items():
            xs, ys = [], []
            for c in chunk_grid:
                n = int(c) * kp
                if rt.body_mode(kp, n) != mode:
                    continue
                r = rt.predict_topk(rt.TopkConfig(N=n, K=kp, rows=1, kernel_rev="main"))
                xs.append(c)
                ys.append(r.device_cycles / GHZ / 1000.0)
            if xs:
                ax.plot(xs, ys, color=color, lw=1.6, zorder=2)
        for p in m:
            ax.plot(p["chunks"], p["us"], marker="o", ms=5.5, color=MODE_COLOR[p["mode"]], ls="none", zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("chunks per row = ceil(N / K')")
        ax.set_title(f"K' = {kp}")
        t = rt.LARGE_INDICES_TERMS_MAIN
        if kp == 512:
            note = (
                f"a_tot {t.a_tot[(512, 'FusedEndToEnd')]:.0f} (E2E, <= 32 chunks)\n{t.a_tot[(512, 'Classic')]:.0f} (Classic) cyc/chunk, "
                f"c0 {t.c0_row[512] + t.c0_op[512]:.0f}"
            )
        else:
            note = (
                f"a_tot {t.a_tot[(kp, 'FusedSegmented')]:.0f} cyc/chunk, c0 {t.c0_row[kp] + t.c0_op[kp]:.0f}\n"
                f"c_seg {t.c_seg[kp]:.0f} per extra 32-chunk segment"
            )
        ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize=7.8, va="top", ha="left", color=MUTED)
    axes[0].set_ylabel("t_row, one row on one core (us)")
    handles = [
        Line2D([], [], marker="o", ls="none", color=BLUE, ms=5.5, label="main tip, FusedEndToEnd"),
        Line2D([], [], marker="o", ls="none", color=ORANGE, ms=5.5, label="main tip, Classic"),
        Line2D([], [], marker="o", ls="none", color=AQUA, ms=5.5, label="main tip, FusedSegmented"),
        Line2D([], [], color=INK, lw=1.6, label="model, kernel_rev main (per body mode)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8.5)
    fig.text(
        0.012,
        0.975,
        "topk_large_indices row law on main tip 2dbd14bf632: t_row against chunks per K' and body mode",
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )
    sub = (
        "R = 1 (one row on one core), bf16 ROW_MAJOR DRAM interleaved, p100a at 1.35 GHz. Points: data/topk/results_L1_L2_L3_L4_L5.csv "
        "(3 timed calls, median), tt-metal f3fbc8984f0 = origin/main 2dbd14bf632 merged, 2026-09-12. "
        "Lines: ttsim/perf/roofline_topk.py predict_topk at R = 1, i.e. c0 + a_tot(K', mode) x chunks + c_seg x (segments - 1), "
        "plus the single-chunk term at chunks = 1."
    )
    fig.text(0.012, 0.935, "\n".join(textwrap.wrap(sub, 150)), fontsize=8.5, va="top", ha="left", color=MUTED)
    save(fig, "tm_li_law_main.png")


FIGURES = {"tm_signed_errors": fig_signed_errors, "tm_li_law_main": fig_li_law_main}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; known: {sorted(FIGURES)}")
    for n in names:
        FIGURES[n]()
