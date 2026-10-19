#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Label a tt-ember run with op names and render a per-op power breakdown.

tt-ember's summary format only understands a compute grid, so ttnn_ops_workload.py encodes the
op index in the grid/cores fields and prints the names as "# OP <idx> <name> ..." lines that
parser.py ignores. This reads both back and produces the breakdown.

Times and energies are per activation, amortised over the batch each op ran at: one batched
call divided by its batch. The batch is chosen per op (and per board), so the sum over the ops
is not the latency of a real unbatched block pass; the batch-1 call time, measured by the
workload, is printed alongside for that. The workload's LONG_MATMUL_MODE decides which ops
there are: the full block (updated), or the 12 ops of the published results (legacy).

--mode (default $LONG_MATMUL_MODE) also sets the time base. updated takes the whole interval
(algo_time_s_reported), the one the iteration count and FLOPs cover; legacy takes the parser's
trimmed window, as the published figures did.
"""
import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# common.py lives one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import save_figure  # noqa: E402

RE_OP = re.compile(
    r"^# OP (?P<idx>\d+) (?P<name>\S+) iters=(?P<iters>\d+) flops_per_iter=(?P<flops>\d+)"
    r"(?: batch=(?P<batch>\d+))?(?: bytes_per_iter=(?P<bytes>\d+))?(?: call_us=\S+)?(?: t1_us=(?P<t1>[\d.]+))?"
)

BAR_WIDTH = 0.8
# Log-axis margins: below the smallest bar and above the largest, so the labels fit.
LOG_BOTTOM_MARGIN = 0.4
LOG_TOP_MARGIN = 3


def finite_positive(values):
    return [v for v in values if v is not None and math.isfinite(v) and v > 0]


def load(run_dir: Path, legacy: bool):
    ops = {}
    for line in (run_dir / "summary.txt").read_text(errors="ignore").splitlines():
        m = RE_OP.match(line)
        if m:
            ops[m.group("idx")] = (
                m.group("name"),
                int(m.group("iters")),
                int(m.group("flops")),
                int(m.group("batch") or 1),
                int(m.group("bytes") or 0),
                float(m.group("t1")) if m.group("t1") else None,
            )

    rows = []
    for r in csv.DictReader(open(run_dir / "program_intervals.csv")):
        idx = r["cores"]
        if idx not in ops:
            continue
        name, iters, flops, batch, nbytes, t1_us = ops[idx]
        time_s = float(r["window_s"]) if legacy else float(r["algo_time_s_reported"])
        dyn = float(r["dynamic_avg_current_a"])
        total_w = float(r["avg_power_vi_w"])
        vcore = float(r["avg_vcore_v"])
        rows.append(
            {
                "idx": int(idx),
                "name": name,
                "iters": iters,
                "flops": flops,
                "batch": batch,
                "t1_us": t1_us,
                "total_w": total_w,
                "dynamic_w": dyn * vcore,
                "base_w": float(r["base_current_a"]) * vcore,
                "us_per_batched_call": time_s / iters * 1e6,
                # Per activation, amortised over the batch. "dyn" is the increment over the idle
                # floor; "idle" is the floor the device burns anyway while this op holds it;
                # "total" is what the op costs in wall-clock terms.
                "dyn_mj_per_activ": dyn * vcore * time_s / iters / batch * 1e3,
                "idle_mj_per_activ": (total_w - dyn * vcore) * time_s / iters / batch * 1e3,
                "us_per_activ": time_s / iters / batch * 1e6,
                "pj_per_flop": (dyn * vcore * time_s / (flops * iters) * 1e12) if flops else None,
                "pj_per_byte": (dyn * vcore * time_s / (nbytes * iters) * 1e12) if nbytes else None,
                "intensity": (flops / nbytes) if nbytes else None,
            }
        )
    return sorted(rows, key=lambda d: d["idx"])


def log_bar_panel(ax, x, values, **bar_kwargs):
    """Bars on a log axis, each labelled with its value. Returns the bars, or None (and leaves
    the panel empty) if no value is finite and positive."""
    shown = finite_positive(values)
    if not shown:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return None
    heights = [v if v in shown else 0.0 for v in values]
    bars = ax.bar(x, heights, BAR_WIDTH, **bar_kwargs)
    ax.set_yscale("log")
    ax.set_ylim(min(shown) * LOG_BOTTOM_MARGIN, max(shown) * LOG_TOP_MARGIN)
    for i, v in enumerate(heights):
        if v > 0:
            ax.annotate(
                f"{v:.3g}", (i, v), ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points"
            )
    ax.grid(axis="y", alpha=0.3)
    return bars


def normalised_chart(rows, out_path, board, dpi=150):
    """Energy per unit of work, under both denominators.

    There is no single one. A compute-bound op should be judged per FLOP and a memory-bound one
    per byte, and these ops span both regimes -- so the two panels rank them in opposite orders.
    Arithmetic intensity, printed under each op, says which panel applies to it.
    """
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(rows) * 1.05), 9), sharex=True)

    for ax, key, ylab, colour in [
        (axes[0], "pj_per_flop", "Energy per FLOP [pJ], log", "tab:blue"),
        (axes[1], "pj_per_byte", "Energy per byte touched [pJ], log", "tab:purple"),
    ]:
        log_bar_panel(ax, x, [r[key] for r in rows], color=colour)
        ax.set_ylabel(ylab)

    axes[0].set_title(
        f"Energy per unit of work - transformer decoder block{board}\n"
        "the two denominators rank the ops in opposite orders; intensity says which one applies"
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f"{n}\n{r['intensity']:.3g} F/B" if r["intensity"] else n for n, r in zip(names, rows)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    axes[1].set_xlabel("Operation, with arithmetic intensity in FLOP/byte")
    save_figure(fig, out_path, dpi)
    print(f"Wrote: {out_path}")


def energy_chart(rows, out_path, board, dpi=150):
    """Energy per op, per activation, split into dynamic and idle-floor share.

    Dynamic alone answers "what did this op make the chip do extra". Total answers "what did
    this op cost", and the two rank the ops differently: a slow op at modest power accumulates
    idle-floor energy simply by holding the device. The idle share depends on each op's batch,
    so it is not the share of a real unbatched pass.
    """
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    dyn = np.array([r["dyn_mj_per_activ"] for r in rows])
    idle = np.array([r["idle_mj_per_activ"] for r in rows])
    tot = dyn + idle

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.05), 6.5))
    ax.bar(x, dyn, BAR_WIDTH, label="dynamic (work the op caused)")
    ax.bar(x, idle, BAR_WIDTH, bottom=dyn, label="idle floor held during the op")
    for i, t in enumerate(tot):
        ax.annotate(
            f"{100*t/tot.sum():.0f}%",
            (i, t),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Operation (transformer decoder block)")
    ax.set_ylabel("Energy per activation [mJ], amortised over the batch")
    ax.set_title(
        f"Energy per op - transformer decoder block{board}\n"
        f"sum over the {len(rows)} ops = {tot.sum():.0f} mJ per activation "
        f"({dyn.sum():.0f} mJ dynamic + {idle.sum():.0f} mJ idle floor), "
        f"{sum(r['us_per_activ'] for r in rows):.0f} us amortised"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, out_path, dpi)
    print(f"Wrote: {out_path}")


def chart(rows, out_path, subtitle, board, dpi=150):
    """Three panels over one shared op axis.

    Efficiency (pJ/FLOP) and absolute cost (mJ per activation) answer different questions and
    must be read together: an op can be inefficient yet negligible, or efficient yet dominant.
    Dynamic power is kept as the third panel for context.
    """
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(rows) * 1.05), 12), sharex=True)

    ax = axes[0]
    # pJ/FLOP spans more than two orders of magnitude between the matmuls and the elementwise
    # ops. On a linear axis the matmul bars are about a pixel tall and read as missing, so use
    # a log axis and label every bar with its value.
    bars = log_bar_panel(ax, x, [r["pj_per_flop"] for r in rows])
    if bars is not None:
        # Matmul FLOP counts are exact; the rest use per-element conventions, so hatch them.
        for i, r in enumerate(rows):
            if not r["name"].startswith(("matmul", "attn")):
                bars[i].set_hatch("//")
        ax.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, fc=bars[0].get_facecolor()),
                plt.Rectangle((0, 0), 1, 1, fc=bars[0].get_facecolor(), hatch="//"),
            ],
            labels=["matmul: exact 2*M*N*K", "other: per-element convention (see workload script)"],
            fontsize=8,
            loc="upper left",
        )
    ax.set_ylabel("Energy per FLOP [pJ], log scale")
    ax.set_title(f"Per-op efficiency and cost - transformer decoder block{board}\n{subtitle}")

    ax = axes[1]
    ax.bar(x, [r["dyn_mj_per_activ"] for r in rows], BAR_WIDTH, color="tab:orange")
    ax.set_ylabel("Dynamic energy per activation [mJ]")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(x, [r["dynamic_w"] for r in rows], BAR_WIDTH, color="tab:green")
    ax.set_ylabel("Dynamic power [W]")
    ax.set_xlabel("Operation (transformer decoder block)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, out_path, dpi)
    print(f"Wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--board", default="", help="Board name for the chart titles, e.g. 'Blackhole p100a'")
    ap.add_argument(
        "--mode",
        choices=("updated", "legacy"),
        default=os.environ.get("LONG_MATMUL_MODE") or "updated",
        help="Time base: updated uses the whole interval, legacy the trimmed window, as the published "
        "figures did. Default: $LONG_MATMUL_MODE, else updated",
    )
    args = ap.parse_args()
    board = f", {args.board}" if args.board else ""

    rows = load(args.run_dir, args.mode == "legacy")
    if not rows:
        raise SystemExit("no op rows found -- is this a ttnn_ops_workload run?")

    print(
        f"{'op':<16} {'batch':>6} {'us/batched call':>15} {'us/activ':>9} {'us@batch 1':>10} "
        f"{'total W':>8} {'dyn W':>7} {'mJ/activ':>9} {'pJ/FLOP':>9}"
    )
    print("-" * 100)
    for r in rows:
        pj = f"{r['pj_per_flop']:9.1f}" if r["pj_per_flop"] else "        -"
        t1 = f"{r['t1_us']:10.0f}" if r["t1_us"] is not None else "         -"
        print(
            f"{r['name']:<16} {r['batch']:6d} {r['us_per_batched_call']:15.0f} {r['us_per_activ']:9.0f} {t1} "
            f"{r['total_w']:8.1f} {r['dynamic_w']:7.1f} {r['dyn_mj_per_activ']:9.4f} {pj}"
        )

    base = np.mean([r["base_w"] for r in rows])
    print("-" * 100)
    print(f"idle floor (mean over intervals): {base:.1f} W")

    energy_chart(rows, args.run_dir / "Figures" / "op_energy_breakdown.png", board)
    normalised_chart(rows, args.run_dir / "Figures" / "op_energy_normalised.png", board)
    out = args.out or (args.run_dir / "Figures" / "op_power_breakdown.png")
    total = sum(r["dyn_mj_per_activ"] for r in rows)
    print(f"total dynamic energy per activation, summed over the {len(rows)} ops: {total:.2f} mJ")
    chart(
        rows,
        out,
        f"sum over the {len(rows)} ops = {total:.1f} mJ dynamic per activation, idle floor {base:.0f} W",
        board,
    )


if __name__ == "__main__":
    main()
