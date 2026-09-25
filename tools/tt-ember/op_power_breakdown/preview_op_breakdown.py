#!/usr/bin/env python3
"""Label a tt-ember run with op names and render a per-op power breakdown.

tt-ember's summary format only understands a compute grid, so ttnn_ops_workload.py encodes the
op index in the grid/cores fields and prints the names as "# OP <idx> <name> ..." lines that
parser.py ignores. This reads both back and produces the breakdown. tt-ember itself is not
modified; this only consumes its output.
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RE_OP = re.compile(r"^# OP (?P<idx>\d+) (?P<name>\S+) iters=(?P<iters>\d+) flops_per_iter=(?P<flops>\d+)(?: batch=(?P<batch>\d+))?(?: bytes_per_iter=(?P<bytes>\d+))?")


def load(run_dir: Path):
    ops = {}
    for line in (run_dir / "summary.txt").read_text(errors="ignore").splitlines():
        m = RE_OP.match(line)
        if m:
            ops[m.group("idx")] = (m.group("name"), int(m.group("iters")),
                                   int(m.group("flops")), int(m.group("batch") or 1),
                                   int(m.group("bytes") or 0))

    rows = []
    for r in csv.DictReader(open(run_dir / "program_intervals.csv")):
        idx = r["cores"]
        if idx not in ops:
            continue
        name, iters, flops, batch, nbytes = ops[idx]
        window = float(r["window_s"])
        dyn = float(r["dynamic_avg_current_a"]) if r["dynamic_avg_current_a"] != "nan" else float("nan")
        total_w = float(r["avg_power_vi_w"])
        vcore = float(r["avg_vcore_v"])
        base_w = float(r["base_current_a"]) * vcore
        rows.append({
            "idx": int(idx), "name": name, "iters": iters, "flops": flops, "batch": batch,
            "window_s": window,
            "total_w": total_w,
            "dynamic_w": dyn * vcore,
            "base_w": base_w,
            # Energy for one invocation of the op in a single block pass. "dyn" is the
            # increment over the idle floor; "idle" is the floor the device burns anyway while
            # this op holds it; "total" is what the op actually costs you in wall-clock terms.
            "dyn_mj_per_call": dyn * vcore * window / iters / batch * 1e3,
            "idle_mj_per_call": (total_w - dyn * vcore) * window / iters / batch * 1e3,
            "total_mj_per_call": total_w * window / iters / batch * 1e3,
            "us_per_call": window / iters / batch * 1e6,
            "tflops": float(r["algo_time_s_reported"]) if False else None,
            "pj_per_flop": (dyn * vcore * window / (flops * iters) * 1e12) if flops else None,
            "pj_per_byte": (dyn * vcore * window / (nbytes * iters) * 1e12) if nbytes else None,
            "intensity": (flops / nbytes) if nbytes else None,
        })
    return sorted(rows, key=lambda d: d["idx"])


def normalised_chart(rows, out_path, dpi=150):
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
            (axes[1], "pj_per_byte", "Energy per byte touched [pJ], log", "tab:purple")]:
        v = [r[key] or 0.0 for r in rows]
        ax.bar(x, v, 0.8, color=colour)
        ax.set_yscale("log")
        ax.set_ylim(min(z for z in v if z) * 0.4, max(v) * 3)
        for i, z in enumerate(v):
            ax.annotate(f"{z:.3g}", (i, z), ha="center", va="bottom", fontsize=8,
                        xytext=(0, 2), textcoords="offset points")
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_title(
        "Energy per unit of work - transformer decoder block, Blackhole p100a\n"
        "the two denominators rank the ops in opposite orders; intensity says which one applies")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [f"{n}\n{r['intensity']:.3g} F/B" if r["intensity"] else n
         for n, r in zip(names, rows)], rotation=45, ha="right", fontsize=8)
    axes[1].set_xlabel("Operation, with arithmetic intensity in FLOP/byte")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def energy_chart(rows, out_path, dpi=150):
    """Energy per op for one block pass, split into dynamic and idle-floor share.

    Dynamic alone answers "what did this op make the chip do extra". Total answers "what did
    this op cost", and the two rank the ops differently: a slow op at modest power accumulates
    idle-floor energy simply by holding the device.
    """
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    dyn = np.array([r["dyn_mj_per_call"] for r in rows])
    idle = np.array([r["idle_mj_per_call"] for r in rows])
    tot = dyn + idle

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.05), 6.5))
    ax.bar(x, dyn, 0.8, label="dynamic (work the op caused)")
    ax.bar(x, idle, 0.8, bottom=dyn, label="idle floor held during the op")
    for i, r in enumerate(rows):
        ax.annotate(f"{100*tot[i]/tot.sum():.0f}%", (i, tot[i]), ha="center",
                    va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Operation (transformer decoder block)")
    ax.set_ylabel("Energy for one block pass [mJ]")
    ax.set_title("Energy per op - transformer decoder block, Blackhole p100a\n"
                 f"one block pass = {tot.sum():.0f} mJ total "
                 f"({dyn.sum():.0f} mJ dynamic + {idle.sum():.0f} mJ idle floor), "
                 f"{sum(r['us_per_call'] for r in rows):.0f} us")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def chart(rows, out_path, subtitle, dpi=150):
    """Three panels over one shared op axis.

    Efficiency (pJ/FLOP) and absolute cost (mJ per block invocation) answer different
    questions and must be read together: an op can be inefficient yet negligible, or
    efficient yet dominant. Dynamic power is kept as the third panel for context.
    """
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(max(10, len(rows) * 1.05), 12), sharex=True)

    ax = axes[0]
    pj = [r["pj_per_flop"] if r["pj_per_flop"] else 0.0 for r in rows]
    bars = ax.bar(x, pj, 0.8)
    # pJ/FLOP spans 0.8 to 270 here -- more than two orders of magnitude. On a linear axis the
    # matmul bars are about a pixel tall and read as missing, so use a log axis and label every
    # bar with its value.
    ax.set_yscale("log")
    ax.set_ylim(0.5, max(pj) * 3)
    for i, v in enumerate(pj):
        ax.annotate(f"{v:.3g}", (i, v), ha="center", va="bottom", fontsize=8,
                    xytext=(0, 2), textcoords="offset points")
    # Matmul FLOP counts are exact; the rest use per-element conventions, so hatch them.
    for i, r in enumerate(rows):
        if not r["name"].startswith(("matmul", "attn")):
            bars[i].set_hatch("//")
    ax.set_ylabel("Energy per FLOP [pJ], log scale")
    ax.set_title(f"Per-op efficiency and cost - transformer decoder block, Blackhole p100a\n{subtitle}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc=bars[0].get_facecolor()),
        plt.Rectangle((0, 0), 1, 1, fc=bars[0].get_facecolor(), hatch="//")],
        labels=["matmul: exact 2*M*N*K", "other: per-element convention (see workload script)"],
        fontsize=8, loc="upper left")

    ax = axes[1]
    ax.bar(x, [r["dyn_mj_per_call"] for r in rows], 0.8, color="tab:orange")
    ax.set_ylabel("Energy per block pass [mJ]")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(x, [r["dynamic_w"] for r in rows], 0.8, color="tab:green")
    ax.set_ylabel("Dynamic power [W]")
    ax.set_xlabel("Operation (transformer decoder block)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = load(args.run_dir)
    if not rows:
        raise SystemExit("no op rows found -- is this a ttnn_ops_workload run?")

    print(f"{'op':<16} {'batch':>6} {'us/call':>9} {'total W':>8} {'dyn W':>7} "
          f"{'mJ/activ':>9} {'pJ/FLOP':>9}")
    print("-" * 82)
    for r in rows:
        pj = f"{r['pj_per_flop']:9.1f}" if r["pj_per_flop"] else "        -"
        print(f"{r['name']:<16} {r['batch']:6d} {r['window_s']/r['iters']*1e6:9.0f} "
              f"{r['total_w']:8.1f} {r['dynamic_w']:7.1f} {r['dyn_mj_per_call']:9.4f} {pj}")

    base = np.mean([r["base_w"] for r in rows])
    print("-" * 82)
    print(f"idle floor (mean over intervals): {base:.1f} W")

    energy_chart(rows, args.run_dir / "Figures" / "op_energy_breakdown.png")
    normalised_chart(rows, args.run_dir / "Figures" / "op_energy_normalised.png")
    out = args.out or (args.run_dir / "Figures" / "op_power_breakdown.png")
    total = sum(r["dyn_mj_per_call"] for r in rows)
    print(f"total dynamic energy for one block pass: {total:.2f} mJ")
    chart(rows, out, f"one block pass = {total:.1f} mJ dynamic, idle floor {base:.0f} W")


if __name__ == "__main__":
    main()
