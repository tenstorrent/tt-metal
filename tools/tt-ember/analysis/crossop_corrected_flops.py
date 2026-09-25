#!/usr/bin/env python3
"""Cross-op energy chart with a per-op FLOP denominator, as an addendum to run_power_cases.py.

run_power_cases.py divides every op's energy by 2*M*N*K*iters -- the matmul's FLOP count -- for
all seven ops. Since the denominator is one global constant, its cross-op chart is a faithful
*energy* comparison at fixed data movement, which is the controlled quantity the sweep varies.
It is not energy per FLOP for the six non-matmul ops, and the two rank the ops in opposite
orders: `add` looks twice as efficient as matmul on the shared denominator and roughly thirty
times worse on its own.

The kernel issues one math instruction per tile pair per step of the shared dimension, so per
interval it performs Kt*Mt*Nt = M*N*K/32768 tile operations. What one tile operation costs in
FLOPs is what differs:

  matmul_tiles   2 * 32^3 = 65536   an exact count
  everything else       1024 * w    one element operation per element of a 32x32 tile,
                                    times a per-element convention w

The conventions (w) are the usual ones and are stated in FLOPS_PER_ELEM below. They are
conventions, not measurements -- there is no agreed FLOP weight for a transcendental -- so this
chart is the softer of the two and the raw-energy one should be read alongside it.

Reads the same <output-root>/<op>/<case>/program_intervals.csv tree the sweep writes. Nothing
in tt-ember is modified.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Engine -> the ablation case whose absence isolates it, matching run_power_cases.py's
# ENGINE_ABLATIONS so the two charts decompose energy the same way.
ALL_ACTIVE = "writer_amp"
ENGINES = [("reader", "reader_idle2"), ("writer", "writer_idle"), ("compute", "compute_idle")]

OPS = ["matmul", "add", "silu", "exp", "sigmoid", "gelu", "recip"]

# FLOPs per element for the non-matmul ops, i.e. w above. matmul is handled exactly and is not
# listed. sigmoid is counted as exp + add + reciprocal; silu adds the multiply by x; gelu is
# counted as the tanh approximation (tanh, two multiplies, an add, a cube, a scale).
FLOPS_PER_ELEM = {"add": 1, "recip": 1, "exp": 1, "sigmoid": 3, "silu": 4, "gelu": 6}

TILE = 32
ELEMS_PER_TILE = TILE * TILE
FLOPS_PER_MATMUL_TILE_OP = 2 * TILE ** 3


def op_flops(op: str, M: int, N: int, K: int, iters: int) -> float:
    """FLOPs actually performed by one interval of `op`, on its own convention."""
    tile_ops = (M / TILE) * (N / TILE) * (K / TILE) * iters
    if op == "matmul":
        return tile_ops * FLOPS_PER_MATMUL_TILE_OP
    return tile_ops * ELEMS_PER_TILE * FLOPS_PER_ELEM[op]


def load(csv_path: Path):
    if not csv_path.exists():
        return None
    with csv_path.open(newline="") as f:
        return {r["grid"]: r for r in csv.DictReader(f)}


def energy_by_engine(op_root: Path):
    """Dynamic energy [J] per engine per grid: (all-active - engine-idle) charge x V_core."""
    active = load(op_root / ALL_ACTIVE / "program_intervals.csv")
    if active is None:
        return None, None
    per_engine = {}
    for engine, idle_case in ENGINES:
        idle = load(op_root / idle_case / "program_intervals.csv")
        if idle is None:
            continue
        per_engine[engine] = {
            g: (float(r["dynamic_charge_avg_as"]) - float(idle[g]["dynamic_charge_avg_as"]))
               * float(r["avg_vcore_v"])
            for g, r in active.items() if g in idle
        }
    measured = {g: float(r["dynamic_charge_avg_as"]) * float(r["avg_vcore_v"])
                for g, r in active.items()}
    return per_engine, measured


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_root", type=Path)
    ap.add_argument("--app-args", nargs=4, type=int, required=True, metavar=("M", "N", "K", "ITERS"))
    ap.add_argument("--board", default="Blackhole p100a")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()
    M, N, K, iters = args.app_args

    data, resid = {}, {}
    for op in OPS:
        per_engine, measured = energy_by_engine(args.output_root / op)
        if not per_engine:
            print(f"note: no data for '{op}', skipping")
            continue
        f = op_flops(op, M, N, K, iters)
        data[op] = {e: {g: j / f * 1e12 for g, j in pg.items()} for e, pg in per_engine.items()}
        # How far the three ablations are from accounting for the measured dynamic energy. The
        # engines overlap in time, so they are not strictly additive; printing the residual
        # keeps that visible instead of hidden inside a normalised stack.
        resid[op] = {g: sum(pg.get(g, 0.0) for pg in per_engine.values()) / measured[g]
                     for g in measured}
    if not data:
        raise SystemExit("no op data found under " + str(args.output_root))

    grids = [g for g in load(args.output_root / OPS[0] / ALL_ACTIVE / "program_intervals.csv")]
    ops = [o for o in OPS if o in data]

    print(f"{'op':9s} {'FLOPs/interval':>15s} {'x vs matmul':>11s} "
          f"{'pJ/FLOP (last grid)':>20s} {'Sum/meas':>9s}")
    mm = op_flops("matmul", M, N, K, iters)
    for op in ops:
        f = op_flops(op, M, N, K, iters)
        last = grids[-1]
        tot = sum(data[op][e].get(last, 0.0) for e, _ in ENGINES if e in data[op])
        print(f"{op:9s} {f:15.4g} {mm/f:11.1f} {tot:20.1f} "
              f"{np.mean(list(resid[op].values())):9.2f}")

    x = np.arange(len(grids))
    width = 0.8 / len(ops)
    colours = {"reader": "C0", "writer": "C1", "compute": "C2"}
    hatches = ["", "//", "xx", "\\\\", "..", "OO", "**"]

    fig, ax = plt.subplots(figsize=(max(14, len(grids) * 1.15), 7))
    for i, op in enumerate(ops):
        offset = (i - (len(ops) - 1) / 2) * width
        bottom = np.zeros(len(grids))
        for engine, _ in ENGINES:
            if engine not in data[op]:
                continue
            v = np.array([data[op][engine].get(g, np.nan) for g in grids])
            ax.bar(x + offset, v, width * 0.92, bottom=bottom, color=colours[engine],
                   hatch=hatches[i % len(hatches)], edgecolor="black", linewidth=0.3)
            bottom += np.nan_to_num(v)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(grids, rotation=45, ha="right")
    ax.set_xlabel("Grid size (core combination)")
    ax.set_ylabel("Energy per FLOP [pJ], log scale")
    ax.set_title(
        f"Energy per FLOP by engine, per grid, per op - {args.board}, each op on its own FLOP count\n"
        f"matmul is exact (2*32^3 per tile op); the rest use the per-element conventions in "
        f"FLOPS_PER_ELEM, so their ratios are convention-dependent")
    ax.grid(axis="y", alpha=0.3, which="both")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colours[e]) for e, _ in ENGINES]
    labels = [e.capitalize() for e, _ in ENGINES]
    handles += [plt.Rectangle((0, 0), 1, 1, fc="white", hatch=hatches[i % len(hatches)],
                              edgecolor="black") for i in range(len(ops))]
    labels += ops
    ax.legend(handles, labels, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    out = args.out or (args.output_root / "figures" / "cross_op" /
                       "energy_per_flop_by_engine_stacked_cross_op_corrected.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
