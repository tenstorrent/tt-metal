#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-engine energy per FLOP (reader / writer / compute), from an existing POWER_CASE sweep.

Isolates each engine's own contribution to dynamic peak charge by single-variable ablation
against the fully-active case (writer_amp, POWER_CASE=1: reader=real compute=real writer=real,
100% write amplification):

    reader_as  = writer_amp.dynamic_charge_peak_as - reader_idle2.dynamic_charge_peak_as    (idle reader,   POWER_CASE=4)
    compute_as = writer_amp.dynamic_charge_peak_as - compute_idle.dynamic_charge_peak_as     (idle compute,  POWER_CASE=2)
    writer_as  = writer_amp.dynamic_charge_peak_as - regular.dynamic_charge_peak_as          (write amp off, POWER_CASE=0)

Charge -> energy -> pJ/FLOP the same way as make_pj_per_flop.py:

    energy_J  = charge_as * avg_vcore_v
    pJ/FLOP   = energy_J / (2 * M * N * K * iterations) * 1e12

The writer row uses `regular` (write amplification 0%) on purpose, to reproduce the published
Fig. 1, which was made before POWER_CASE=5 (`writer_idle`) existed. It isolates the cost of the
write-amplified path, not the writer's total contribution (the writer still performs its one
real write per output tile in `regular`) -- read it as "cost of 100% write amplification", not
"cost of writing at all". The energy_by_engine steps in run_sweep.py use `writer_idle` instead.

Uses dynamic *peak* charge (not average), per request: peak-based ablation subtracts two
single-sample-driven peak values from different runs, which is a valid arithmetic operation but
a noisier, less physically grounded signal than averaging over the whole interval would be --
treat these as a peak-power-driven upper bound, not as tight as the avg-current pJ/FLOP figures
elsewhere in this directory.

Examples
--------
    python3 analysis/make_pj_per_flop_by_engine.py out_fig1/matmul --label "Wormhole n300" \\
        --seq 1024 --hidden 2048 --k 2048 --iters 160

    python3 analysis/make_pj_per_flop_by_engine.py out_fig1/matmul --label "Wormhole n300" \\
        --compare /path/to/blackhole_run/matmul --compare-label "Blackhole p100a" \\
        --seq 1024 --hidden 2048 --k 2048 --iters 160
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# common.py lives one directory up.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import PJ_PER_J, add_mode_argument, find_case_csv, save_figure, whole_run_value  # noqa: E402

ENGINES = [
    ("reader", "Reader (writer_amp - reader_idle2)"),
    ("writer", "Writer amplification (writer_amp - regular)"),
    ("compute", "Compute (writer_amp - compute_idle)"),
]

REQUIRED_CASES = ["regular", "writer_amp", "compute_idle", "reader_idle2"]


def load_rows(path: Path) -> dict:
    rows = {}
    for r in csv.DictReader(open(path)):
        rows[r["grid"]] = r
    return rows


def per_engine_pj_per_flop(root: Path, flops: float, legacy: bool) -> tuple[dict[str, dict[str, float]], list[str]]:
    cases = {}
    for name in REQUIRED_CASES:
        p = find_case_csv(root, name)
        if p is None:
            raise SystemExit(f"no CSV for case '{name}' under {root}")
        cases[name] = load_rows(p)

    grids = [g for g in cases["writer_amp"] if all(g in cases[c] for c in REQUIRED_CASES)]

    per_engine = {name: {} for name, _ in ENGINES}
    for g in grids:
        wa = cases["writer_amp"][g]
        vcore = float(wa["avg_vcore_v"])
        wa_as = whole_run_value(wa, "dynamic_charge_peak_as", legacy)

        def ablate(idle_case):
            idle_as = whole_run_value(cases[idle_case][g], "dynamic_charge_peak_as", legacy)
            energy_j = (wa_as - idle_as) * vcore
            return energy_j / flops * PJ_PER_J

        per_engine["reader"][g] = ablate("reader_idle2")
        per_engine["compute"][g] = ablate("compute_idle")
        per_engine["writer"][g] = ablate("regular")

    return per_engine, grids


def grid_order(grids_by_dataset):
    seen = []
    for grids in grids_by_dataset:
        for g in grids:
            if g not in seen:
                seen.append(g)
    return seen


def chart(datasets, ylim, out_path, subtitle, dpi=150):
    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(14, 6 * n), squeeze=False)
    for ax, (label, per_engine, grids) in zip(axes[:, 0], datasets):
        x = np.arange(len(grids))
        width = 0.8 / len(ENGINES)
        for i, (name, elabel) in enumerate(ENGINES):
            vals = [per_engine[name].get(g, float("nan")) for g in grids]
            ax.bar(x + (i - (len(ENGINES) - 1) / 2) * width, vals, width, label=elabel)
        ax.set_xticks(x)
        ax.set_xticklabels(grids, rotation=45, ha="right")
        ax.set_xlabel("Grid size (core combination)")
        ax.set_ylabel("Energy per FLOP [pJ], peak-current basis")
        ax.set_title(f"Per-engine energy per FLOP per grid - {label}\n{subtitle}")
        ax.set_ylim(0, ylim)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    save_figure(fig, out_path, dpi)
    print(f"Wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="tt-ember output root, or a dir of <case>.csv files")
    ap.add_argument("--label", default="run")
    ap.add_argument("--compare", type=Path, default=None, help="second run, plotted on a shared y-axis")
    ap.add_argument("--compare-label", default="comparison")
    ap.add_argument("--seq", type=int, required=True, help="M")
    ap.add_argument("--hidden", type=int, required=True, help="N")
    ap.add_argument("--k", type=int, required=True, help="K")
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument(
        "--keep-first-grid",
        action="store_true",
        help="Include the first interval; its baseline is one-sided and unreliable.",
    )
    add_mode_argument(ap)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("pj_per_flop_by_engine.png"))
    args = ap.parse_args()

    flops = 2 * args.seq * args.hidden * args.k * args.iters
    subtitle = (
        f"M={args.seq} N={args.hidden} K={args.k} x{args.iters} iterations, "
        f"dynamic peak charge x V_core / {flops:.3g} FLOPs"
    )
    print(f"FLOPs per interval: {flops:.4g}")

    datasets = []
    for root, label in [(args.run, args.label)] + ([(args.compare, args.compare_label)] if args.compare else []):
        per_engine, grids = per_engine_pj_per_flop(root, flops, args.mode == "legacy")
        if not args.keep_first_grid and grids:
            grids = grids[1:]
        datasets.append((label, per_engine, grids))
        print(f"\n{label}:")
        for name, _ in ENGINES:
            vals = [per_engine[name][g] for g in grids if g in per_engine[name]]
            best = min((per_engine[name][g], g) for g in grids if g in per_engine[name])
            print(f"  {name:<8} {min(vals):6.2f} - {max(vals):6.2f} pJ/FLOP   best {best[0]:.2f} at {best[1]}")

    ylim = max(v for _, pe, gs in datasets for name in pe for g, v in pe[name].items() if g in gs) * 1.10
    chart(datasets, ylim, args.out, subtitle, args.dpi)


if __name__ == "__main__":
    main()
