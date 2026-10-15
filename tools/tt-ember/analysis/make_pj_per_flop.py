#!/usr/bin/env python3
"""Energy per FLOP from an existing tt-ember run. No new measurement required.

pJ/FLOP is derived, not measured: `energy_j_vi` is already in program_intervals.csv and the
FLOP count is fixed by the shape, since split mode gives every grid the same total work.

    pJ/FLOP = energy_j_vi / (2 * M * N * K * iterations) * 1e12

So any run that has already been captured -- including ones from months ago -- can be converted
without touching the hardware.

Examples
--------
Wormhole, from the run committed in turkmanovic/tt-energyprofiler@master:

    ./make_pj_per_flop.py out_new2 --label "Wormhole n150" \\
        --seq 1024 --hidden 2048 --k 2048 --iters 160

Two boards on one shared y-axis, for comparison:

    ./make_pj_per_flop.py out_new2 --label "Wormhole n150" \\
        --compare /path/to/blackhole_run --compare-label "Blackhole p100a" \\
        --seq 1024 --hidden 2048 --k 2048 --iters 160
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Only cases that actually execute the matmuls. POWER_CASE 2 and 3 skip matmul_tiles, so no
# FLOPs are performed and energy-per-FLOP is undefined for them; case 4 runs them on stale data.
DEFAULT_CASES = [
    ("regular", "Regular (POWER_CASE=0)"),
    ("writer_amp", "Writer amp, 100% write amplification (POWER_CASE=1)"),
]


def find_csv(root: Path, case: str) -> Path | None:
    """Accept either a tt-ember output root (<root>/<case>/program_intervals.csv) or a
    directory of flat per-case CSVs (<root>/<case>.csv)."""
    for cand in (root / case / "program_intervals.csv", root / f"{case}.csv"):
        if cand.exists():
            return cand
    return None


def pj_per_flop(path: Path, flops: float) -> dict:
    out = {}
    for r in csv.DictReader(open(path)):
        try:
            out[r["grid"]] = float(r["energy_j_vi"]) / flops * 1e12
        except (KeyError, ValueError):
            continue
    return out


def grid_order(per_case: dict) -> list:
    """Sweep order as it appears in the CSV, which is the order the workload ran them."""
    seen = []
    for vals in per_case.values():
        for g in vals:
            if g not in seen:
                seen.append(g)
    return seen


def collect(root: Path, cases, flops, skip_first):
    per_case = {}
    for name, _ in cases:
        p = find_csv(root, name)
        if p is None:
            print(f"  note: no CSV for case '{name}' under {root}, skipping")
            continue
        per_case[name] = pj_per_flop(p, flops)
    if not per_case:
        raise SystemExit(f"no case CSVs found under {root}")
    grids = grid_order(per_case)
    if skip_first and grids:
        # The first interval has a one-sided baseline estimate and is the least reliable
        # point in any sweep; excluded by default.
        grids = grids[1:]
    return per_case, grids


def chart(datasets, cases, ylim, out_path, subtitle, dpi=150):
    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(max(10, 14), 6 * n), squeeze=False)
    for ax, (label, per_case, grids) in zip(axes[:, 0], datasets):
        x = np.arange(len(grids))
        width = 0.8 / max(len(cases), 1)
        for i, (name, clabel) in enumerate(cases):
            if name not in per_case:
                continue
            vals = [per_case[name].get(g, float("nan")) for g in grids]
            ax.bar(x + (i - (len(cases) - 1) / 2) * width, vals, width, label=clabel)
        ax.set_xticks(x)
        ax.set_xticklabels(grids, rotation=45, ha="right")
        ax.set_xlabel("Grid size (core combination)")
        ax.set_ylabel("Energy per FLOP [pJ]")
        ax.set_title(f"Energy per FLOP per grid, per use case - {label}\n{subtitle}")
        ax.set_ylim(0, ylim)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="tt-ember output root, or a dir of <case>.csv files")
    ap.add_argument("--label", default="run")
    ap.add_argument("--compare", type=Path, default=None, help="second run, plotted on a shared y-axis")
    ap.add_argument("--compare-label", default="comparison")
    ap.add_argument("--seq", type=int, required=True, help="M")
    ap.add_argument("--hidden", type=int, required=True, help="N")
    ap.add_argument("--k", type=int, required=True, help="K")
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--keep-first-grid", action="store_true",
                    help="Include the first interval; its baseline is one-sided and unreliable.")
    ap.add_argument("--out", type=Path, default=Path("pj_per_flop.png"))
    args = ap.parse_args()

    flops = 2 * args.seq * args.hidden * args.k * args.iters
    subtitle = (f"M={args.seq} N={args.hidden} K={args.k} x{args.iters} iterations, "
                f"total energy including idle floor / {flops:.3g} FLOPs")
    print(f"FLOPs per interval: {flops:.4g}")

    datasets = []
    for root, label in [(args.run, args.label)] + (
            [(args.compare, args.compare_label)] if args.compare else []):
        per_case, grids = collect(root, DEFAULT_CASES, flops, not args.keep_first_grid)
        datasets.append((label, per_case, grids))
        print(f"\n{label}:")
        for name, _ in DEFAULT_CASES:
            if name not in per_case:
                continue
            vals = [per_case[name][g] for g in grids if g in per_case[name]]
            best = min((per_case[name][g], g) for g in grids if g in per_case[name])
            print(f"  {name:<12} {min(vals):6.1f} - {max(vals):6.1f} pJ/FLOP   best {best[0]:.1f} at {best[1]}")

    ylim = max(v for _, pc, gs in datasets for vals in pc.values()
               for g, v in vals.items() if g in gs) * 1.10
    chart(datasets, DEFAULT_CASES, ylim, args.out, subtitle)


if __name__ == "__main__":
    main()
