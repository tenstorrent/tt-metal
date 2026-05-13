#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Aggregate and compare CombineDeviceOperation timings across Galaxy and 8x1 (LoudBox).

INPUTS
======

Galaxy CSV
  Produced by running test_prefill_transformer.py with TT_METAL_DEVICE_PROFILER=1
  (one full forward, 58 MoE layers, 32 chips). Each combine row carries
  DEVICE ID and GLOBAL CALL COUNT; per device the N-th CombineDeviceOperation
  row corresponds to MoE layer (N-1)+3 because layers 0..2 are dense.

LB CSVs
  Produced by running test_combine_replay.py on an 8x1 mesh.  Each test
  instance (one capture file × one num_links) contains:
        n_warmup + n_timed CombineDeviceOperation invocations × 8 chips
  Run each combo separately so each invocation produces its own CSV with a
  unique filename suffix. The expected suffix encodes attribution:
        ops_perf_results_<ts>_L<NN>_col<K>_<linksid>.csv
  where <linksid> is e.g. "linear-8-1link" or "linear-8-2link".

  Achieve the suffix with `--name-append`:
        python -m tracy --process-logs-only -n "L30_col0_linear-8-1link"

USAGE
=====

  python analyze_combine_perf.py galaxy <galaxy_csv>
      Per-layer max/median/min combine kernel ns across 32 chips on Galaxy.

  python analyze_combine_perf.py lb <one_lb_csv> [--n-warmup 3]
      Per-chip median across timed iters in that one run, plus max-across-chips.

  python analyze_combine_perf.py compare --galaxy <csv> --lb-glob "<glob>"
      Per-layer comparison.  For each layer × num_links combo:
        galaxy_max_chip_ns:    Galaxy actual (max over 32 chips of single invocation)
        lb_max_col_chip_ns:    Predicted LB no-contention time
                               = max over 4 columns of (max over 8 chips of timed-iter-median)
        overhead_ratio:        galaxy_max / lb_max_col
"""

import argparse
import glob
import re
import sys
from pathlib import Path

import pandas as pd

# Layers 0..2 are dense FFN, so the N-th CombineDeviceOperation invocation per
# device corresponds to MoE layer N+3 (with N starting at 0).
FIRST_MOE_LAYER = 3

# LB filename suffix: ..._L<NN>_col<K>_<linksid>.csv where linksid contains "<n>link".
LB_NAME_RE = re.compile(
    r"L(?P<layer>\d+)_col(?P<col>\d+)_(?P<links>linear-8-\d+link)",
    re.IGNORECASE,
)


def _load_combine(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["OP CODE"] == "CombineDeviceOperation"].copy()
    df["dur_ns"] = pd.to_numeric(df["DEVICE KERNEL DURATION [ns]"], errors="coerce")
    df = df.dropna(subset=["dur_ns"])
    df["dur_ns"] = df["dur_ns"].astype("int64")
    df = df.sort_values(["DEVICE ID", "GLOBAL CALL COUNT"]).reset_index(drop=True)
    df["iter_in_dev"] = df.groupby("DEVICE ID").cumcount()
    return df


def galaxy_per_layer(csv_path: str) -> pd.DataFrame:
    """Galaxy 8x4 / 32 chips / one combine call per MoE layer per chip."""
    df = _load_combine(csv_path)
    df["layer_idx"] = df["iter_in_dev"] + FIRST_MOE_LAYER
    per_layer = (
        df.groupby("layer_idx")["dur_ns"]
        .agg(["max", "median", "min", "mean", "count"])
        .rename(
            columns={
                "max": "galaxy_max_chip_ns",
                "median": "galaxy_med_chip_ns",
                "min": "galaxy_min_chip_ns",
                "mean": "galaxy_mean_chip_ns",
                "count": "n_chip_samples",
            }
        )
        .reset_index()
    )
    return per_layer


def lb_run_per_chip(csv_path: str, n_warmup: int = 3) -> pd.Series:
    """LB single-run: per-chip median across timed iters, indexed by DEVICE ID."""
    df = _load_combine(csv_path)
    timed = df[df["iter_in_dev"] >= n_warmup]
    if timed.empty:
        raise ValueError(
            f"{csv_path}: no timed iters after dropping {n_warmup} warmup; "
            f"got iter range [{df['iter_in_dev'].min()}, {df['iter_in_dev'].max()}]"
        )
    return timed.groupby("DEVICE ID")["dur_ns"].median()


def lb_aggregate(lb_glob: str, n_warmup: int = 3) -> pd.DataFrame:
    """Walk LB CSVs, parse (layer, col, num_links) from filename, build one row per file."""
    paths = sorted(glob.glob(lb_glob))
    rows = []
    for p in paths:
        m = LB_NAME_RE.search(Path(p).name)
        if not m:
            print(f"[skip] no attribution suffix in: {p}", file=sys.stderr)
            continue
        layer = int(m.group("layer"))
        col = int(m.group("col"))
        links = m.group("links").lower()
        try:
            per_chip = lb_run_per_chip(p, n_warmup=n_warmup)
        except ValueError as e:
            print(f"[skip] {e}", file=sys.stderr)
            continue
        rows.append(
            {
                "layer_idx": layer,
                "col": col,
                "num_links": links,
                "lb_max_chip_med_ns": int(per_chip.max()),
                "lb_median_chip_med_ns": int(per_chip.median()),
                "n_chips": len(per_chip),
                "csv": p,
            }
        )
    if not rows:
        raise SystemExit(f"No LB CSVs matched at {lb_glob!r} with expected suffix")
    return pd.DataFrame(rows)


def compare(galaxy_csv: str, lb_glob: str, n_warmup: int = 3) -> pd.DataFrame:
    galaxy = galaxy_per_layer(galaxy_csv).set_index("layer_idx")
    lb = lb_aggregate(lb_glob, n_warmup=n_warmup)

    # For each (layer, num_links), take max across the 4 columns of per-column max-chip-median.
    # That mirrors the per-layer effective combine time on a no-contention 8x1 replay.
    lb_per_layer_links = (
        lb.groupby(["layer_idx", "num_links"])
        .agg(
            lb_max_col_chip_ns=("lb_max_chip_med_ns", "max"),
            lb_min_col_chip_ns=("lb_max_chip_med_ns", "min"),
            n_cols=("col", "nunique"),
        )
        .reset_index()
    )

    out = lb_per_layer_links.merge(
        galaxy[["galaxy_max_chip_ns", "galaxy_med_chip_ns"]].reset_index(),
        on="layer_idx",
        how="left",
    )
    out["overhead_ratio"] = out["galaxy_max_chip_ns"] / out["lb_max_col_chip_ns"]
    out["overhead_abs_ns"] = out["galaxy_max_chip_ns"] - out["lb_max_col_chip_ns"]
    return out.sort_values(["layer_idx", "num_links"]).reset_index(drop=True)


def _print_df(df: pd.DataFrame):
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("galaxy", help="Aggregate Galaxy CSV: one row per layer")
    g.add_argument("csv")

    l = sub.add_parser("lb", help="Aggregate one LB CSV: per-chip median across timed iters")
    l.add_argument("csv")
    l.add_argument("--n-warmup", type=int, default=3)

    c = sub.add_parser("compare", help="Build per-layer × num_links comparison table")
    c.add_argument("--galaxy", required=True)
    c.add_argument(
        "--lb-glob",
        required=True,
        help=(
            "Glob matching LB CSVs with attribution suffix, e.g. "
            '"generated/profiler/reports/*/ops_perf_results_*_L*_col*_linear-8-*link.csv"'
        ),
    )
    c.add_argument("--n-warmup", type=int, default=3)
    c.add_argument("--out", default=None, help="Optional path to save the comparison CSV.")

    args = ap.parse_args()

    if args.cmd == "galaxy":
        _print_df(galaxy_per_layer(args.csv))
    elif args.cmd == "lb":
        per_chip = lb_run_per_chip(args.csv, n_warmup=args.n_warmup)
        print(f"per-chip median (ns):")
        print(per_chip.to_string())
        print()
        print(f"max-across-chips of chip-median: {per_chip.max():,.0f} ns")
        print(f"median-across-chips of chip-median: {per_chip.median():,.0f} ns")
    elif args.cmd == "compare":
        df = compare(args.galaxy, args.lb_glob, n_warmup=args.n_warmup)
        _print_df(df)
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
