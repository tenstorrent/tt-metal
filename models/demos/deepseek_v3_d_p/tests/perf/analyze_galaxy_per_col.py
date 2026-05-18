#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-column Galaxy 8x4 aggregation for Dispatch and Combine ops.

Each (layer, op, col) cell = max over 8 chips in that column of the kernel
duration for that op's N-th invocation (= N-th MoE layer). Per-layer max-of-
4-cols matches what `analyze_combine_perf.py galaxy` reports.

Galaxy 8x4 device ID layout (row-major, row=SP axis=8, col=EP axis=4):
    col_idx = DEVICE_ID % 4
    row_idx = DEVICE_ID // 4
    Col 0 chips = {0, 4, 8, 12, 16, 20, 24, 28}
    Col 1 chips = {1, 5, 9, ...}
    ...

Usage:
    python analyze_galaxy_per_col.py <ops_perf_results_*.csv> \\
        --ops Dispatch,Combine \\
        --first-moe-layer 3

Output: per-(op, layer) row with col0..col3 columns and layer_max.
"""
import argparse

import pandas as pd

OP_FULL_NAMES = {
    "Dispatch": "DispatchDeviceOperation",
    "Combine": "CombineDeviceOperation",
}


def galaxy_per_col_per_layer(csv_path: str, op: str, first_moe_layer: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    op_full = OP_FULL_NAMES.get(op, op)
    df = df[df["OP CODE"] == op_full].copy()
    if df.empty:
        raise SystemExit(f"No rows with OP CODE == {op_full!r} in {csv_path}")

    df["dur_ns"] = pd.to_numeric(df["DEVICE KERNEL DURATION [ns]"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["dur_ns"])
    df = df.sort_values(["DEVICE ID", "GLOBAL CALL COUNT"]).reset_index(drop=True)

    # N-th op invocation on each chip = N-th MoE layer for that chip
    df["iter_in_dev"] = df.groupby("DEVICE ID").cumcount()
    df["layer_idx"] = df["iter_in_dev"] + first_moe_layer
    df["col_idx"] = df["DEVICE ID"] % 4

    # Per (layer, col): max across the 8 chips in that column
    per_layer_col = (
        df.groupby(["layer_idx", "col_idx"])["dur_ns"]
        .max()
        .unstack("col_idx")
        .rename(columns={i: f"col{i}_max_chip_ns" for i in range(4)})
    )
    per_layer_col["layer_max_ns"] = per_layer_col.max(axis=1)
    per_layer_col["layer_min_col_ns"] = per_layer_col[[f"col{i}_max_chip_ns" for i in range(4)]].min(axis=1)
    per_layer_col["col_spread_ns"] = per_layer_col["layer_max_ns"] - per_layer_col["layer_min_col_ns"]
    return per_layer_col.reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="Galaxy ops_perf_results_*.csv path")
    ap.add_argument(
        "--ops",
        default="Dispatch,Combine",
        help="Comma list of ops: Dispatch, Combine (default both)",
    )
    ap.add_argument(
        "--first-moe-layer",
        type=int,
        default=3,
        help="Layer index of the first MoE block (default 3, since layers 0-2 are dense)",
    )
    ap.add_argument("--out", help="Optional CSV path to save the combined table")
    args = ap.parse_args()

    all_dfs = []
    for op in [x.strip() for x in args.ops.split(",") if x.strip()]:
        t = galaxy_per_col_per_layer(args.csv, op, args.first_moe_layer)
        t.insert(0, "op", op)
        all_dfs.append(t)
    out = pd.concat(all_dfs, ignore_index=True)

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(out.to_string(index=False))

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
