#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Script to compare AllGather operation configurations between the fused op unit test
and the module test to ensure they match exactly.

Usage:
    python compare_all_gather_configs.py <fused_op_csv> <module_csv>

Example:
    python compare_all_gather_configs.py \\
        generated/ops_perf_results/deepseek_v3_fused_ops_device_perf/ops_perf_results_*.csv \\
        generated/ops_perf_results/ops_perf_results_*.csv
"""

import sys
from pathlib import Path

import pandas as pd


def extract_all_gather_ops(csv_path: Path) -> pd.DataFrame:
    """Extract AllGather operations from a CSV file."""
    df = pd.read_csv(csv_path)
    # Filter for AllGather operations
    all_gather_ops = df[df["OP CODE"].str.contains("AllGather", case=False, na=False)]
    return all_gather_ops


def compare_op_properties(fused_op_df: pd.DataFrame, module_df: pd.DataFrame) -> bool:
    """
    Compare properties of AllGather operations between fused op test and module test.

    Returns:
        True if all properties match, False otherwise.
    """
    if len(fused_op_df) == 0:
        print("ERROR: No AllGather operations found in fused op test CSV")
        return False

    if len(module_df) == 0:
        print("ERROR: No AllGather operations found in module test CSV")
        return False

    print(f"Found {len(fused_op_df)} AllGather op(s) in fused op test")
    print(f"Found {len(module_df)} AllGather op(s) in module test")
    print()

    # Compare the first AllGather operation from each test
    fused_op = fused_op_df.iloc[0]
    module_op = module_df.iloc[0]

    # Properties to compare
    properties_to_compare = [
        "OP CODE",
        "INPUT_0_SHAPE",
        "INPUT_0_DTYPE",
        "INPUT_0_LAYOUT",
        "INPUT_0_MEMORY_CONFIG",
        "OUTPUT_0_SHAPE",
        "OUTPUT_0_DTYPE",
        "OUTPUT_0_LAYOUT",
        "OUTPUT_0_MEMORY_CONFIG",
    ]

    all_match = True
    print("=" * 80)
    print("COMPARING ALLGATHER OPERATION PROPERTIES")
    print("=" * 80)

    for prop in properties_to_compare:
        if prop not in fused_op.index or prop not in module_op.index:
            print(f"⚠️  Property '{prop}' not found in one or both CSVs")
            continue

        fused_val = fused_op[prop]
        module_val = module_op[prop]

        # Handle NaN values
        if pd.isna(fused_val) and pd.isna(module_val):
            match = True
            status = "✓"
        elif pd.isna(fused_val) or pd.isna(module_val):
            match = False
            status = "✗"
        else:
            match = str(fused_val) == str(module_val)
            status = "✓" if match else "✗"

        if not match:
            all_match = False

        print(f"{status} {prop}:")
        print(f"    Fused Op: {fused_val}")
        print(f"    Module:   {module_val}")
        if not match:
            print(f"    ❌ MISMATCH!")
        print()

    print("=" * 80)
    if all_match:
        print("✅ ALL PROPERTIES MATCH!")
    else:
        print("❌ SOME PROPERTIES DO NOT MATCH!")
    print("=" * 80)

    return all_match


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_all_gather_configs.py <fused_op_csv> <module_csv>")
        print()
        print("Example:")
        print("    python compare_all_gather_configs.py \\")
        print("        generated/ops_perf_results/deepseek_v3_fused_ops_device_perf/ops_perf_results_*.csv \\")
        print("        generated/ops_perf_results/ops_perf_results_*.csv")
        sys.exit(1)

    fused_op_csv = Path(sys.argv[1])
    module_csv = Path(sys.argv[2])

    if not fused_op_csv.exists():
        print(f"ERROR: Fused op CSV file not found: {fused_op_csv}")
        sys.exit(1)

    if not module_csv.exists():
        print(f"ERROR: Module CSV file not found: {module_csv}")
        sys.exit(1)

    print(f"Comparing AllGather configurations:")
    print(f"  Fused Op Test: {fused_op_csv}")
    print(f"  Module Test:   {module_csv}")
    print()

    fused_op_df = extract_all_gather_ops(fused_op_csv)
    module_df = extract_all_gather_ops(module_csv)

    match = compare_op_properties(fused_op_df, module_df)

    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
