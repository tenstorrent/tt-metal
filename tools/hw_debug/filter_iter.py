#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Filter a utilization report CSV to keep only the last steady-state iteration.

Detects the iteration boundary by finding the longest repeating op-code suffix
with the most consecutive repetitions, then keeps only the final occurrence.

Usage:
  python filter_iter.py -i model_util_report.csv -o filtered_report.csv
"""

import argparse
import pandas as pd


def filter_last_steady_state_iteration(df):
    npe_col = "NOC UTIL (%)"
    if npe_col not in df.columns:
        return df

    valid = df[df[npe_col].notna()].copy()
    if len(valid) == 0:
        print("  Warning: No rows with NOC UTIL data found")
        return df

    op_codes = valid["OP CODE"].tolist()
    n = len(op_codes)
    best_size = None
    best_reps = 0
    for size in range(1, n // 2 + 1):
        reps = 1
        while reps * size + size <= n:
            if op_codes[n - (reps + 1) * size : n - reps * size] == op_codes[n - size :]:
                reps += 1
            else:
                break
        if reps >= 2 and (best_size is None or reps > best_reps or (reps == best_reps and size < best_size)):
            best_size = size
            best_reps = reps

    if best_size is not None:
        result = valid.iloc[-best_size:]
        print(f"  Detected iteration size: {best_size} ops ({best_reps} repetitions found), keeping last iteration")
        return result.reset_index(drop=True)
    else:
        print("  Warning: No repeating iteration pattern found, keeping all rows with NOC data")
        return valid.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Filter utilization report to last steady-state iteration")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file path")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file path")
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} rows loaded")

    filtered = filter_last_steady_state_iteration(df)
    print(f"  {len(filtered)} rows after filtering")

    filtered.to_csv(args.output, index=False)
    print(f"Done! Filtered report saved to {args.output}")


if __name__ == "__main__":
    main()
