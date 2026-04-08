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

from util_report_iter import filter_last_steady_state_iteration


def main():
    parser = argparse.ArgumentParser(description="Filter utilization report to last steady-state iteration")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file path")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file path")
    args = parser.parse_args()

    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  {len(df)} rows loaded")

    filtered = filter_last_steady_state_iteration(df, log=print)
    print(f"  {len(filtered)} rows after filtering")

    filtered.to_csv(args.output, index=False)
    print(f"Done! Filtered report saved to {args.output}")


if __name__ == "__main__":
    main()
