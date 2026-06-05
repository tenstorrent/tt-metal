#!/usr/bin/env python3
"""Print the op name and device kernel duration for each row of an ops perf CSV.

Reads the `OP CODE` and `DEVICE KERNEL DURATION [ns]` columns from a
profiler `ops_perf_results_*.csv` report and prints one line per op.
"""

import argparse
import csv
import sys
import glob
import os

OP_NAME_COL = "OP CODE"
DURATION_COL = "DEVICE KERNEL DURATION [ns]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="",
        help="Path to the ops_perf_results CSV (default: %(default)s)",
    )
    args = parser.parse_args()
    print(args.csv_path)
    if args.csv_path == "":
        base_path = "/home/ttuser/smanoj/metal_2/generated/profiler/reports/decode"
        files = glob.glob(os.path.join(base_path, "**", "ops_perf_results_*.csv"), recursive=True)
        files.sort(key=os.path.getmtime)
        print(files[-1])
        input_file = files[-1]
    else:
        input_file = args.csv_path
    with open(input_file, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"error: empty or invalid CSV: {args.csv_path}", file=sys.stderr)
            return 1
        for col in (OP_NAME_COL, DURATION_COL):
            if col not in reader.fieldnames:
                print(f"error: column '{col}' not found in {args.csv_path}", file=sys.stderr)
                return 1

        for row in reader:
            op_name = (row.get(OP_NAME_COL) or "").strip()
            duration = (row.get(DURATION_COL) or "").strip()
            if not op_name and not duration:
                continue
            print(f"{op_name}\t{duration or 'N/A'} ns")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
