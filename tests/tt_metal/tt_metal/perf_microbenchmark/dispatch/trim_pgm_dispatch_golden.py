#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Trim aggregate entries from pgm dispatch golden JSON files.

The compare_pgm_dispatch_perf_ci.py comparison script only uses:
  - repetitions=1: the single iteration entry
  - repetitions=2: the min of the two iteration entries
  - repetitions>2: the median aggregate entry

All other aggregate entries (mean, stddev, cv, and median when
repetitions<=2) are unused and can be removed to reduce file size.
"""

import argparse
import json


def trim_benchmarks(data):
    kept = []
    for b in data["benchmarks"]:
        run_type = b.get("run_type", "iteration")
        reps = b.get("repetitions", 1)
        agg = b.get("aggregate_name", "")

        if run_type == "iteration":
            kept.append(b)
        elif reps > 2 and agg == "median":
            kept.append(b)

    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Golden JSON files to trim in place")
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print stats without modifying files",
    )
    args = parser.parse_args()

    for path in args.files:
        with open(path) as f:
            data = json.load(f)

        original_count = len(data["benchmarks"])
        data["benchmarks"] = trim_benchmarks(data)
        trimmed_count = len(data["benchmarks"])

        print(f"{path}: {original_count} -> {trimmed_count} benchmarks")

        if not args.dry_run:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
                f.write("\n")

    if args.dry_run:
        print("(dry run, no files modified)")


if __name__ == "__main__":
    main()
