#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Trim rw_buffer benchmark golden JSON files to reduce size.

The compare_benchmark_rw_buffer.py comparison script only uses median
aggregate entries and reads these fields per entry:
  - name, repetitions, run_type, aggregate_name, bytes_per_second

This script:
  1. Keeps only median aggregate entries (the only ones used for comparison).
  2. Strips fields from each entry that the comparison never reads.
  3. Preserves the top-level context block unchanged (machine/run metadata).
"""

import argparse
import json
import os

KEEP_BENCHMARK_FIELDS = {
    "name",
    "repetitions",
    "run_type",
    "aggregate_name",
    "bytes_per_second",
    "error_occurred",
    "error_message",
}


def trim_benchmarks(benchmarks):
    kept = []
    for b in benchmarks:
        if b.get("run_type") != "aggregate" or b.get("aggregate_name") != "median":
            continue
        trimmed = {k: v for k, v in b.items() if k in KEEP_BENCHMARK_FIELDS}
        kept.append(trimmed)
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
        data["benchmarks"] = trim_benchmarks(data["benchmarks"])
        trimmed_count = len(data["benchmarks"])

        original_size = os.path.getsize(path)
        new_content = json.dumps(data, indent=2) + "\n"
        new_size = len(new_content.encode())

        print(
            f"{path}: {original_count} -> {trimmed_count} benchmarks, "
            f"{original_size / 1024:.1f} KB -> {new_size / 1024:.1f} KB"
        )

        if not args.dry_run:
            with open(path, "w") as f:
                f.write(new_content)

    if args.dry_run:
        print("(dry run, no files modified)")


if __name__ == "__main__":
    main()
