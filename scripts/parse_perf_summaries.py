#!/usr/bin/env python3
import csv
import glob
import os
import sys
from typing import Dict, List, Tuple


def parse_stacked_csv(path: str) -> Tuple[float, float]:
    """
    Return (total_ms, total_excl_ms) where:
    - total_ms sums 'Device Time Sum [μs]' across all rows
    - total_excl_ms excludes rows whose 'Op Code' contains 'AllGather'
    """
    total_us = 0.0
    excl_us = 0.0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Column name in stacked CSVs
        col = "Device Time Sum [μs]"
        op_col = "Op Code"
        if col not in reader.fieldnames:
            raise RuntimeError(f"{path}: Missing column '{col}'")
        for row in reader:
            val = row.get(col, "").strip()
            if not val:
                continue
            try:
                us = float(val)
                total_us += us
                op = (row.get(op_col, "") or "").strip()
                # Exclude only pure, unfused AllGather ops (Pre/Post/Async), but include
                # any MinimalMatmul variants that contain AllGather in the name.
                is_pure_allgather = ("AllGather" in op) and ("MinimalMatmul" not in op)
                if not is_pure_allgather:
                    excl_us += us
            except ValueError:
                # Skip non-numeric
                continue
    return total_us / 1000.0, excl_us / 1000.0  # ms


def main(argv: List[str]) -> int:
    pattern = os.environ.get("PERF_STACKED_GLOB", "perf-summaries/ops_perf_summary_*_stacked.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matched {pattern}", file=sys.stderr)
        return 1

    results: List[Tuple[str, float, float]] = []
    for path in files:
        test_id = os.path.basename(path)
        # ops_perf_summary_<ID>_stacked.csv -> extract <ID>
        if test_id.startswith("ops_perf_summary_") and test_id.endswith("_stacked.csv"):
            test_id = test_id[len("ops_perf_summary_") : -len("_stacked.csv")]
        total_ms, excl_ms = parse_stacked_csv(path)
        results.append((test_id, total_ms, excl_ms))

    # Print as a minimal table
    name_w = max(12, max(len(k) for k, _, _ in results))
    print(f"{'Test ID'.ljust(name_w)} | {'Device Time (ms)'} | {'No AllGather (ms)'}")
    print(f"{'-'*name_w}-+-{'-'*16}-+-{'-'*17}")
    for k, v, e in results:
        print(f"{k.ljust(name_w)} | {v:>16.2f} | {e:>17.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
