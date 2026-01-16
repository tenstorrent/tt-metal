# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

from models.perf.perf_utils import check_perf_results, merge_perf_files, today

expected_cols = [
    "Model",
    "Setting",
    "Batch",
    "First Run (sec)",
    "Second Run (sec)",
    "Compile Time (sec)",
    "Expected Compile Time (sec)",
    "Inference Time (sec)",
    "Expected Inference Time (sec)",
    "Throughput (Batch*inf/sec)",
    "Inference Time CPU (sec)",
    "Throughput CPU (Batch*inf/sec)",
]

check_cols = ["Inference Time (sec)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge device performance CSV reports and validate key results. Use 'REPORT' to generate the merged report, 'CHECK' to validate results, or leave empty to do both."
    )
    parser.add_argument(
        "modelperf",
        type=Path,
        nargs="?",
        default="",
        help="REPORT or CHECK or leave empty to do both",
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=1.0,
        help="Upper threshold multiplier for performance regression checks (default: 1.0)",
    )
    parser.add_argument(
        "--lower-threshold",
        type=float,
        default=0.80,
        help="Lower threshold multiplier for performance improvement checks (default: 0.80)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fname = f"Models_Perf_{today}.csv"
    if str(args.modelperf) == "REPORT":
        merge_perf_files(fname, "perf", expected_cols)
    elif str(args.modelperf) == "CHECK":
        check_perf_results(
            fname, expected_cols, check_cols, lower_threshold=args.lower_threshold, upper_threshold=args.upper_threshold
        )
    else:
        merge_perf_files(fname, "perf", expected_cols)
        check_perf_results(
            fname, expected_cols, check_cols, lower_threshold=args.lower_threshold, upper_threshold=args.upper_threshold
        )
