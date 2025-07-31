# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

from models.perf.device_perf_utils import check_device_perf_results
from models.perf.perf_utils import merge_perf_files, today

DEFAULT_FILENAME = f"Models_Device_Perf_{today}.csv"

expected_cols = [
    "Model",
    "Setting",
    "Batch",
    "AVG DEVICE FW SAMPLES/S",
    "MIN DEVICE FW SAMPLES/S",
    "MAX DEVICE FW SAMPLES/S",
    "AVG DEVICE FW DURATION [ns]",
    "MIN DEVICE FW DURATION [ns]",
    "MAX DEVICE FW DURATION [ns]",
    "AVG DEVICE KERNEL SAMPLES/S",
    "Lower Threshold AVG DEVICE KERNEL SAMPLES/S",
    "Upper Threshold AVG DEVICE KERNEL SAMPLES/S",
    "MIN DEVICE KERNEL SAMPLES/S",
    "MAX DEVICE KERNEL SAMPLES/S",
    "AVG DEVICE KERNEL DURATION [ns]",
    "MIN DEVICE KERNEL DURATION [ns]",
    "MAX DEVICE KERNEL DURATION [ns]",
    "AVG DEVICE BRISC KERNEL SAMPLES/S",
    "MIN DEVICE BRISC KERNEL SAMPLES/S",
    "MAX DEVICE BRISC KERNEL SAMPLES/S",
    "AVG DEVICE BRISC KERNEL DURATION [ns]",
    "MIN DEVICE BRISC KERNEL DURATION [ns]",
    "MAX DEVICE BRISC KERNEL DURATION [ns]",
]

check_cols = ["AVG DEVICE KERNEL SAMPLES/S"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merges device performance CSV reports")
    parser.add_argument(
        "output_filename",
        type=Path,
        nargs="?",
        default=DEFAULT_FILENAME,
        help="The output filename",
    )
    parser.add_argument(
        "devperf",
        type=Path,
        nargs="?",
        default="",
        help="model name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert (
        args.output_filename
    ), f"Expected user to provide an output filename for merged report (arguments provided were {args})"
    if str(args.devperf) == "REPORT":
        merge_perf_files(args.output_filename, f"device_perf", expected_cols)
    elif str(args.devperf) == "CHECK":
        check_device_perf_results(args.output_filename, expected_cols, check_cols)
    else:
        merge_perf_files(args.output_filename, f"device_perf", expected_cols)
        check_device_perf_results(args.output_filename, expected_cols, check_cols)
