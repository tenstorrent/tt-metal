#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
import os
import pathlib

DEFAULT_GOLDEN_FILE = os.path.join(
    pathlib.Path(__file__).parent.resolve(),
    "benchmark_rw_buffer_golden.json",
)

DEFAULT_BANDWIDTH_VARIANCE_TOLERANCE_PCT = 5


def parse_args():
    parser = argparse.ArgumentParser(description="Compare performance benchmarks against golden values.")
    parser.add_argument(
        "json",
        type=argparse.FileType("r"),
        help="Path to the JSON file with benchmark results.",
    )
    parser.add_argument(
        "--golden",
        type=argparse.FileType("r"),
        help="Path to the JSON file with golden benchmark values.",
        default=DEFAULT_GOLDEN_FILE,
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        help="Tolerance for benchmark results.",
        default=DEFAULT_BANDWIDTH_VARIANCE_TOLERANCE_PCT,
    )
    args = parser.parse_args()
    return args.golden, args.json, args.tolerance


def collect_benchmarks(benchmark_obj):
    successed_benchmarks, failed_benchmarks = {}, {}
    for benchmark in benchmark_obj["benchmarks"]:
        if "error_occurred" in benchmark and benchmark["error_occurred"]:
            failed_benchmarks[benchmark["name"]] = benchmark["error_message"]
        else:
            successed_benchmarks[benchmark["name"]] = float(benchmark["bytes_per_second"])
    return successed_benchmarks, failed_benchmarks


def compare_benchmarks(golden_benchmarks, result_benchmarks, result_failed_benchmarks, tolerance):
    golden_benchmarks_names = set(golden_benchmarks.keys())
    result_benchmarks_names = set(result_benchmarks.keys())

    successed_benchmarks = []
    underperforming_benchmarks = []

    for name in golden_benchmarks_names & result_benchmarks_names:
        golden_value = golden_benchmarks[name]
        result_value = result_benchmarks[name]

        pct_diff = ((result_value - golden_value) / golden_value) * 100
        if abs(pct_diff) > tolerance:
            underperforming_benchmarks.append((name, result_value, golden_value, pct_diff))
        else:
            successed_benchmarks.append((name, result_value, golden_value, pct_diff))

    success = (
        golden_benchmarks_names == result_benchmarks_names
        and not underperforming_benchmarks
        and not result_failed_benchmarks
    )

    print(f"Benchmark {'FAILED' if not success else 'PASSED'}:")
    print("----------------------------------------------------")
    print(f"# normal benchmarks: {len(successed_benchmarks)}")
    print(f"# failed benchmarks: {len(result_failed_benchmarks)}")
    print(f"# benchmarks outside of expected variance: {len(underperforming_benchmarks)}")
    print(f"# mismatched benchmarks: {len(golden_benchmarks_names ^ result_benchmarks_names)}")
    print(f"# total benchmarks: {len(result_benchmarks_names) + len(result_failed_benchmarks)}")
    print("----------------------------------------------------")

    print("Successed benchmarks: (bytes per second, result vs golden)")
    for name, result_value, golden_value, pct_diff in successed_benchmarks:
        print(f"PASSED | Benchmark {name}: {result_value:.2f} vs {golden_value:.2f} (diff: {pct_diff:+.2f}%)")
    print("----------------------------------------------------")

    print("Failed benchmarks:")
    for name, error_message in result_failed_benchmarks.items():
        print(f"FAILED | Benchmark {name} failed to run: {error_message}")
    print("----------------------------------------------------")

    print("Missing benchmarks:")
    for name in golden_benchmarks_names - result_benchmarks_names:
        print(f"FAILED | Benchmark {name}: missing from results")
    for name in result_benchmarks_names - golden_benchmarks_names:
        print(f"FAILED | Benchmark {name}: excess from results")
    print("----------------------------------------------------")

    print("Benchmarks outside of expected variance: (bytes per second, result vs golden)")
    for name, result_value, golden_value, pct_diff in sorted(underperforming_benchmarks, key=lambda x: x[3]):
        print(f"FAILED | Benchmark {name}: {result_value:.2f} vs {golden_value:.2f} (diff: {pct_diff:+.2f}%)")
    print("----------------------------------------------------")

    return success


if __name__ == "__main__":
    golden_file, result_file, tolerance = parse_args()

    golden_benchmarks, golden_failed_benchmarks = collect_benchmarks(json.load(golden_file))
    result_benchmarks, result_failed_benchmarks = collect_benchmarks(json.load(result_file))

    assert not golden_failed_benchmarks, "Golden benchmarks should not have failed benchmarks"

    print(f"Comparing throughput benchmarks ({golden_file} vs {result_file})...")
    print("Note: Benchmark name follows: Function/ Page Size/ Transfer Size/ Device ID")
    print(f"Tolerance is {tolerance}%.")
    print("----------------------------------------------------")

    if not compare_benchmarks(golden_benchmarks, result_benchmarks, result_failed_benchmarks, tolerance):
        sys.exit("Some benchmarks did not meet the golden values. Please review the output above.")
