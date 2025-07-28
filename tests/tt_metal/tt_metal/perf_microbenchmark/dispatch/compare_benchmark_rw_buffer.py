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

BANDWIDTH_VAIRANCE_TOLERANCE_PCT = 5

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare performance benchmarks against golden values."
    )
    parser.add_argument(
        "--json",
        type=argparse.FileType("r"),
        required=True,
        help="Path to the JSON file with benchmark results.",
    )
    parser.add_argument(
        "--golden",
        type=argparse.FileType("r"),
        help="Path to the JSON file with golden benchmark values.",
        default=DEFAULT_GOLDEN_FILE,
    )
    args = parser.parse_args()
    return args.golden, args.json

def collect_benchmarks(benchmark_obj):
    result = {}
    for benchmark in benchmark_obj["benchmarks"]:
        result[benchmark["name"]] = float(benchmark["bytes_per_second"])
    return result

def compare_benchmarks(golden_benchmarks, result_benchmarks):
    """
    >>> BANDWIDTH_VAIRANCE_TOLERANCE_PCT = 5
    >>> golden = {'bench1': 100.0, 'bench2': 200.0}
    >>> result = {'bench1': 102.0, 'bench2': 212.0}
    >>> compare_benchmarks(golden, result)
    Benchmark bench1 passed: 102.0 vs 100.0 (diff: 2.00%)
    Benchmark bench2 failed: 212.0 vs 200.0 (diff: 6.00%)
    False

    >>> BANDWIDTH_VAIRANCE_TOLERANCE_PCT = 10
    >>> golden = {'bench1': 100.0}
    >>> result = {'bench1': 105.0}
    >>> compare_benchmarks(golden, result)
    Benchmark bench1 passed: 105.0 vs 100.0 (diff: 5.00%)
    True

    >>> BANDWIDTH_VAIRANCE_TOLERANCE_PCT = 5
    >>> golden = {'bench1': 100.0}
    >>> result = {}
    >>> compare_benchmarks(golden, result)
    Benchmark bench1 failed: missing from results
    False
    """
    success = True
    for name, golden_value in golden_benchmarks.items():
        if name not in result_benchmarks:
            print(f"FAILED | Benchmark {name}: missing from results")
            success = False
            continue

        result_value = result_benchmarks[name]
        pct_diff = abs((result_value - golden_value) / golden_value) * 100

        if pct_diff > BANDWIDTH_VAIRANCE_TOLERANCE_PCT:
            print(f"FAILED | Benchmark {name}: {result_value:.2f} vs {golden_value:.2f} (diff: {pct_diff:.2f}%)")
            success = False
        else:
            print(f"PASSED | Benchmark {name}: {result_value:.2f} vs {golden_value:.2f} (diff: {pct_diff:.2f}%)")
    return success

if __name__ == "__main__":
    golden_file, result_file = parse_args()

    golden_benchmarks = collect_benchmarks(json.load(golden_file))
    result_benchmarks = collect_benchmarks(json.load(result_file))

    print("Comparing throughput benchmarks...")
    print("Note: Benchmark name follows: Function/ Page Size/ Transfer Size/ Device ID")
    print("Benchmark results are in bytes per second.")
    print("----------------------------------------------------")

    if not compare_benchmarks(golden_benchmarks, result_benchmarks):
        sys.exit("Some benchmarks did not meet the golden values. Please review the output above.")
