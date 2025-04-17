#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import sys
import pathlib


THRESHOLD_PCT = 5

parser = argparse.ArgumentParser(description="Compare benchmark JSON to golden")
parser.add_argument("json", help="JSON file to compare", type=argparse.FileType("r"))
parser.add_argument(
    "-g",
    "--golden",
    help="Golden JSON file",
    type=argparse.FileType("r"),
    default=os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        "pgm_dispatch_golden.json",
    ),
)
args = parser.parse_args()

golden = json.load(args.golden)

result = json.load(args.json)

golden_benchmarks = {}


def add_benchmark_iterations(input):
    out = {}
    for benchmark in input["benchmarks"]:
        if benchmark["repetitions"] == 1:
            out[benchmark["name"]] = benchmark
        elif benchmark["run_type"] == "aggregate" and benchmark["aggregate_name"] == "mean":
            out[benchmark["name"].replace("_mean", "")] = benchmark
    return out


golden_benchmarks = add_benchmark_iterations(golden)

result_benchmarks = add_benchmark_iterations(result)

exit_code = 0

for name, benchmark in golden_benchmarks.items():
    if name not in result_benchmarks:
        print(f"Error: Golden benchmark {name} missing from results")
        exit_code = 1
        continue
    result = result_benchmarks[benchmark["name"]]

    if "error_occurred" in benchmark:
        if "error_occurred" not in result:
            result_time = result["IterationTime"] * 1000000
            print(f"Consider adjusting baselines. Error in {name} was fixed in result (with time {result_time:.2f}us).")
        continue

    if "error_occurred" in result:
        if "error_occurred" not in benchmark:
            print(f"Error: Benchmark {name} gave unexpected error: {result['error_message']}")
            exit_code = 1
        continue

    golden_time = benchmark["IterationTime"] * 1000000
    result_time = result["IterationTime"] * 1000000
    result_diff_pct = result_time / golden_time * 100 - 100

    golden_clock = benchmark.get("Clock", "?")
    result_clock = result.get("Clock", "?")

    clock_string = (
        f"(Clock now {result_clock}MHz but golden was {golden_clock}MHz)" if result_clock != golden_clock else ""
    )

    if result_diff_pct > THRESHOLD_PCT:
        print(
            f"Error:Test {name} expected value {golden_time:.2f}us but got {result_time:.2f}us ({result_diff_pct:.2f}% worse) {clock_string}"
        )
        exit_code = 1
    if result_diff_pct < -THRESHOLD_PCT:
        print(
            f"Consider adjusting baselines. Test {name} got value {result_time:.2f}us but expected {golden_time:.2f}us ({-result_diff_pct:.2f}% better) {clock_string}."
        )

for name in result_benchmarks:
    if name not in golden_benchmarks:
        print(f"Error: Result benchmark {name} missing from goldens")
        exit_code = 1

if exit_code == 0:
    print("Test successful")
sys.exit(exit_code)
