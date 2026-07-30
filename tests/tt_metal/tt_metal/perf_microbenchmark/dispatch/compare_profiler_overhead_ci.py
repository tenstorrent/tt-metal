#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Compare device-profiler-overhead benchmark JSON against a golden.

These benchmarks report host-side latency via Google Benchmark manual time (the
``real_time`` field, in nanoseconds): the cost of ``ReadMeshDeviceProfilerResults`` (the
profiler-buffer writeout) and of profiler-enabled dispatch. A result is a regression if it
is more than ``--tolerance`` percent slower than the golden. Being faster only prints an
advisory to refresh the golden. ``--ignore-times`` performs a structural check only (names
must match, no benchmark errored), useful while a new benchmark's timings are still being
characterized on a given SKU.
"""

import argparse
import json
import os
import pathlib
import sys

# The profiler read path (L1->DRAM writeout + CSV post-process) is inherently more variable than a
# pure host-latency benchmark, so a generous tolerance is used: a real regression in the read path
# is far larger than this, while normal run-to-run noise stays well under it.
DEFAULT_TOLERANCE_PCT = 25

DEFAULT_GOLDEN_FILE = os.path.join(
    pathlib.Path(__file__).parent.resolve(),
    "profiler_overhead_golden.json",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare profiler-overhead benchmark JSON to golden")
    parser.add_argument("json", type=argparse.FileType("r"), help="JSON file to compare")
    parser.add_argument("-g", "--golden", type=argparse.FileType("r"), default=None, help="Golden JSON file")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_PCT, help="Regression tolerance percent")
    parser.add_argument("--ignore-times", action="store_true", help="Structural check only; do not fail on timings")
    return parser.parse_args()


def collect(bench_obj):
    """Return {name: real_time_ns} and {name: error_message}, collapsing repetitions.

    For a single repetition, the lone iteration entry is used. For multiple repetitions, the
    ``min`` aggregate is used (its name is normalized by stripping the ``_min`` suffix): a slow
    flake on a contended host only inflates samples, so the minimum is the most representative
    measure of true overhead, while a genuine regression still raises it.
    """
    times, errors = {}, {}
    for b in bench_obj["benchmarks"]:
        name = b["name"]
        if b.get("error_occurred"):
            errors[name] = b.get("error_message", "unknown error")
            continue
        if b.get("run_type") == "aggregate":
            if b.get("aggregate_name") != "min":
                continue
            name = name.removesuffix("_min")
        elif b.get("repetitions", 1) != 1:
            # A raw iteration entry from a multi-repetition run; skip in favor of the min aggregate.
            continue
        if "real_time" in b:
            times[name] = float(b["real_time"])
    return times, errors


def main():
    args = parse_args()
    golden_file = args.golden or open(DEFAULT_GOLDEN_FILE, "r")

    golden_times, golden_errors = collect(json.load(golden_file))
    result_times, result_errors = collect(json.load(args.json))

    assert not golden_errors, f"Golden should not contain errored benchmarks: {list(golden_errors)}"

    exit_code = 0

    for name in result_errors:
        print(f"Error: Benchmark {name} gave unexpected error: {result_errors[name]}")
        exit_code = 1

    for name, golden_ns in golden_times.items():
        if name not in result_times:
            print(f"Error: Golden benchmark {name} missing from results")
            exit_code = 1
            continue
        result_ns = result_times[name]
        diff_pct = result_ns / golden_ns * 100 - 100
        if diff_pct > args.tolerance:
            msg = f"Test {name} expected {golden_ns:.0f}ns but got {result_ns:.0f}ns ({diff_pct:.2f}% worse)"
            if args.ignore_times:
                print(f"Advisory (times ignored): {msg}")
            else:
                print(f"Error: {msg}")
                exit_code = 1
        elif diff_pct < -args.tolerance:
            print(
                f"Consider adjusting baselines. Test {name} got {result_ns:.0f}ns but expected "
                f"{golden_ns:.0f}ns ({-diff_pct:.2f}% better)."
            )

    for name in result_times:
        if name not in golden_times:
            print(f"Error: Result benchmark {name} missing from goldens")
            exit_code = 1

    if exit_code == 0:
        print("Test successful")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
