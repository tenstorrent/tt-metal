#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run ring joint SDPA perf test with tracy and check for WAIT-K spikes."""

import csv
import subprocess
import sys
import glob
import os

SPIKE_THRESHOLD = 70  # cycles
TEST_PATH = "tests/nightly/blackhole/ccl/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_sweep_perf_impl"


def run_test():
    """Run the perf test with tracy profiling and return the profile CSV path."""
    cmd = ["python", "-m", "tracy", "-r", "-m", "-p", f"pytest {TEST_PATH}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    # Extract the report directory from tracy output
    for line in output.splitlines():
        if "OPs csv generated at:" in line:
            ops_csv = line.split("OPs csv generated at:")[-1].strip()
            report_dir = os.path.dirname(ops_csv)
            profile_csv = os.path.join(report_dir, "profile_log_device.csv")
            if os.path.exists(profile_csv):
                return profile_csv

    # Fallback: find latest profile
    files = glob.glob("generated/profiler/reports/*/profile_log_device.csv")
    if files:
        return max(files, key=os.path.getmtime)
    return None


def check_spike(profile_csv):
    """Parse profile CSV and return peak WAIT-K duration in cycles."""
    starts = {}
    peak = 0
    with open(profile_csv) as f:
        reader = csv.reader(f)
        next(reader)  # ARCH header
        next(reader)  # column header
        for row in reader:
            if len(row) < 12 or "WAIT-K" not in row[10]:
                continue
            key = (row[0], row[1], row[2], row[3])
            cycles = int(row[5])
            ztype = row[11]
            if ztype == "ZONE_START":
                starts[key] = cycles
            elif ztype == "ZONE_END" and key in starts:
                dur = cycles - starts[key]
                if dur > peak:
                    peak = dur
                del starts[key]
    return peak


if __name__ == "__main__":
    print("Running perf test with tracy...")
    profile_csv = run_test()
    if not profile_csv:
        print("No profile_log_device.csv found after test run")
        sys.exit(1)

    print(f"Analyzing {profile_csv}")
    peak = check_spike(profile_csv)
    has_spike = peak > SPIKE_THRESHOLD
    print(f"Spike: {'YES' if has_spike else 'NO'}")
    print(f"Peak: {peak} cycles")
    sys.exit(1 if has_spike else 0)
