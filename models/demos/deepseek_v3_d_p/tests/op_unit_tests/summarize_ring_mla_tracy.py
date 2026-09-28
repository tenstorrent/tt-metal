# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Summarize traced ring_mla replays from a tracy ops CSV (test_ring_mla_chunk_sweep.py).

Per trace replay: critical path = max DEVICE KERNEL DURATION over the 32 devices. Reports the
median over replays, CORE COUNT (includes the CCL workers), the SDPA grid from ATTRIBUTES,
PM IDEAL / PM FPU UTIL, and the Q/KV shapes that identify the config.

    python models/demos/deepseek_v3_d_p/tests/op_unit_tests/summarize_ring_mla_tracy.py [csv]

Without a path, the newest generated/profiler/reports/*/ops_perf_results_*.csv is used.
"""

import argparse
import glob
import os
import re
import statistics
import sys

import pandas as pd

OP = "RingJointSDPADeviceOperation"
DUR = "DEVICE KERNEL DURATION [ns]"
REPLAY = "METAL TRACE REPLAY SESSION ID"


def _latest_csv():
    paths = glob.glob("generated/profiler/reports/**/ops_perf_results_*.csv", recursive=True)
    if not paths:
        sys.exit("no ops_perf_results CSV under generated/profiler/reports")
    return max(paths, key=os.path.getmtime)


def _sdpa_grid(attrs):
    m = re.search(r"compute_with_storage_grid_size=(\d+)-(\d+)", str(attrs))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _chunk_sizes(attrs):
    q = re.search(r"q_chunk_size=(\d+)", str(attrs))
    k = re.search(r"k_chunk_size=(\d+)", str(attrs))
    return (int(q.group(1)) if q else None, int(k.group(1)) if k else None)


def summarize(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["OP CODE"] == OP]
    if df.empty:
        sys.exit(f"no {OP} rows in {csv_path}")
    traced = df[df[REPLAY].notna()] if REPLAY in df.columns else df.iloc[0:0]
    if traced.empty:
        sys.exit(f"no traced {OP} replays in {csv_path}")

    # The first replay is the test's warm replay (outside the signposts); drop it.
    traced = traced[traced[REPLAY] != traced[REPLAY].min()]
    per_replay = traced.groupby(REPLAY).agg(
        dur=(DUR, "max"),
        dur_min=(DUR, "min"),
        devices=("DEVICE ID", "nunique"),
        cores=("CORE COUNT", "max"),
    )
    first = traced.iloc[0]
    grid = _sdpa_grid(first.get("ATTRIBUTES"))
    q_chunk, k_chunk = _chunk_sizes(first.get("ATTRIBUTES"))
    ideal = traced["PM IDEAL [ns]"].max() if "PM IDEAL [ns]" in traced.columns else float("nan")
    median = statistics.median(per_replay["dur"])
    out = {
        "csv": csv_path,
        "replays": len(per_replay),
        "devices": int(per_replay["devices"].max()),
        "q_shape": "x".join(str(int(first[f"INPUT_0_{d}_PAD[LOGICAL]"].split("[")[0])) for d in "WZYX")
        if "INPUT_0_W_PAD[LOGICAL]" in traced.columns
        else None,
        "q_chunk": q_chunk,
        "k_chunk": k_chunk,
        "core_count": int(per_replay["cores"].max()),
        "sdpa_grid": grid,
        "sdpa_cores": grid[0] * grid[1] if grid else None,
        "median_us": round(median / 1e3, 2),
        "min_us": round(per_replay["dur"].min() / 1e3, 2),
        "max_us": round(per_replay["dur"].max() / 1e3, 2),
        "device_skew_us": round(float((per_replay["dur"] - per_replay["dur_min"]).median()) / 1e3, 2),
        "pm_ideal_us": round(ideal / 1e3, 2),
        "pm_fpu_util_pct": round(100 * ideal / median, 2) if ideal == ideal else None,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    args = ap.parse_args()
    for k, v in summarize(args.csv or _latest_csv()).items():
        print(f"{k:>16}: {v}")


if __name__ == "__main__":
    main()
