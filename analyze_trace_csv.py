#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Summarize an ops_perf_results CSV from an MLA CCL perf run (test_mla_ccl_perf.py).

CRITICAL: under metal trace, tracy writes TWO rows per captured op -- a capture record
(METAL TRACE ID empty, DEVICE KERNEL DURATION BLANK) and a replay record (METAL TRACE ID set,
real device time). Averaging the blanks as 0 gives garbage. This script keeps only rows with a
real DEVICE KERNEL DURATION, and (for trace runs) prefers the replay rows.

Also reports utilization against the perf-model floor: PM IDEAL [ns] (the analytical roofline from
the op's create_op_performance_model) divided by the achieved crit-path. PM IDEAL is only present
when the run was profiled with op profiling on (tracy -p / TTNN_OP_PROFILER=1).

Usage:
    python analyze_trace_csv.py <ops_perf_results_*.csv> [op_code]
    # op_code defaults to AllGatherDeviceOperation (the new top-level ttnn.all_gather)
    # reduce_scatter ops use ReduceScatterMinimalAsyncDeviceOperation
"""
import csv
import statistics
import sys
from collections import defaultdict


def main():
    path = sys.argv[1]
    op_code = sys.argv[2] if len(sys.argv) > 2 else "AllGatherDeviceOperation"
    rows = [r for r in csv.DictReader(open(path)) if r["OP CODE"] == op_code]
    real = [r for r in rows if r["DEVICE KERNEL DURATION [ns]"].strip() != ""]
    replay = [r for r in real if r["METAL TRACE ID"].strip() != ""]
    use = replay if replay else real  # trace -> replay rows; untraced -> all real rows
    blanks = len(rows) - len(real)

    perdev = defaultdict(list)
    for r in use:
        perdev[int(r["DEVICE ID"])].append(float(r["DEVICE KERNEL DURATION [ns]"]) / 1000)
    ndev = len(perdev)
    n = min(len(v) for v in perdev.values())
    aligned = {d: perdev[d][-n:] for d in perdev}  # tail-align, drops leading cold op
    crit = [max(aligned[d][i] for d in aligned) for i in range(n)]
    within = [max(aligned[d][i] for d in aligned) - min(aligned[d][i] for d in aligned) for i in range(n)]
    devmeans = sorted(statistics.mean(v) for v in aligned.values())

    # PM IDEAL [ns]: the analytical (roofline) floor from create_op_performance_model -- constant
    # across rows of an op. Read the first non-blank value (capture rows may carry it too).
    pm_ideal_us = None
    for r in rows:
        v = r.get("PM IDEAL [ns]", "").strip()
        if v not in ("", "0"):
            pm_ideal_us = float(v) / 1000
            break

    crit_mean = statistics.mean(crit)
    print(f"file: {path}")
    print(f"op: {op_code}  | total rows {len(rows)}  real {len(real)}  replay {len(replay)}  blank(capture) {blanks}")
    print(f"mode: {'TRACE (replay rows)' if replay else 'UNTRACED (all real rows)'}  | {ndev} devices x {n} iters")
    print(
        f"  CROSS-DEVICE (per-dev mean):  {min(devmeans):.1f}..{max(devmeans):.1f} us  "
        f"spread {max(devmeans)/min(devmeans):.3f}x  range {max(devmeans)-min(devmeans):.1f}us"
    )
    print(
        f"  CRIT-PATH (max over devs/iter): mean {crit_mean:.1f}  "
        f"min {min(crit):.1f}  max {max(crit):.1f}  stdev {statistics.pstdev(crit):.2f}"
    )
    print(f"  WITHIN-ITER spread (max-min): mean {statistics.mean(within):.1f}us")
    # Utilization vs the perf-model floor (PM IDEAL / achieved). Report against the crit-path mean
    # and against the best (min) iteration -- the latter is the issue-#47803 "best-achieved" figure.
    if pm_ideal_us is not None:
        print(
            f"  PM IDEAL: {pm_ideal_us:.1f}us  |  UTILIZATION: "
            f"{pm_ideal_us / crit_mean * 100:.1f}% vs crit-path mean, "
            f"{pm_ideal_us / min(crit) * 100:.1f}% vs best iter"
        )
    else:
        print("  PM IDEAL: not found in CSV (run under tracy -p / TTNN_OP_PROFILER=1 to populate it)")


if __name__ == "__main__":
    main()
