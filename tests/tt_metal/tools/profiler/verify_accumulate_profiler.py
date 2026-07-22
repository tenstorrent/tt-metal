#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sanity check that L1-accumulate device profiling accumulates worker zones, runs the DRAM-push flush, coexists with dispatch-core profiling, and skips the per-op perf report."""

import sys
from pathlib import Path

from tracy.common import (
    PROFILER_LOGS_DIR,
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_CPP_DEVICE_PERF_REPORT,
)

# Lower bound well above one program's marker rows but below what ~1000 accumulated invocations yield.
MIN_ACCUMULATED_WORKER_ROWS = 20000


def main():
    device_log = Path(PROFILER_LOGS_DIR) / PROFILER_DEVICE_SIDE_LOG
    if not device_log.exists():
        sys.exit(f"FAIL: device log was not produced: {device_log}")

    text = device_log.read_text()
    worker_rows = [line for line in text.splitlines() if ("BRISC" in line or "NCRISC" in line or "TRISC" in line)]

    if len(worker_rows) < MIN_ACCUMULATED_WORKER_ROWS:
        sys.exit(
            f"FAIL: expected accumulated worker zones (>= {MIN_ACCUMULATED_WORKER_ROWS} rows), "
            f"got {len(worker_rows)} -- accumulation did not occur"
        )

    if "PROFILER-DRAM-PUSH" not in text:
        sys.exit("FAIL: no PROFILER-DRAM-PUSH zone in device log -- accumulate flush path did not run")

    if "CQ-DISPATCH" not in text:
        sys.exit(
            "FAIL: no CQ-DISPATCH zone in device log -- dispatch-core profiling did not run alongside "
            "accumulate (coexistence broken or --profile-dispatch-cores not honored)"
        )

    perf_report = Path(PROFILER_LOGS_DIR) / PROFILER_CPP_DEVICE_PERF_REPORT
    if perf_report.exists():
        sys.exit(f"FAIL: per-op perf report must be skipped in accumulate mode but exists: {perf_report}")

    print(
        f"PASS: accumulate sanity OK -- {len(worker_rows)} accumulated worker rows, "
        f"PROFILER-DRAM-PUSH present, CQ-DISPATCH present (dispatch coexistence), perf report correctly skipped"
    )


if __name__ == "__main__":
    main()
