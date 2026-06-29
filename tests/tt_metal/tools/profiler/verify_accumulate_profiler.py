#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sanity check for L1-accumulate device profiling alongside dispatch-core profiling.

Run AFTER a multi-invocation workload has executed under
``python -m tracy -p --enable-accumulate-profiling --profile-dispatch-cores ...``.
Verifies that:
  1. the device log was produced and contains worker-core zones,
  2. zones from many invocations were accumulated (not just one program's worth),
  3. the accumulate flush path ran (PROFILER-DRAM-PUSH zones present),
  4. dispatch-core profiling coexisted (CQ-DISPATCH zones present; the run completing at
     all means no start/end marker mismatch between the accumulating workers and the
     classic-path dispatch cores),
  5. the per-op perf report was skipped (meaningless without per-op IDs in accumulate).

Exits non-zero with a clear message on any failure.
"""

import sys
from pathlib import Path

from tracy.common import (
    PROFILER_LOGS_DIR,
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_CPP_DEVICE_PERF_REPORT,
)

# Single uninstrumented program across a worker grid produces at most a few thousand
# marker rows; accumulating ~1000 invocations produces far more. Use a lower bound well
# above one program but well below what accumulation yields.
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
