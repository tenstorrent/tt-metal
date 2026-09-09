# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Re-run tt-metal's Tracy ops post-processing, dropping host ops that have no device record.

``python -m tracy -r`` aborts in ``process_ops_logs._enrich_ops_from_perf_csv`` with
"Device data missing: Op N not present in cpp_device_perf_report.csv" when one host-recorded op
(here: one setup-phase op, before any measured window) has no matching device row. That leaves
the host logs (``tracy_ops_times.csv``, ``tracy_ops_data.csv``) and the device log
(``profile_log_device.csv``) intact under ``$TT_METAL_HOME/generated/profiler/.logs``. This script
processes those same logs again with the offending host ops filtered out and prints what it
dropped, so the ops CSV (with OP CODE / DEVICE ID columns) can still be produced for
``tt-perf-report``. Nothing under tt-metal is modified.

    python scripts/tracy_postprocess_tolerant.py [name_append]
"""
import sys

import tracy.process_ops_logs as pol
from loguru import logger

_orig = pol._enrich_ops_from_perf_csv


def _tolerant(host_ops_by_device, device_perf_by_device, trace_replays):
    dropped = []
    for device_id, ops in list(host_ops_by_device.items()):
        present = {op_id for (op_id, _trace_id, _session) in device_perf_by_device.get(device_id, {})}
        kept = []
        for op in ops:
            if int(op["global_call_count"]) in present:
                kept.append(op)
            else:
                dropped.append((device_id, op.get("op_code"), op["global_call_count"]))
        host_ops_by_device[device_id] = kept
    logger.warning(f"tolerant post-processing: dropped {len(dropped)} host op(s) without device data: {dropped[:20]}")
    return _orig(host_ops_by_device, device_perf_by_device, trace_replays)


pol._enrich_ops_from_perf_csv = _tolerant

if __name__ == "__main__":
    name_append = sys.argv[1] if len(sys.argv) > 1 else ""
    pol.process_ops(None, name_append, True, device_only=False)
