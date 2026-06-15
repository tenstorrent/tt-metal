# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Per-op NoC bandwidth from on-device noc_status counters / noc-trace events.

Bandwidth is an across-cores aggregate: sum num_bytes over all of an op's
NoC-active cores (never sample them -- per-core NoC load is non-uniform) and
divide by the op duration and the part's peak. This is the interim on-device
source for NoC/DRAM/ETH BW % until the noc-trace fabric capture is unblocked.

Input events follow the noc_trace schema emitted by profiler.cpp: dicts with
``run_host_id``, ``noc`` (NOC_0/NOC_1), ``num_bytes``, and ``type``. Zone
endpoints (no num_bytes) are ignored here; op durations are supplied separately
from the per-op device-time view.
"""

from math import nan


def aggregate_noc_bytes_per_op(events, op_key="run_host_id"):
    """Sum num_bytes per op, broken out per NoC plus a 'total'.

    Returns {op_id: {"NOC_0": bytes, "NOC_1": bytes, ..., "total": bytes}}.
    """
    agg = {}
    for ev in events:
        nbytes = ev.get("num_bytes")
        if nbytes is None:
            continue  # zone endpoint or non-transfer marker
        op_id = ev[op_key]
        noc = ev.get("noc", "NOC_0")
        bucket = agg.setdefault(op_id, {})
        bucket[noc] = bucket.get(noc, 0) + nbytes
        bucket["total"] = bucket.get("total", 0) + nbytes
    return agg


def noc_bw_util_pct(total_bytes, duration_ns, peak_gbps):
    """Achieved bandwidth as a percentage of peak. NaN if duration is unknown."""
    if not duration_ns or duration_ns <= 0:
        return nan
    achieved_gbps = total_bytes / (duration_ns * 1e-9) / 1e9
    return achieved_gbps / peak_gbps * 100.0


def per_op_noc_bw_pct(events, durations_ns, peak_gbps, op_key="run_host_id"):
    """Per-op NoC BW % = Σbytes over the op's cores / (duration × peak)."""
    agg = aggregate_noc_bytes_per_op(events, op_key=op_key)
    return {op_id: noc_bw_util_pct(b["total"], durations_ns.get(op_id), peak_gbps) for op_id, b in agg.items()}
