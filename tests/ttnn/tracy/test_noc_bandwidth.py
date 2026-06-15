# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Phase 3a of the Nsight-counters plan: NoC bytes -> per-op bandwidth %.

Bandwidth is an across-cores aggregate, so NoC bytes are summed over all of an
op's cores (never sampled) and divided by the op's duration and the part's peak
bandwidth. These tests pin the aggregation and the BW% math, including the
analytical check the Phase 3 gate requires (measured bytes match expectation).
"""

import pytest

import glob

from tracy.noc_bandwidth import (
    aggregate_noc_bytes_per_op,
    noc_bw_util_pct,
    noc_bytes_from_trace_dir,
    per_op_noc_bw_pct,
)


def _ev(run_host_id, noc, num_bytes, sx=0, sy=0, etype="WRITE"):
    return {
        "run_host_id": run_host_id,
        "noc": noc,
        "num_bytes": num_bytes,
        "sx": sx,
        "sy": sy,
        "type": etype,
    }


def test_bytes_summed_across_cores_and_nocs():
    events = [
        _ev(1, "NOC_0", 1024, sx=0),
        _ev(1, "NOC_0", 1024, sx=1),  # different core, same op -> adds
        _ev(1, "NOC_1", 512, sx=0),
        _ev(2, "NOC_0", 4096, sx=0),  # different op
    ]
    agg = aggregate_noc_bytes_per_op(events)
    assert agg[1]["NOC_0"] == 2048
    assert agg[1]["NOC_1"] == 512
    assert agg[1]["total"] == 2560
    assert agg[2]["total"] == 4096


def test_bw_util_pct_against_peak():
    # 512 GB in 1 second at a 512 GB/s peak = 100%.
    assert noc_bw_util_pct(512e9, duration_ns=1e9, peak_gbps=512.0) == pytest.approx(100.0)
    # half the bytes -> half the utilization.
    assert noc_bw_util_pct(256e9, duration_ns=1e9, peak_gbps=512.0) == pytest.approx(50.0)


def test_bw_util_zero_duration_is_nan_not_crash():
    import math

    assert math.isnan(noc_bw_util_pct(1024, duration_ns=0, peak_gbps=512.0))


def test_per_op_bw_joins_durations():
    events = [_ev(1, "NOC_0", int(402e6)), _ev(2, "NOC_0", int(100e6))]
    durations_ns = {1: 785000, 2: 1_000_000}  # op 1 ~ the eltwise's 402MB/785us
    bw = per_op_noc_bw_pct(events, durations_ns, peak_gbps=512.0)
    # 402e6 bytes / 785us = ~512 GB/s -> ~100% of a 512 GB/s peak.
    assert bw[1] == pytest.approx(100.0, abs=2.0)
    assert bw[2] < bw[1]


def test_trace_dir_aggregates_real_capture():
    """If a real noc-trace capture is on disk, its summed bytes must match the
    analytical eltwise traffic (the Phase 3 gate, end-to-end through the dir
    reader)."""
    import os

    dirs = glob.glob("generated/profiler/*/.logs")
    cap = next((d for d in dirs if glob.glob(os.path.join(d, "noc_trace*BinaryNg*ID*.json"))), None)
    if cap is None:
        pytest.skip("no binary-op noc-trace capture on disk")
    agg = noc_bytes_from_trace_dir(cap)
    expected = 3 * 8192 * 8192 * 2  # 8192x8192 bf16 add: 2 reads + 1 write
    warm = [v["total"] for v in agg.values() if abs(v["total"] - expected) / expected < 0.02]
    assert warm, f"no op matched analytical {expected} bytes; saw {[v['total'] for v in agg.values()]}"


def test_eltwise_known_bytes_match_analytical():
    """The Phase 3 gate: a DRAM-resident eltwise add of an 8192x8192 bf16
    tensor moves 2 reads + 1 write = 402 MB; aggregated NoC bytes must match."""
    side, dbytes = 8192, 2
    expected = 3 * side * side * dbytes
    # Synthesize the transfers a DRAM-bound add would emit across 120 cores.
    per_core = expected // 120
    events = [_ev(7, "NOC_0", per_core, sx=c) for c in range(120)]
    events.append(_ev(7, "NOC_0", expected - per_core * 120))  # remainder
    agg = aggregate_noc_bytes_per_op(events)
    assert agg[7]["total"] == expected
