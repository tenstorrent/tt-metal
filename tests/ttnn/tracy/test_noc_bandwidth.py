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
    aggregate_fabric_bytes_per_op,
    aggregate_noc_bytes_per_op,
    aggregate_noc_links_per_op,
    all_gather_bytes_per_link,
    all_gather_fabric_bw,
    eth_bw_util_pct,
    noc_bw_util_pct,
    noc_bytes_from_trace_dir,
    noc_port_peak_gbps,
    per_op_noc_bw_pct,
    read_trained_link_bw_gbps,
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


def test_read_trained_link_bw_takes_min_across_devices(tmp_path):
    """The captured per-link BW is read from the device sidecars; a mixed-speed
    mesh must report its slowest link so util% cannot exceed 100%."""
    import json
    import math

    for dev, bw in [(0, 50.0), (1, 50.0), (2, 25.0)]:
        (tmp_path / f"fabric_link_bw_{dev}.json").write_text(json.dumps({"device_id": dev, "per_link_gb_s": bw}))
    assert read_trained_link_bw_gbps(tmp_path) == 25.0
    # No sidecar -> None (caller falls back / prints nothing, never a fabricated peak).
    assert read_trained_link_bw_gbps(tmp_path / "empty") is None


def _fab(run_host_id, num_bytes, proc="NCRISC", dx=None, dy=None, sx=0, sy=0):
    ev = {"run_host_id": run_host_id, "num_bytes": num_bytes, "proc": proc, "sx": sx, "sy": sy, "fabric_send": {}}
    if dx is not None:
        ev["dx"], ev["dy"] = dx, dy
    return ev


def test_fabric_bytes_and_link_count():
    # 3 worker fabric sends to 2 distinct ingress eth cores -> 2 links, summed bytes.
    events = [
        _fab(1, 1000, dx=1, dy=9),
        _fab(1, 1000, dx=1, dy=9),
        _fab(1, 2000, dx=2, dy=9),
        _ev(1, "NOC_0", 999),  # local (no fabric_send) -> excluded
    ]
    agg = aggregate_fabric_bytes_per_op(events)
    assert agg[1]["bytes"] == 4000
    assert agg[1]["links"] == 2
    # ERISC events, when profiled, give the link count directly (distinct eth cores).
    erisc = [_fab(2, 500, proc="ERISC", sx=18, sy=0), _fab(2, 500, proc="ERISC", sx=19, sy=0)]
    agg2 = aggregate_fabric_bytes_per_op(erisc)
    assert agg2[2]["links"] == 2


def test_fabric_link_count_prefers_real_eth_chan():
    """eth_chan is the channel the profiler actually resolved; it must win over the
    dst-coord heuristic (many workers can funnel through one eth link, or one dst
    can fan out across links)."""
    events = [
        # 4 workers, 3 dst coords, but only 2 real eth channels -> links must be 2.
        {"run_host_id": 5, "num_bytes": 100, "dx": 1, "dy": 9, "fabric_send": {"eth_chan": 0}},
        {"run_host_id": 5, "num_bytes": 100, "dx": 2, "dy": 9, "fabric_send": {"eth_chan": 0}},
        {"run_host_id": 5, "num_bytes": 100, "dx": 3, "dy": 9, "fabric_send": {"eth_chan": 1}},
        {"run_host_id": 5, "num_bytes": 100, "dx": 3, "dy": 9, "fabric_send": {"eth_chan": 1}},
    ]
    agg = aggregate_fabric_bytes_per_op(events)
    assert agg[5]["bytes"] == 400
    assert agg[5]["links"] == 2


def test_noc_links_and_aiclk_derived_peak():
    """NoC BW% divisor must be (#active ports x per-port peak) with the per-port peak
    coming from the REAL measured AICLK, not a hardcoded GB/s."""
    events = [
        _ev(1, "NOC_0", 1000, sx=0),
        _ev(1, "NOC_1", 1000, sx=0),  # same core, other NoC -> distinct port
        _ev(1, "NOC_0", 1000, sx=1),  # other core -> distinct port
    ]
    agg = aggregate_noc_links_per_op(events)
    assert agg[1]["bytes"] == 3000
    assert agg[1]["links"] == 3
    # 32 B/cycle at 1350 MHz -> 43.2 GB/s per port; unknown clock -> None (never fabricated).
    assert noc_port_peak_gbps(1350.0) == pytest.approx(43.2)
    assert noc_port_peak_gbps(0) is None
    assert noc_port_peak_gbps(None) is None


def test_eth_bw_util_reproduces_allgather_measurement():
    """Pins the hand-validated stage-2 AllGather number: 39.71 MB injected over 2
    links in ~0.924 ms against a real 50 GB/s/link peak == ~43% fabric util."""
    util = eth_bw_util_pct(int(39.71e6), duration_ns=924_000, per_link_gbps=50.0, num_links=2)
    assert util == pytest.approx(43.0, abs=1.0)
    # A real peak that doubled (800G link) halves the util for the same traffic.
    assert eth_bw_util_pct(int(39.71e6), 924_000, per_link_gbps=100.0, num_links=2) == pytest.approx(21.5, abs=0.5)


def test_all_gather_bytes_per_link_canonical_formula():
    """Per-link fabric bytes match the algorithm-BW definition used by the perf sweeps:
    output_bytes x (N-1)/N / links / (2 if ring). Stage-2 gather: (1,1,9696,4096) bf16 over 2
    devices, 2 links, Linear -> 19.86 MB/link (half the 39.71 MB total across the two links)."""
    output_bytes = 9696 * 4096 * 2  # (1,1,9696,4096) bf16 = 79.43 MB gathered output
    per_link = all_gather_bytes_per_link(output_bytes, num_devices=2, num_links=2, is_ring=False)
    assert per_link == pytest.approx(19_857_408)
    # A ring moves the same payload in both directions -> half the per-link bytes.
    assert all_gather_bytes_per_link(output_bytes, 2, 2, is_ring=True) == pytest.approx(per_link / 2)
    # Degenerate / missing inputs never fabricate traffic.
    assert all_gather_bytes_per_link(output_bytes, num_devices=1, num_links=2, is_ring=False) == 0.0
    assert all_gather_bytes_per_link(output_bytes, num_devices=2, num_links=0, is_ring=False) == 0.0


def test_all_gather_fabric_bw_reproduces_43pct_and_scales_with_real_peak():
    """The analytical all-gather BW (no noc-traces) reproduces the hand-validated 43% at the real
    50 GB/s peak, and %util tracks the *trained* peak so the number stays honest across machines."""
    output_bytes = 9696 * 4096 * 2
    gbps, pct = all_gather_fabric_bw(output_bytes, 2, 2, False, duration_ns=924_000, per_link_peak_gbps=50.0)
    assert gbps == pytest.approx(21.49, abs=0.1)  # achieved per-link
    assert pct == pytest.approx(43.0, abs=1.0)
    # 800G link (100 GB/s) -> same traffic is half the util.
    _, pct_800 = all_gather_fabric_bw(output_bytes, 2, 2, False, 924_000, 100.0)
    assert pct_800 == pytest.approx(21.5, abs=0.5)
    # No trained peak on disk -> GB/s still reported, %util blank (never fabricated).
    gbps_np, pct_np = all_gather_fabric_bw(output_bytes, 2, 2, False, 924_000, None)
    assert gbps_np == pytest.approx(21.49, abs=0.1)
    assert pct_np is None
    # Missing duration -> nothing at all.
    assert all_gather_fabric_bw(output_bytes, 2, 2, False, 0, 50.0) == (None, None)
