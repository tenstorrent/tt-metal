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


def noc_bytes_from_trace_dir(log_folder, op_key="run_host_id"):
    """Aggregate per-op NoC bytes from every noc_trace*.json in a .logs dir.

    Independent of tt-npe: reads the profiler's own noc-trace JSON (flat lists
    of events + zone endpoints) and sums num_bytes per op. Returns the same
    shape as aggregate_noc_bytes_per_op.
    """
    import glob
    import json
    import os

    events = []
    for path in sorted(glob.glob(os.path.join(str(log_folder), "noc_trace*.json"))):
        with open(path) as f:
            events.extend(json.load(f))
    return aggregate_noc_bytes_per_op(events, op_key=op_key)


def read_trained_link_bw_gbps(log_folder):
    """Real per-link fabric bandwidth (GB/s) captured on device, or None if absent.

    The device profiler drops one ``fabric_link_bw_<device_id>.json`` per device carrying the
    per-link bandwidth it read from the eth-FW ``train_speed`` telemetry (falling back to the
    arch nominal when the arch does not expose train_speed or no link is up). We take the MIN
    across devices so a mixed-speed mesh is reported at its bottleneck and util% never exceeds
    100%. This is what makes BW-util stats honest across machines (200/400/800G) instead of a
    hardcoded per-arch peak.
    """
    import glob
    import json
    import os

    speeds = []
    for path in sorted(glob.glob(os.path.join(str(log_folder), "fabric_link_bw_*.json"))):
        try:
            with open(path) as f:
                info = json.load(f)
        except (OSError, ValueError):
            continue
        v = info.get("per_link_gb_s")
        if v and v > 0:
            speeds.append(float(v))
    return min(speeds) if speeds else None


def is_fabric_event(ev):
    """A noc-trace event that crossed the ethernet fabric carries a 'fabric_send' block."""
    return isinstance(ev, dict) and "fabric_send" in ev


def aggregate_fabric_bytes_per_op(events, op_key="run_host_id"):
    """Per-op ethernet-fabric bytes and the distinct fabric links each op drove.

    Fabric bytes are the payload injected into the fabric by an op's worker cores (events with a
    'fabric_send' block). ``links`` counts the distinct eth links carrying the op's traffic, in
    order of fidelity: (1) the real ``fabric_send.eth_chan`` the profiler resolved (the eth channel
    actually used on the src device); (2) distinct ERISC cores when the eth cores are themselves
    profiled; (3) the distinct fabric-ingress destinations (dx, dy) of the worker sends. The link
    count is the divisor for ETH BW-util %, so getting it from eth_chan (not a heuristic) matters.
    Returns {op_id: {"bytes": int, "links": int}}.
    """
    by_op_bytes = {}
    by_op_chan = {}
    by_op_erisc = {}
    by_op_dst = {}
    for ev in events:
        if not is_fabric_event(ev):
            continue
        nbytes = ev.get("num_bytes")
        if nbytes is None:
            continue
        op_id = ev[op_key]
        by_op_bytes[op_id] = by_op_bytes.get(op_id, 0) + nbytes
        eth_chan = (ev.get("fabric_send") or {}).get("eth_chan")
        if eth_chan is not None:
            by_op_chan.setdefault(op_id, set()).add((ev.get("src_device_id"), eth_chan))
        if str(ev.get("proc", "")).upper().startswith("ERISC"):
            by_op_erisc.setdefault(op_id, set()).add((ev.get("src_device_id"), ev.get("sx"), ev.get("sy")))
        if ev.get("dx") is not None and ev.get("dy") is not None:
            by_op_dst.setdefault(op_id, set()).add((ev.get("src_device_id"), ev.get("dx"), ev.get("dy")))

    out = {}
    for op_id, nbytes in by_op_bytes.items():
        links = len(by_op_chan.get(op_id, ())) or len(by_op_erisc.get(op_id, ())) or len(by_op_dst.get(op_id, ())) or 1
        out[op_id] = {"bytes": nbytes, "links": links}
    return out


def aggregate_noc_links_per_op(events, op_key="run_host_id"):
    """Per-op total NoC bytes + the distinct (core, NoC) injection ports the op drove.

    A per-link NoC BW-util % needs a physically meaningful divisor: each Tensix core injects on
    NOC_0/NOC_1 independently, and each port's peak is AICLK x datapath width. Counting distinct
    (src_device, sx, sy, noc) ports lets util% be measured against (#ports x per-port peak) rather
    than an ill-defined aggregate. Returns {op_id: {"bytes": int, "links": int}}.
    """
    by_op_bytes = {}
    by_op_ports = {}
    for ev in events:
        nbytes = ev.get("num_bytes")
        if nbytes is None:
            continue
        op_id = ev[op_key]
        by_op_bytes[op_id] = by_op_bytes.get(op_id, 0) + nbytes
        by_op_ports.setdefault(op_id, set()).add(
            (ev.get("src_device_id"), ev.get("sx"), ev.get("sy"), ev.get("noc", "NOC_0"))
        )
    return {op_id: {"bytes": b, "links": len(by_op_ports.get(op_id, ())) or 1} for op_id, b in by_op_bytes.items()}


# NoC datapath is 32 B/cycle (256-bit) per port on both Wormhole and Blackhole; the per-port peak
# scales with the real AICLK, so deriving it from the measured clock (never a hardcoded GB/s) keeps
# NoC BW-util % honest across parts and clock states.
_NOC_BYTES_PER_CYCLE = 32.0


def noc_port_peak_gbps(aiclk_mhz, bytes_per_cycle=_NOC_BYTES_PER_CYCLE):
    """Per-port NoC injection peak (GB/s) from the real AICLK. None if the clock is unknown."""
    if not aiclk_mhz or aiclk_mhz <= 0:
        return None
    return aiclk_mhz * 1e6 * bytes_per_cycle / 1e9


def eth_bw_util_pct(fabric_bytes, duration_ns, per_link_gbps, num_links):
    """ETH BW util % = achieved fabric throughput / (per-link peak x links) x 100.

    per_link_gbps is really GB/s (achieved-peak of one link); num_links is how many links the op
    drove. NaN when any input is missing so a lie is never printed as a number.
    """
    if not duration_ns or duration_ns <= 0 or not per_link_gbps or not num_links:
        return nan
    achieved_gbps = fabric_bytes / (duration_ns * 1e-9) / 1e9
    return achieved_gbps / (per_link_gbps * num_links) * 100.0


# Algorithm-BW "wire factor" per collective: the multiple of the FULL tensor each rank moves across
# the fabric. all-gather and reduce-scatter are duals that each move (N-1)/N; all-reduce = the two
# back to back, so 2·(N-1)/N. (Canonical NCCL/tt-metal busbw factors — the same the all-gather sweep
# perf tests use.) 'output_bytes' means the op's OUTPUT tensor: for all-gather / all-reduce that is
# the full tensor, but reduce-scatter's output is the 1/N scattered chunk, so its full tensor is
# output_bytes·N (handled in collective_bytes_per_link).
_COLLECTIVE_ALG_FACTOR = {"all_gather": 1.0, "reduce_scatter": 1.0, "all_reduce": 2.0}


def collective_bytes_per_link(output_bytes, num_devices, num_links, is_ring, kind="all_gather"):
    """Per-link ethernet bytes a collective pushes across the fabric (exact from shape + topology).

    Exact from shape + topology alone — the point being that a full-grid collective emits ~11.6k
    markers/core and overflows the noc-trace DRAM buffer, so its fabric bytes can NOT be read back
    from a noc-trace. The traffic is spread across ``num_links`` and both directions of a ring (the
    /2 ring divisor, applied to every collective so the per-link numbers are comparable). Returns 0.0
    for a degenerate 1-device collective, missing links, or an unknown kind.
    """
    if kind not in _COLLECTIVE_ALG_FACTOR:
        return 0.0
    if not num_devices or num_devices <= 1 or not num_links or num_links <= 0:
        return 0.0
    # reduce-scatter's output is the 1/N chunk; every other kind's output is the full tensor.
    full_bytes = output_bytes * num_devices if kind == "reduce_scatter" else output_bytes
    factor = _COLLECTIVE_ALG_FACTOR[kind] * (num_devices - 1) / num_devices
    ring_div = 2 if is_ring else 1
    return full_bytes * factor / num_links / ring_div


def collective_fabric_bw(
    output_bytes, num_devices, num_links, is_ring, duration_ns, per_link_peak_gbps, kind="all_gather"
):
    """(achieved per-link GB/s, % of trained peak) for a collective — analytical, no noc-traces.

    Bytes from collective_bytes_per_link (shape+topology), duration from the op's measured device-FW
    time, peak from the fabric's *trained* per-link speed so the % is honest on any machine. (None,
    None) when bytes/duration are missing; % is None (blank, never fabricated) when no trained peak.
    """
    moved = collective_bytes_per_link(output_bytes, num_devices, num_links, is_ring, kind)
    if not moved or not duration_ns or duration_ns <= 0:
        return None, None
    achieved_gbps = moved / (duration_ns * 1e-9) / 1e9
    pct = (achieved_gbps / per_link_peak_gbps * 100.0) if per_link_peak_gbps else None
    return achieved_gbps, pct


def all_gather_bytes_per_link(output_bytes, num_devices, num_links, is_ring):
    """Back-compat wrapper: per-link bytes for an all-gather (see collective_bytes_per_link)."""
    return collective_bytes_per_link(output_bytes, num_devices, num_links, is_ring, kind="all_gather")


def all_gather_fabric_bw(output_bytes, num_devices, num_links, is_ring, duration_ns, per_link_peak_gbps):
    """Back-compat wrapper: analytical all-gather fabric BW (see collective_fabric_bw)."""
    return collective_fabric_bw(
        output_bytes, num_devices, num_links, is_ring, duration_ns, per_link_peak_gbps, kind="all_gather"
    )


def _load_trace_events(log_folder):
    import glob
    import json
    import os

    events = []
    for path in sorted(glob.glob(os.path.join(str(log_folder), "noc_trace*.json"))):
        with open(path) as f:
            events.extend(json.load(f))
    return events


def fabric_bytes_from_trace_dir(log_folder, op_key="run_host_id"):
    """Per-op fabric bytes + link counts from every noc_trace*.json in a .logs dir."""
    return aggregate_fabric_bytes_per_op(_load_trace_events(log_folder), op_key=op_key)


def noc_links_from_trace_dir(log_folder, op_key="run_host_id"):
    """Per-op total NoC bytes + distinct (core, NoC) ports from every noc_trace*.json in a dir."""
    return aggregate_noc_links_per_op(_load_trace_events(log_folder), op_key=op_key)
