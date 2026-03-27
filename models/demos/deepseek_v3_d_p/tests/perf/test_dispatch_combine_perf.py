# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Performance test for dispatch and combine operations.

Measures host-side wall-clock time for dispatch and combine operations
across Linear vs Ring topologies to quantify the performance delta.

Usage:
    # Run on 8-chip linear:
    pytest models/demos/deepseek_v3_d_p/tests/perf/test_dispatch_combine_perf.py -k "linear-8" -s

    # Run on 8-chip ring:
    pytest models/demos/deepseek_v3_d_p/tests/perf/test_dispatch_combine_perf.py -k "ring-8" -s

    # Run all configurations:
    pytest models/demos/deepseek_v3_d_p/tests/perf/test_dispatch_combine_perf.py -s

    # With device-side DPRINT cycle counters (requires ENABLE_PERF_TRACE=1 in kernel .cpp files):
    TT_METAL_DPRINT_CORES="0,0" TT_METAL_DPRINT_CHIPS=0 \
    pytest models/demos/deepseek_v3_d_p/tests/perf/test_dispatch_combine_perf.py -s

    DPRINT output is auto-redirected to a temp file for parsing.  The test
    prints a per-kernel cycle breakdown after each config and a device-cycle
    comparison table in the final summary.
"""

import os
import re
import tempfile
import time
from collections import defaultdict

# Auto-redirect DPRINT to a parseable file when DPRINT is enabled.
# Must happen before ttnn import so the DPRINT server picks up the path.
_DPRINT_FILE: str | None = os.environ.get("TT_METAL_DPRINT_FILE")
if _DPRINT_FILE is None and os.environ.get("TT_METAL_DPRINT_CORES"):
    _DPRINT_FILE = os.path.join(tempfile.gettempdir(), f"dprint_perf_{os.getpid()}.log")
    os.environ["TT_METAL_DPRINT_FILE"] = _DPRINT_FILE

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    extract_mesh_config,
    get_dispatch_input_mesh_mapper,
    get_ep_mesh_mapper,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule

WARMUP_ITERS = 2
MEASURE_ITERS = 5

SEQ_LEN_PER_CHIP = 3200
EMB_DIM = 7168
NUM_ROUTED_EXPERTS = 64
NUM_EXPERTS_PER_TOK = 2
CAPACITY_FACTOR = 2

_collected_results: list[dict] = []

# ---------------------------------------------------------------------------
# DPRINT parsing helpers
# ---------------------------------------------------------------------------

_PERF_RE = re.compile(r"PERF\s+(\w+)\s+(.*?)$", re.MULTILINE)
_KV_RE = re.compile(r"(\w+)=(\d+)")
_COUNTER_FIELDS = frozenset({"chip", "core", "total", "n_local", "n_remote", "n_sends"})


def _dprint_file_pos() -> int:
    if not _DPRINT_FILE or not os.path.exists(_DPRINT_FILE):
        return 0
    return os.path.getsize(_DPRINT_FILE)


def _read_perf_entries(start_pos: int) -> list[tuple[str, dict[str, int]]]:
    """Read PERF lines from the DPRINT file starting at *start_pos*."""
    if not _DPRINT_FILE or not os.path.exists(_DPRINT_FILE):
        return []
    with open(_DPRINT_FILE) as f:
        f.seek(start_pos)
        text = f.read()
    entries = []
    for m in _PERF_RE.finditer(text):
        kernel = m.group(1)
        fields = {k: int(v) for k, v in _KV_RE.findall(m.group(2))}
        entries.append((kernel, fields))
    return entries


def _median(values: list[int]) -> int:
    s = sorted(values)
    return s[len(s) // 2]


def _summarize_perf(entries: list[tuple[str, dict[str, int]]]) -> dict[str, dict]:
    """Group PERF entries by kernel and compute median cycles + breakdown."""
    by_kernel: dict[str, list[dict[str, int]]] = defaultdict(list)
    for kernel, fields in entries:
        by_kernel[kernel].append(fields)

    results = {}
    for kernel in sorted(by_kernel):
        samples = by_kernel[kernel]
        total_med = _median([s.get("total", 0) for s in samples])

        all_fields = set()
        for s in samples:
            all_fields.update(s.keys())

        cycle_fields = {}
        counter_fields = {}
        for field in sorted(all_fields):
            vals = [s.get(field, 0) for s in samples]
            med = _median(vals)
            if field in _COUNTER_FIELDS:
                counter_fields[field] = med
            else:
                cycle_fields[field] = med

        results[kernel] = {
            "total": total_med,
            "cycles": cycle_fields,
            "counters": counter_fields,
            "n_samples": len(samples),
        }
    return results


def _format_kernel_perf(perf: dict[str, dict]) -> str:
    """Format per-kernel perf breakdown for a single test config."""
    if not perf:
        return ""
    lines = [
        f"  {'Kernel':<20} {'Total':>10} {'N':>4}  Breakdown (% of total)",
        f"  {'-'*20} {'-'*10} {'-'*4}  {'-'*55}",
    ]
    for kernel in sorted(perf):
        r = perf[kernel]
        total = r["total"]
        n = r["n_samples"]
        breakdown = sorted(r["cycles"].items(), key=lambda kv: -kv[1])
        parts = []
        for field, val in breakdown:
            pct = val * 100.0 / total if total else 0
            parts.append(f"{field}={pct:.0f}%")
        lines.append(f"  {kernel:<20} {total:>10} {n:>4}  {' '.join(parts)}")
        counters = {k: v for k, v in r["counters"].items() if k not in ("chip", "core")}
        if counters:
            cstr = " ".join(f"{k}={v}" for k, v in counters.items())
            lines.append(f"  {'':<20} {'':<10} {'':<4}  ({cstr})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def print_summary():
    """Print a comparison summary table after all parametrized tests in this module complete."""
    _collected_results.clear()
    if _DPRINT_FILE:
        logger.info(f"DPRINT output file: {_DPRINT_FILE}")
    yield
    if not _collected_results:
        return

    # --- Wall-clock summary ---
    header = (
        f"\n{'='*90}\n"
        f"  COMPARISON SUMMARY  (median us, {MEASURE_ITERS} iters after {WARMUP_ITERS} warmup)\n"
        f"  seq_len={SEQ_LEN_PER_CHIP}, emb_dim={EMB_DIM}, experts={NUM_ROUTED_EXPERTS}, top-k={NUM_EXPERTS_PER_TOK}\n"
        f"{'='*90}\n"
        f"  {'Config':<22} {'Dispatch':>12} {'Combine':>12} {'Round-trip':>12}\n"
        f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12}"
    )
    rows = []
    for r in _collected_results:
        rows.append(
            f"  {r['config']:<22} {r['dispatch_med']:>12.1f} {r['combine_med']:>12.1f} {r['roundtrip_med']:>12.1f}"
        )

    speedup_lines = []
    lookup: dict[tuple, dict] = {}
    for r in _collected_results:
        lookup[(r["num_devices"], r["topo"], r["num_links"])] = r

    def fmt_speedup(base, fast):
        d_sp = base["dispatch_med"] / fast["dispatch_med"] if fast["dispatch_med"] else 0
        c_sp = base["combine_med"] / fast["combine_med"] if fast["combine_med"] else 0
        r_sp = base["roundtrip_med"] / fast["roundtrip_med"] if fast["roundtrip_med"] else 0
        return f"{d_sp:>12.2f}x {c_sp:>12.2f}x {r_sp:>12.2f}x"

    seen_devices = sorted({r["num_devices"] for r in _collected_results})
    for ndev in seen_devices:
        for nlinks in (1, 2):
            lin = lookup.get((ndev, "Linear", nlinks))
            ring = lookup.get((ndev, "Ring", nlinks))
            if lin and ring:
                label = f"Ring/Lin {ndev}dev {nlinks}L"
                speedup_lines.append(f"  {label:<22} {fmt_speedup(lin, ring)}")
        for topo in ("Linear", "Ring"):
            one = lookup.get((ndev, topo, 1))
            two = lookup.get((ndev, topo, 2))
            if one and two:
                label = f"2L/1L {topo[:3]} {ndev}dev"
                speedup_lines.append(f"  {label:<22} {fmt_speedup(one, two)}")

    summary = "\n".join([header] + rows)
    if speedup_lines:
        summary += (
            f"\n\n  {'Speedup':<22} {'Dispatch':>12} {'Combine':>12} {'Round-trip':>12}\n"
            f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12}\n" + "\n".join(speedup_lines)
        )
    summary += f"\n{'='*90}"
    logger.info(summary)

    # --- Device cycle summary (if DPRINT data was collected) ---
    has_perf = any(r.get("device_perf") for r in _collected_results)
    if not has_perf:
        return

    kernels = sorted({k for r in _collected_results if r.get("device_perf") for k in r["device_perf"]})

    for kernel in kernels:
        # Collect all cycle-breakdown fields across configs for this kernel
        all_fields: list[str] = []
        seen: set[str] = set()
        for r in _collected_results:
            kp = r.get("device_perf", {}).get(kernel)
            if not kp:
                continue
            for field, _ in sorted(kp["cycles"].items(), key=lambda kv: -kv[1]):
                if field not in seen:
                    all_fields.append(field)
                    seen.add(field)

        col_w = 8
        hdr_config = f"  {'Config':<22} {'total':>10}"
        hdr_fields = "".join(f" {f:>{col_w}}" for f in all_fields)
        sep_config = f"  {'-'*22} {'-'*10}"
        sep_fields = (f" {'-'*col_w}") * len(all_fields)

        lines = [
            f"\n{'='*(35 + (col_w + 1) * len(all_fields))}",
            f"  DEVICE CYCLES: {kernel}  (median, % of total)",
            f"{'='*(35 + (col_w + 1) * len(all_fields))}",
            hdr_config + hdr_fields,
            sep_config + sep_fields,
        ]
        for r in _collected_results:
            kp = r.get("device_perf", {}).get(kernel)
            if not kp:
                row = f"  {r['config']:<22} {'N/A':>10}" + (f" {'':>{col_w}}") * len(all_fields)
            else:
                total = kp["total"]
                row = f"  {r['config']:<22} {total:>10}"
                for f in all_fields:
                    val = kp["cycles"].get(f, 0)
                    pct = val * 100.0 / total if total else 0
                    row += f" {pct:>{col_w - 1}.0f}%"
            lines.append(row)

        # Add counter row if available
        counter_keys: list[str] = []
        counter_seen: set[str] = set()
        for r in _collected_results:
            kp = r.get("device_perf", {}).get(kernel)
            if kp:
                for ck in kp["counters"]:
                    if ck not in ("chip", "core", "total") and ck not in counter_seen:
                        counter_keys.append(ck)
                        counter_seen.add(ck)
        if counter_keys:
            lines.append("")
            lines.append(f"  {'Config':<22} " + "".join(f" {ck:>{col_w}}" for ck in counter_keys))
            lines.append(f"  {'-'*22} " + (f" {'-'*col_w}") * len(counter_keys))
            for r in _collected_results:
                kp = r.get("device_perf", {}).get(kernel)
                row = f"  {r['config']:<22} "
                if kp:
                    for ck in counter_keys:
                        row += f" {kp['counters'].get(ck, 0):>{col_w}}"
                lines.append(row)

        lines.append(f"{'='*(35 + (col_w + 1) * len(all_fields))}")
        logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="linear-4-1link",
        ),
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="linear-4-2link",
        ),
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Ring,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="ring"),
            id="ring-4-1link",
        ),
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Ring,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="ring"),
            id="ring-4-2link",
        ),
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="linear-8-1link",
        ),
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="linear-8-2link",
        ),
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Ring,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="ring"),
            id="ring-8-1link",
        ),
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Ring,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="ring"),
            id="ring-8-2link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_dispatch_combine_perf(
    mesh_device,
    num_links,
    topology,
):
    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        SEQ_LEN_PER_CHIP, NUM_ROUTED_EXPERTS, NUM_EXPERTS_PER_TOK, num_devices, dispatch_group_size, CAPACITY_FACTOR
    )

    topo_name = "Ring" if topology == ttnn.Topology.Ring else "Linear"
    logger.info(
        f"\n{'='*70}\n"
        f"  PERF TEST: {topo_name} topology, {num_devices} devices, {num_links} link(s)\n"
        f"  seq_len={SEQ_LEN_PER_CHIP}, emb_dim={EMB_DIM}, experts={NUM_ROUTED_EXPERTS}, top-k={NUM_EXPERTS_PER_TOK}\n"
        f"  experts_per_chip={experts_per_chip}, max_tok_per_expert={max_dispatched_tokens_per_expert}\n"
        f"  dispatch_group_size={dispatch_group_size}, num_dispatch_groups={num_dispatch_groups}\n"
        f"{'='*70}"
    )

    x, weights, indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=SEQ_LEN_PER_CHIP,
        emb_dim=EMB_DIM,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seed=42,
        num_dispatch_groups=num_dispatch_groups,
    )

    mesh_mapper_dispatch_inputs = get_dispatch_input_mesh_mapper(mesh_device, sp_axis)

    tt_x = ttnn.from_torch(
        x,
        mesh_mapper=mesh_mapper_dispatch_inputs,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_weights = ttnn.from_torch(
        weights,
        mesh_mapper=mesh_mapper_dispatch_inputs,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_indices = ttnn.from_torch(
        indices,
        mesh_mapper=mesh_mapper_dispatch_inputs,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    expert_offsets, expert_token_counts, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        NUM_ROUTED_EXPERTS,
        experts_per_chip,
        SEQ_LEN_PER_CHIP,
        NUM_EXPERTS_PER_TOK,
    )

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=NUM_ROUTED_EXPERTS,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    tt_expert_offsets = TtDispatchModule.shard_expert_offsets(mesh_device, expert_offsets)
    tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)

    mesh_mapper_combine_counter = get_ep_mesh_mapper(mesh_device)
    tt_expert_token_counts = ttnn.from_torch(
        expert_token_counts,
        mesh_mapper=mesh_mapper_combine_counter,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    tt_dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=SEQ_LEN_PER_CHIP,
        emb_dim=EMB_DIM,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
    )

    tt_combine_module = TtCombineModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        seq_len_per_chip=SEQ_LEN_PER_CHIP,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        init_zeros=True,
    )

    # Warmup
    for i in range(WARMUP_ITERS):
        tt_dispatched_buffer, tt_metadata = tt_dispatch_module(
            tt_x, tt_weights, tt_indices, tt_expert_offsets, tt_expert_dispatch_table
        )
        tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_expert_token_counts)
        ttnn.synchronize_device(mesh_device)

    # Record DPRINT file position after warmup so we only parse measurement entries
    dprint_pos = _dprint_file_pos()

    # Measure dispatch
    dispatch_times = []
    for i in range(MEASURE_ITERS):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        tt_dispatched_buffer, tt_metadata = tt_dispatch_module(
            tt_x, tt_weights, tt_indices, tt_expert_offsets, tt_expert_dispatch_table
        )
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()
        dispatch_times.append((t1 - t0) * 1e6)

    # Measure combine
    combine_times = []
    for i in range(MEASURE_ITERS):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_expert_token_counts)
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()
        combine_times.append((t1 - t0) * 1e6)

    # Measure dispatch+combine together
    roundtrip_times = []
    for i in range(MEASURE_ITERS):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        tt_dispatched_buffer, tt_metadata = tt_dispatch_module(
            tt_x, tt_weights, tt_indices, tt_expert_offsets, tt_expert_dispatch_table
        )
        tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_expert_token_counts)
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()
        roundtrip_times.append((t1 - t0) * 1e6)

    def stats(times):
        times_sorted = sorted(times)
        median = times_sorted[len(times_sorted) // 2]
        return min(times), median, max(times), sum(times) / len(times)

    d_min, d_med, d_max, d_avg = stats(dispatch_times)
    c_min, c_med, c_max, c_avg = stats(combine_times)
    r_min, r_med, r_max, r_avg = stats(roundtrip_times)

    config_label = f"{topo_name}-{num_devices} ({num_links}L)"
    logger.info(
        f"\n{'='*70}\n"
        f"  RESULTS: {topo_name} | {num_devices} devices | {num_links} link(s)\n"
        f"  Iterations: {MEASURE_ITERS} (after {WARMUP_ITERS} warmup)\n"
        f"{'='*70}\n"
        f"  {'Operation':<20} {'Min (us)':>12} {'Median (us)':>12} {'Avg (us)':>12} {'Max (us)':>12}\n"
        f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}\n"
        f"  {'Dispatch':<20} {d_min:>12.1f} {d_med:>12.1f} {d_avg:>12.1f} {d_max:>12.1f}\n"
        f"  {'Combine':<20} {c_min:>12.1f} {c_med:>12.1f} {c_avg:>12.1f} {c_max:>12.1f}\n"
        f"  {'Dispatch+Combine':<20} {r_min:>12.1f} {r_med:>12.1f} {r_avg:>12.1f} {r_max:>12.1f}\n"
        f"{'='*70}"
    )

    # Parse DPRINT PERF entries collected during measurements
    perf_entries = _read_perf_entries(dprint_pos)
    device_perf = _summarize_perf(perf_entries) if perf_entries else {}

    if device_perf:
        logger.info(f"\n  DEVICE CYCLES: {config_label}\n" + _format_kernel_perf(device_perf))

    _collected_results.append(
        {
            "config": config_label,
            "topo": topo_name,
            "num_devices": num_devices,
            "num_links": num_links,
            "dispatch_med": d_med,
            "combine_med": c_med,
            "roundtrip_med": r_med,
            "device_perf": device_perf,
        }
    )
