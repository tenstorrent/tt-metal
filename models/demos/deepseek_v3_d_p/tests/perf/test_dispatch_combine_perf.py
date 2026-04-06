# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Performance test for dispatch and combine operations.

Measures device-side kernel durations via the Tracy device profiler for
dispatch and combine operations across Linear vs Ring topologies.

Usage:
    # Run on 8-chip linear:
    pytest models/demos/deepseek_v3_d_p/tests/perf/test_dispatch_combine_perf.py -k "linear-8" -s

    # Run on 8-chip ring:
    pytest models/demos/deepseek_v3_d_p/tests/perf/test_dispatch_combine_perf.py -k "ring-8" -s

    # Run all configurations:
    pytest models/demos/deepseek_v3_d_p/tests/perf/test_dispatch_combine_perf.py -s

    # Run linear-8, 1-link configs (both payload sizes; 14K variant auto-skips on Wormhole):
    pytest models/demos/deepseek_v3_d_p/tests/perf/test_dispatch_combine_perf.py -k "linear-8 and 1link" -s

Device profiling requires these environment variables to be set before device open:
    TT_METAL_DEVICE_PROFILER=1          (required)
    TT_METAL_PROFILER_MID_RUN_DUMP=1    (required)
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 (required)
    TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES=1  (optional, suppresses file output)

Durations are read programmatically through ttnn.ReadDeviceProfiler().
"""

import os

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    MAX_PAYLOAD_SIZE_BH,
    MAX_PAYLOAD_SIZE_WH,
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
assert MEASURE_ITERS % 2 == 1, "MEASURE_ITERS must be odd for _median to return the exact median"

SEQ_LEN_PER_CHIP = 3200
EMB_DIM = 7168
NUM_ROUTED_EXPERTS = 64
NUM_EXPERTS_PER_TOK = 2
CAPACITY_FACTOR = 2

# Key analysis names from the device profiler (see cpp_device_analyses.json).
DEVICE_KERNEL_DURATION = "DEVICE KERNEL DURATION [ns]"

_REQUIRED_PROFILER_ENV_VARS = [
    "TT_METAL_DEVICE_PROFILER",
    "TT_METAL_PROFILER_MID_RUN_DUMP",
    "TT_METAL_PROFILER_CPP_POST_PROCESS",
]

# Populated by test_dispatch_combine_perf; consumed by the print_summary fixture.
_collected_results: list[dict] = []


def _payload_label(payload_size: int) -> str:
    return f"{payload_size // 1024}K"


# ---------------------------------------------------------------------------
# Device profiler helpers
# ---------------------------------------------------------------------------


def _check_profiler_env():
    """Raise if the required profiler environment variables are not set."""
    missing = [v for v in _REQUIRED_PROFILER_ENV_VARS if os.environ.get(v) != "1"]
    if missing:
        pytest.skip(
            f"Device profiler not configured. Set these env vars before device open: "
            f"{', '.join(f'{v}=1' for v in missing)}"
        )


def _read_device_duration(mesh_device, analysis_name: str = DEVICE_KERNEL_DURATION) -> float:
    """Read device profiler and return the op duration (us) for the given analysis.

    Calls ReadDeviceProfiler to flush profiler data, then reads the latest
    programs.  Devices execute in parallel, so the duration is the max across
    all devices (the slowest device determines the actual wall-clock time).
    Within each device, program durations are summed (sequential programs).
    """
    ttnn.ReadDeviceProfiler(mesh_device)
    latest = ttnn.get_latest_programs_perf_data()

    if not latest:
        pytest.fail("ReadDeviceProfiler returned no data -- is TT_METAL_DEVICE_PROFILER=1?")

    # Sum program durations within each device, then take the max across devices.
    max_device_dur = 0.0
    for _device_id, programs in latest.items():
        device_dur = 0.0
        for program in programs:
            analysis = program.program_analyses_results.get(analysis_name)
            if analysis is not None:
                device_dur += analysis.duration / 1000.0  # ns -> us
        max_device_dur = max(max_device_dur, device_dur)
    return max_device_dur


def _measure_op(mesh_device, op_fn, num_iters: int) -> list[float]:
    """Run *op_fn* for *num_iters* and return the device kernel duration (us) per iteration.

    Each iteration: run op -> synchronize -> ReadDeviceProfiler -> max across devices.
    """
    iter_durations: list[float] = []
    for _ in range(num_iters):
        op_fn()
        ttnn.synchronize_device(mesh_device)
        iter_durations.append(_read_device_duration(mesh_device))
    return iter_durations


def _median(values: list[float]) -> float:
    # MEASURE_ITERS is odd, so s[len(s)//2] is the exact median.
    s = sorted(values)
    return s[len(s) // 2]


def _stats(times: list[float]) -> tuple[float, float, float, float]:
    """Return (min, median, avg, max) for *times*."""
    median = _median(times)
    return min(times), median, sum(times) / len(times), max(times)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def print_summary():
    """Print a comparison summary table after all parametrized tests in this module complete."""
    _collected_results.clear()
    yield
    if not _collected_results:
        return

    header = (
        f"\n{'='*100}\n"
        f"  COMPARISON SUMMARY  (device kernel duration, median us, {MEASURE_ITERS} iters after {WARMUP_ITERS} warmup)\n"
        f"  seq_len={SEQ_LEN_PER_CHIP}, emb_dim={EMB_DIM}, experts={NUM_ROUTED_EXPERTS}, top-k={NUM_EXPERTS_PER_TOK}\n"
        f"{'='*100}\n"
        f"  {'Config':<30} {'Dispatch':>12} {'Combine':>12} {'Round-trip':>12}\n"
        f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}"
    )
    rows = []
    for r in _collected_results:
        rows.append(
            f"  {r['config']:<30} {r['dispatch_med']:>12.1f} {r['combine_med']:>12.1f} {r['roundtrip_med']:>12.1f}"
        )

    speedup_lines = []
    lookup: dict[tuple, dict] = {}
    for r in _collected_results:
        lookup[(r["num_devices"], r["topo"], r["num_links"], r["payload_size"])] = r

    def fmt_speedup(base, fast):
        d_sp = base["dispatch_med"] / fast["dispatch_med"] if fast["dispatch_med"] else 0
        c_sp = base["combine_med"] / fast["combine_med"] if fast["combine_med"] else 0
        r_sp = base["roundtrip_med"] / fast["roundtrip_med"] if fast["roundtrip_med"] else 0
        return f"{d_sp:>12.2f}x {c_sp:>12.2f}x {r_sp:>12.2f}x"

    seen_devices = sorted({r["num_devices"] for r in _collected_results})
    seen_payloads = sorted({r["payload_size"] for r in _collected_results})

    for ndev in seen_devices:
        for ps in seen_payloads:
            # Ring vs Linear comparison (same link count, same payload size)
            for nlinks in (1, 2):
                lin = lookup.get((ndev, "Linear", nlinks, ps))
                ring = lookup.get((ndev, "Ring", nlinks, ps))
                if lin and ring:
                    label = f"Ring/Lin {ndev}dev {nlinks}L {_payload_label(ps)}"
                    speedup_lines.append(f"  {label:<30} {fmt_speedup(lin, ring)}")
            # 2-link vs 1-link comparison (same topology, same payload size)
            for topo in ("Linear", "Ring"):
                one = lookup.get((ndev, topo, 1, ps))
                two = lookup.get((ndev, topo, 2, ps))
                if one and two:
                    label = f"2L/1L {topo[:3]} {ndev}dev {_payload_label(ps)}"
                    speedup_lines.append(f"  {label:<30} {fmt_speedup(one, two)}")

        # 14K vs 7K payload comparison (same topology, same link count)
        for topo in ("Linear", "Ring"):
            for nlinks in (1, 2):
                small = lookup.get((ndev, topo, nlinks, MAX_PAYLOAD_SIZE_WH))
                large = lookup.get((ndev, topo, nlinks, MAX_PAYLOAD_SIZE_BH))
                if small and large:
                    label = f"14K/7K {topo[:3]} {ndev}dev {nlinks}L"
                    speedup_lines.append(f"  {label:<30} {fmt_speedup(small, large)}")

    summary = "\n".join([header] + rows)
    if speedup_lines:
        summary += (
            f"\n\n  {'Speedup':<30} {'Dispatch':>12} {'Combine':>12} {'Round-trip':>12}\n"
            f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}\n" + "\n".join(speedup_lines)
        )
    summary += f"\n{'='*100}"
    logger.info(summary)


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------


def _make_params(mesh_shape, fabric_config, payload_size, num_links, topology, ring_or_linear):
    """Build a single pytest.param entry.

    Attaches requires_mesh_topology mark, and a skipif mark for 14K payloads
    on non-Blackhole architectures (exceeds Wormhole hardware limit).
    Test ID format: "{topology}-{ndev}-{nlinks}link-{payload_label}".
    """
    ps_label = _payload_label(payload_size)
    marks = [pytest.mark.requires_mesh_topology(mesh_shape=mesh_shape, topology=ring_or_linear)]
    if payload_size > MAX_PAYLOAD_SIZE_WH:
        marks.append(
            pytest.mark.skipif(
                not is_blackhole(),
                reason=f"14K payload exceeds Wormhole limit (MAX_PAYLOAD_SIZE_WH={MAX_PAYLOAD_SIZE_WH})",
            )
        )
    ndev = mesh_shape[0] * mesh_shape[1]
    return pytest.param(
        mesh_shape,
        {
            "fabric_config": fabric_config,
            "fabric_router_config": create_fabric_router_config(max_payload_size=payload_size),
        },
        num_links,
        topology,
        payload_size,
        marks=marks,
        id=f"{ring_or_linear}-{ndev}-{num_links}link-{ps_label}",
    )


_TEST_PARAMS = []
for mesh, fab_1d, fab_ring, topo_lin, topo_ring in [
    ((4, 1), ttnn.FabricConfig.FABRIC_1D, ttnn.FabricConfig.FABRIC_1D_RING, ttnn.Topology.Linear, ttnn.Topology.Ring),
    ((8, 1), ttnn.FabricConfig.FABRIC_1D, ttnn.FabricConfig.FABRIC_1D_RING, ttnn.Topology.Linear, ttnn.Topology.Ring),
]:
    for nlinks in (1, 2):
        for ps in (MAX_PAYLOAD_SIZE_WH, MAX_PAYLOAD_SIZE_BH):
            _TEST_PARAMS.append(_make_params(mesh, fab_1d, ps, nlinks, topo_lin, "linear"))
            _TEST_PARAMS.append(_make_params(mesh, fab_ring, ps, nlinks, topo_ring, "ring"))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology, payload_size",
    _TEST_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_dispatch_combine_perf(
    mesh_device,
    num_links,
    topology,
    payload_size,
):
    _check_profiler_env()

    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        SEQ_LEN_PER_CHIP, NUM_ROUTED_EXPERTS, NUM_EXPERTS_PER_TOK, num_devices, dispatch_group_size, CAPACITY_FACTOR
    )

    topo_name = "Ring" if topology == ttnn.Topology.Ring else "Linear"
    ps_label = _payload_label(payload_size)
    logger.info(
        f"\n{'='*70}\n"
        f"  PERF TEST: {topo_name} topology, {num_devices} devices, {num_links} link(s), payload={ps_label}\n"
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

    # Warmup (no profiler reads)
    for _ in range(WARMUP_ITERS):
        tt_dispatched_buffer, tt_metadata = tt_dispatch_module(
            tt_x, tt_weights, tt_indices, tt_expert_offsets, tt_expert_dispatch_table
        )
        tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_expert_token_counts)
        ttnn.synchronize_device(mesh_device)

    # Flush any warmup profiler data
    ttnn.ReadDeviceProfiler(mesh_device)

    # Measure dispatch (device kernel duration per iteration)
    def run_dispatch():
        nonlocal tt_dispatched_buffer, tt_metadata
        tt_dispatched_buffer, tt_metadata = tt_dispatch_module(
            tt_x, tt_weights, tt_indices, tt_expert_offsets, tt_expert_dispatch_table
        )

    def run_combine():
        nonlocal tt_output
        tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_expert_token_counts)

    def run_roundtrip():
        run_dispatch()
        run_combine()

    dispatch_durations = _measure_op(mesh_device, run_dispatch, MEASURE_ITERS)
    combine_durations = _measure_op(mesh_device, run_combine, MEASURE_ITERS)
    roundtrip_durations = _measure_op(mesh_device, run_roundtrip, MEASURE_ITERS)

    d_min, d_med, d_avg, d_max = _stats(dispatch_durations)
    c_min, c_med, c_avg, c_max = _stats(combine_durations)
    r_min, r_med, r_avg, r_max = _stats(roundtrip_durations)

    config_label = f"{topo_name}-{num_devices} ({num_links}L {ps_label})"
    logger.info(
        f"\n{'='*70}\n"
        f"  RESULTS: {topo_name} | {num_devices} devices | {num_links} link(s) | payload={ps_label}\n"
        f"  Device kernel durations ({MEASURE_ITERS} iters after {WARMUP_ITERS} warmup)\n"
        f"{'='*70}\n"
        f"  {'Operation':<20} {'Min (us)':>12} {'Median (us)':>12} {'Avg (us)':>12} {'Max (us)':>12}\n"
        f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}\n"
        f"  {'Dispatch':<20} {d_min:>12.1f} {d_med:>12.1f} {d_avg:>12.1f} {d_max:>12.1f}\n"
        f"  {'Combine':<20} {c_min:>12.1f} {c_med:>12.1f} {c_avg:>12.1f} {c_max:>12.1f}\n"
        f"  {'Dispatch+Combine':<20} {r_min:>12.1f} {r_med:>12.1f} {r_avg:>12.1f} {r_max:>12.1f}\n"
        f"{'='*70}"
    )

    _collected_results.append(
        {
            "config": config_label,
            "topo": topo_name,
            "num_devices": num_devices,
            "num_links": num_links,
            "payload_size": payload_size,
            "dispatch_med": d_med,
            "combine_med": c_med,
            "roundtrip_med": r_med,
        }
    )
