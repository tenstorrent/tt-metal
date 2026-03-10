"""
Benchmark test comparing dispatch+combine performance with 1 vs 2 ethernet links.

Runs the full dispatch→combine round-trip multiple times and reports device kernel
durations extracted from the device profiler.

Usage:
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
      pytest models/demos/deepseek_v3_d_p/tests/pcc/test_links_comparison.py -v -s

Without profiler env vars, falls back to wall-clock timing.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.common import (
    compute_constants,
    create_fabric_router_config,
    initialize_predictable_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch_combined import TtDispatchCombinedModule

NUM_WARMUP = 10
NUM_ITERATIONS = 100

# Accumulate results across parametrized runs for side-by-side comparison
_results: dict[str, dict[str, float]] = {}


def _extract_profiler_durations(mesh_device):
    """
    Call ReadDeviceProfiler once, then extract per-program kernel durations.

    Programs execute in order: dispatch, combine, dispatch, combine, ...
    For each program we take the max duration across devices (the mesh op
    finishes when the slowest device finishes).

    Returns (dispatch_durations_ns, combine_durations_ns) or (None, None).
    """
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
        latest = ttnn.get_latest_programs_perf_data()
    except Exception:
        return None, None
    if not latest:
        return None, None

    device_ids = sorted(latest.keys())
    if not device_ids:
        return None, None

    num_programs_per_device = min(len(latest[did]) for did in device_ids)
    if num_programs_per_device == 0:
        return None, None

    per_program_durations = []
    for prog_idx in range(num_programs_per_device):
        max_dur = None
        for did in device_ids:
            p = latest[did][prog_idx]
            for key in ("DEVICE KERNEL DURATION [ns]", "DEVICE FW DURATION [ns]"):
                if key in p.program_analyses_results:
                    d = p.program_analyses_results[key].duration
                    if d is not None:
                        max_dur = max(max_dur, d) if max_dur is not None else d
                    break
        if max_dur is not None:
            per_program_durations.append(int(max_dur))

    dispatch_durations = per_program_durations[0::2]
    combine_durations = per_program_durations[1::2]
    return dispatch_durations, combine_durations


def _extract_dispatch_only_profiler_durations(mesh_device):
    """Extract profiler durations when only dispatch programs are present (no combine)."""
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
        latest = ttnn.get_latest_programs_perf_data()
        device_ids = sorted(latest.keys()) if latest else []
        if not device_ids:
            return []
        num_programs = min(len(latest[did]) for did in device_ids)
        durations = []
        for prog_idx in range(num_programs):
            max_dur = None
            for did in device_ids:
                p = latest[did][prog_idx]
                for key in ("DEVICE KERNEL DURATION [ns]", "DEVICE FW DURATION [ns]"):
                    if key in p.program_analyses_results:
                        d = p.program_analyses_results[key].duration
                        if d is not None:
                            max_dur = max(max_dur, d) if max_dur is not None else d
                        break
            if max_dur is not None:
                durations.append(int(max_dur))
        return durations
    except Exception:
        return []


def _print_comparison():
    """Print side-by-side comparison table if results for multiple variants are available."""
    if len(_results) < 2:
        return

    variants = sorted(_results.keys())

    header = f"{'metric':<20}"
    for v in variants:
        header += f"  {v:>14s}"
    header += f"  {'speedup':>10s}"

    rows = []
    for metric in ("dispatch_us", "combine_us", "total_us"):
        label = metric.replace("_us", "").capitalize()
        row = f"{label + ' (us)':<20}"
        vals = [_results[v].get(metric, 0.0) for v in variants]
        for val in vals:
            row += f"  {val:>14.2f}"
        if vals[0] > 0 and vals[-1] > 0:
            row += f"  {vals[0] / vals[-1]:>9.2f}x"
        rows.append(row)

    sep = "=" * len(header)
    table = "\n".join([sep, header, "-" * len(header)] + rows + [sep])
    logger.info(f"\n{table}")


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, capacity_factor",
    [
        (512, 7168, 16, 4, 2),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=15232),
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
                "fabric_router_config": create_fabric_router_config(max_payload_size=15232),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="linear-4-2link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_links_comparison(
    mesh_device,
    seq_len_per_chip,
    hidden_dim,
    n_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    num_links,
    topology,
):
    """Benchmark dispatch+combine with a given link count and verify correctness."""

    num_devices = num_chips = mesh_device.get_num_devices()
    logger.info(f"=== num_links={num_links}  mesh={mesh_device.shape}  devices={num_devices} ===")
    ttnn.visualize_mesh_device(mesh_device)

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor
    )

    x, weights, indices = initialize_predictable_test_inputs(
        num_chips=num_chips,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
    )

    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, None))

    tt_x = ttnn.from_torch(
        x, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_weights = ttnn.from_torch(
        weights, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
    )

    tt_dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        cluster_axis=0,
        num_links=num_links,
        topology=topology,
    )

    tt_combine_module = TtCombineModule(
        mesh_device=mesh_device,
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=0,
        num_links=num_links,
        topology=topology,
    )

    # Warmup
    for _ in range(NUM_WARMUP):
        tt_dispatched_buffer, tt_metadata, experts_tok_counter, offsets, cum_sum = tt_dispatch_module(
            tt_x, tt_weights, tt_indices
        )
        tt_experts_tok_counter = ttnn.from_torch(
            experts_tok_counter,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
        )
        tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_experts_tok_counter)
    ttnn.synchronize_device(mesh_device)

    # Measured iterations — run all, then read profiler once
    t_wall_start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        tt_dispatched_buffer, tt_metadata, experts_tok_counter, offsets, cum_sum = tt_dispatch_module(
            tt_x, tt_weights, tt_indices
        )
        ttnn.synchronize_device(mesh_device)

        tt_experts_tok_counter = ttnn.from_torch(
            experts_tok_counter,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
        )

        tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_experts_tok_counter)
        ttnn.synchronize_device(mesh_device)
    wall_total_ns = int((time.perf_counter() - t_wall_start) * 1e9)

    dispatch_durations, combine_durations = _extract_profiler_durations(mesh_device)

    if dispatch_durations and combine_durations:
        avg_dispatch_us = sum(dispatch_durations) / len(dispatch_durations) / 1e3
        avg_combine_us = sum(combine_durations) / len(combine_durations) / 1e3
        source = "profiler"
    else:
        avg_dispatch_us = wall_total_ns / NUM_ITERATIONS / 1e3
        avg_combine_us = 0
        source = "wall"
        logger.warning("Device profiler unavailable, reporting wall-clock total (dispatch+combine)")

    avg_total_us = avg_dispatch_us + avg_combine_us

    result_key = f"{num_links}link"
    _results[result_key] = {
        "dispatch_us": avg_dispatch_us,
        "combine_us": avg_combine_us,
        "total_us": avg_total_us,
    }

    logger.info(
        f"num_links={num_links}: dispatch={avg_dispatch_us:.2f} us  "
        f"combine={avg_combine_us:.2f} us  total={avg_total_us:.2f} us  ({source})"
    )

    _print_comparison()

    # Correctness check on last iteration
    mesh_composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(dims=[0, 1]),
    )
    y = ttnn.to_torch(tt_output, mesh_composer=mesh_composer, dtype=torch.bfloat16)
    y = y / num_experts_per_tok
    y = y.sum(dim=2)

    max_diff = torch.max(torch.abs(x - y)).item()
    logger.info(f"Correctness check — max abs diff: {max_diff}")
    assert torch.allclose(x, y, atol=1e-6), f"Round-trip mismatch: max diff {max_diff}"
    logger.info("Correctness verified.")


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, capacity_factor",
    [
        (512, 7168, 16, 4, 2),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=15232),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="combined-linear-4-2link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_combined_dispatch(
    mesh_device,
    seq_len_per_chip,
    hidden_dim,
    n_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    num_links,
    topology,
):
    """Benchmark combined dispatch (metadata+payload in one transfer) with 2 links."""

    num_devices = num_chips = mesh_device.get_num_devices()
    logger.info(f"=== Combined dispatch: num_links={num_links}  mesh={mesh_device.shape}  devices={num_devices} ===")
    ttnn.visualize_mesh_device(mesh_device)

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor
    )

    x, weights, indices = initialize_predictable_test_inputs(
        num_chips=num_chips,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
    )

    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, None))

    tt_x = ttnn.from_torch(
        x, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_weights = ttnn.from_torch(
        weights, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
    )

    tt_dispatch_combined_module = TtDispatchCombinedModule(
        mesh_device=mesh_device,
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        cluster_axis=0,
        num_links=num_links,
        topology=topology,
    )

    tt_dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        cluster_axis=0,
        num_links=num_links,
        topology=topology,
    )

    tt_combine_module = TtCombineModule(
        mesh_device=mesh_device,
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=0,
        num_links=num_links,
        topology=topology,
    )

    # Run normal dispatch once to produce valid combine inputs
    tt_dispatched_buffer, tt_metadata, experts_tok_counter_torch, _, _ = tt_dispatch_module(
        tt_x, tt_weights, tt_indices
    )
    tt_experts_tok_counter = ttnn.from_torch(
        experts_tok_counter_torch,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # --- Warmup combined dispatch ---
    for _ in range(NUM_WARMUP):
        tt_combined_buffer, _, _, _ = tt_dispatch_combined_module(tt_x, tt_weights, tt_indices)
    ttnn.synchronize_device(mesh_device)

    # --- Measure combined dispatch ---
    t_wall_start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        tt_combined_buffer, _, _, _ = tt_dispatch_combined_module(tt_x, tt_weights, tt_indices)
        ttnn.synchronize_device(mesh_device)
    wall_dispatch_ns = int((time.perf_counter() - t_wall_start) * 1e9)

    dispatch_durations = _extract_dispatch_only_profiler_durations(mesh_device)

    if dispatch_durations:
        avg_dispatch_us = sum(dispatch_durations) / len(dispatch_durations) / 1e3
        dispatch_source = "profiler"
    else:
        avg_dispatch_us = wall_dispatch_ns / NUM_ITERATIONS / 1e3
        dispatch_source = "wall"
        logger.warning("Device profiler unavailable for combined dispatch, reporting wall-clock")

    logger.info(f"Combined dispatch num_links={num_links}: dispatch={avg_dispatch_us:.2f} us  ({dispatch_source})")
    logger.info(f"Combined buffer shape: {tt_combined_buffer.shape}")

    # --- Warmup normal combine (reuse dispatch outputs) ---
    for _ in range(NUM_WARMUP):
        tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_experts_tok_counter)
    ttnn.synchronize_device(mesh_device)

    # --- Measure normal combine only ---
    t_wall_start = time.perf_counter()
    for i in range(NUM_ITERATIONS):
        tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_experts_tok_counter)
        ttnn.synchronize_device(mesh_device)
    wall_combine_ns = int((time.perf_counter() - t_wall_start) * 1e9)

    combine_durations = _extract_dispatch_only_profiler_durations(mesh_device)

    if combine_durations:
        avg_combine_us = sum(combine_durations) / len(combine_durations) / 1e3
        combine_source = "profiler"
    else:
        avg_combine_us = wall_combine_ns / NUM_ITERATIONS / 1e3
        combine_source = "wall"
        logger.warning("Device profiler unavailable for combine, reporting wall-clock")

    logger.info(f"Normal combine num_links={num_links}: combine={avg_combine_us:.2f} us  ({combine_source})")

    avg_total_us = avg_dispatch_us + avg_combine_us

    result_key = f"combined-{num_links}link"
    _results[result_key] = {
        "dispatch_us": avg_dispatch_us,
        "combine_us": avg_combine_us,
        "total_us": avg_total_us,
    }

    _print_comparison()

    logger.info("Skipping correctness check — no combined-aware combine op yet.")


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, capacity_factor",
    [
        (512, 7168, 16, 4, 2),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=15232),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="smoke-linear-4-2link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_smoke_all_ops(
    mesh_device,
    seq_len_per_chip,
    hidden_dim,
    n_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    num_links,
    topology,
):
    """Run dispatch, combine, and combined dispatch once each to verify they execute without error."""

    num_devices = num_chips = mesh_device.get_num_devices()
    logger.info(f"=== Smoke test: num_links={num_links}  mesh={mesh_device.shape}  devices={num_devices} ===")

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor
    )

    x, weights, indices = initialize_predictable_test_inputs(
        num_chips=num_chips,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
    )

    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, None))

    tt_x = ttnn.from_torch(
        x, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_weights = ttnn.from_torch(
        weights, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
    )

    # --- Dispatch ---
    logger.info("Running dispatch...")
    tt_dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        cluster_axis=0,
        num_links=num_links,
        topology=topology,
    )
    tt_dispatched_buffer, tt_metadata, experts_tok_counter, offsets, cum_sum = tt_dispatch_module(
        tt_x, tt_weights, tt_indices
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(f"Dispatch done. Buffer shape: {tt_dispatched_buffer.shape}")

    # --- Combine ---
    logger.info("Running combine...")
    tt_combine_module = TtCombineModule(
        mesh_device=mesh_device,
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=0,
        num_links=num_links,
        topology=topology,
    )
    tt_experts_tok_counter = ttnn.from_torch(
        experts_tok_counter,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_experts_tok_counter)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"Combine done. Output shape: {tt_output.shape}")

    # --- Combined dispatch ---
    logger.info("Running combined dispatch...")
    tt_dispatch_combined_module = TtDispatchCombinedModule(
        mesh_device=mesh_device,
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        cluster_axis=0,
        num_links=num_links,
        topology=topology,
    )
    tt_combined_buffer, combined_counter, combined_offsets, combined_cum_sum = tt_dispatch_combined_module(
        tt_x, tt_weights, tt_indices
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(f"Combined dispatch done. Buffer shape: {tt_combined_buffer.shape}")

    logger.info("All operations completed successfully.")
