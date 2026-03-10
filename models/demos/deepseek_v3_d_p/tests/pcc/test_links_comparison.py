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

NUM_WARMUP = 10
NUM_ITERATIONS = 100

# Accumulate results across parametrized runs for side-by-side comparison
_results: dict[int, dict[str, float]] = {}


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


def _print_comparison():
    """Print side-by-side comparison table if results for multiple link counts are available."""
    if len(_results) < 2:
        return

    link_counts = sorted(_results.keys())
    base = _results[link_counts[0]]

    header = f"{'metric':<20}"
    for nl in link_counts:
        header += f"  {'%d link' % nl:>12s}"
    header += f"  {'speedup':>10s}"

    rows = []
    for metric in ("dispatch_us", "combine_us", "total_us"):
        label = metric.replace("_us", "").capitalize()
        row = f"{label + ' (us)':<20}"
        vals = [_results[nl][metric] for nl in link_counts]
        for v in vals:
            row += f"  {v:>12.2f}"
        if vals[0] > 0:
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

    _results[num_links] = {
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
