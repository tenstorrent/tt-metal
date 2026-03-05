"""
Benchmark test comparing dispatch+combine performance with 1 vs 2 ethernet links.

Runs the full dispatch→combine round-trip multiple times and reports per-iteration
timing so that Tracy profiles and perf reports can be compared across link counts.
"""

import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.common import (
    compute_constants,
    create_fabric_router_config,
    initialize_predictable_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule

NUM_WARMUP = 2
NUM_ITERATIONS = 5


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

    dispatch_times = []
    combine_times = []

    for i in range(NUM_WARMUP + NUM_ITERATIONS):
        label = "warmup" if i < NUM_WARMUP else f"iter-{i - NUM_WARMUP}"
        signpost(f"dispatch+combine  links={num_links}  {label}")

        t0 = time.perf_counter()
        tt_dispatched_buffer, tt_metadata, experts_tok_counter, offsets, cum_sum = tt_dispatch_module(
            tt_x, tt_weights, tt_indices
        )
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()

        tt_experts_tok_counter = ttnn.from_torch(
            experts_tok_counter,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
        )

        t2 = time.perf_counter()
        tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_experts_tok_counter)
        ttnn.synchronize_device(mesh_device)
        t3 = time.perf_counter()

        if i >= NUM_WARMUP:
            dispatch_times.append(t1 - t0)
            combine_times.append(t3 - t2)
            logger.info(
                f"[{label}] dispatch={1000*(t1-t0):.2f}ms  combine={1000*(t3-t2):.2f}ms  "
                f"total={1000*(t1-t0+t3-t2):.2f}ms"
            )

    avg_dispatch = 1000 * sum(dispatch_times) / len(dispatch_times)
    avg_combine = 1000 * sum(combine_times) / len(combine_times)
    logger.info(
        f"\n{'='*60}\n"
        f"  num_links={num_links}  ({NUM_ITERATIONS} iters after {NUM_WARMUP} warmup)\n"
        f"  avg dispatch : {avg_dispatch:.2f} ms\n"
        f"  avg combine  : {avg_combine:.2f} ms\n"
        f"  avg total    : {avg_dispatch + avg_combine:.2f} ms\n"
        f"{'='*60}"
    )

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
