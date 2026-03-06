"""
Test for end-to-end TTNN MoE dispatch→combine round-trip.

This test verifies that tokens dispatched to experts using TTNN dispatch and then
combined back using TTNN combine produce the original input after host-side reduction,
validating the full round-trip through TTNN dispatch and combine operations.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.common import (
    compute_constants,
    create_expert_dispatch_table,
    create_fabric_router_config,
    get_gate_outputs,
    initialize_predictable_test_inputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule


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
            (2, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 1), topology="linear"),
            id="linear-2",
        ),
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="linear-4",
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
            id="ring-4",
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
            id="linear-8",
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
            id="ring-8",
        ),
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-4x2"),
            id="mesh-2x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("use_predictable_data", [True, False], ids=["predictable", "random"])
def test_ttnn_dispatch_combine(
    mesh_device,
    seq_len_per_chip,
    hidden_dim,
    n_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    num_links,
    topology,
    use_predictable_data,
):
    """Test end-to-end TTNN dispatch→combine round-trip with host reduction."""

    num_devices = mesh_device.get_num_devices()

    # Compute sp_axis and chip counts for 2D mesh handling
    if mesh_device.shape[0] > 1 and mesh_device.shape[1] > 1:
        sp_axis = 0
        num_chips_sp = mesh_device.shape[sp_axis]
        num_chips_rep = mesh_device.shape[1]
    else:
        sp_axis = 0 if mesh_device.shape[0] > 1 else 1
        num_chips_sp = num_devices
        num_chips_rep = 1

    logger.info(f"Testing with {mesh_device.shape=}, {num_devices=} {num_chips_sp=} {num_chips_rep=}")
    ttnn.visualize_mesh_device(mesh_device)

    signpost(
        f"TTNN Dispatch+Combine {mesh_device=} {num_devices=} {num_chips_sp=} {num_chips_rep=} "
        f"{seq_len_per_chip=} {hidden_dim=} {n_routed_experts=} {num_experts_per_tok=} "
        f"{capacity_factor=} {use_predictable_data=}"
    )
    print("\n")

    # Compute configuration constants (use num_chips_sp for dispatch/combine parallelism)
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_devices, capacity_factor
    )
    logger.info(f"{experts_per_chip=}, {metadata_len=}, {max_dispatched_tokens_per_expert=}")

    # Generate test inputs
    # For 2D mesh, generate different weights per EP rank
    if use_predictable_data:
        x, weights, indices = initialize_predictable_test_inputs(
            num_chips=num_chips_sp,
            seq_len_per_chip=seq_len_per_chip,
            hidden_dim=hidden_dim,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            num_ep_ranks=num_chips_rep,
        )
        logger.info("Using PREDICTABLE test data for debugging")
    else:
        x, weights, indices = initialize_test_inputs(
            num_chips=num_chips_sp,
            seq_len_per_chip=seq_len_per_chip,
            hidden_dim=hidden_dim,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            seed=42,
            num_ep_ranks=num_chips_rep,
        )
        logger.info("Using RANDOM test data")

    logger.info(f"Input shapes: {x.shape=}, {weights.shape=}, {indices.shape=}")

    # x, weights, and indices: sharded across SP axis, replicated across EP ranks
    mesh_mapper_replicated = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(sp_axis, None),  # Shard on sp_axis, replicate on other axis
    )

    tt_x = ttnn.from_torch(
        x, mesh_mapper=mesh_mapper_replicated, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_weights = ttnn.from_torch(
        weights,
        mesh_mapper=mesh_mapper_replicated,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=mesh_mapper_replicated, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
    )

    # Initialize TTNN dispatch module
    tt_dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        num_chips=num_chips_sp,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
    )

    # Compute gate outputs (offsets and token counts) before dispatch
    chip_to_n_routed_expert_offset, experts_tok_counter, cum_sum = get_gate_outputs(
        indices,
        num_chips_sp,
        n_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
    )

    # Create expert dispatch table
    expert_dispatch_table = create_expert_dispatch_table(
        n_routed_experts=n_routed_experts,
        num_chips_sp=num_chips_sp,
        num_chips_rep=num_chips_rep,
    )
    logger.info(f"{expert_dispatch_table.shape=}")
    logger.info(f"expert_dispatch_table:\n{expert_dispatch_table}")

    # Run TTNN dispatch
    logger.info("Running TTNN dispatch...")
    tt_chip_to_n_routed_expert_offset = TtDispatchModule.shard_offset_tensor(
        mesh_device, chip_to_n_routed_expert_offset
    )
    tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)
    tt_dispatched_buffer, tt_metadata = tt_dispatch_module(
        tt_x, tt_weights, tt_indices, tt_chip_to_n_routed_expert_offset, tt_expert_dispatch_table
    )
    ttnn.synchronize_device(mesh_device)
    logger.info("Dispatch complete!")

    logger.info(f"Dispatch outputs: {experts_tok_counter.shape=}")
    logger.info(f"  {experts_tok_counter=}")
    logger.info(f"  {chip_to_n_routed_expert_offset.shape=}, {chip_to_n_routed_expert_offset=}")
    logger.info(f"  {cum_sum.shape=}, {cum_sum=}")

    # Convert counter to TTNN tensor for combine module
    # For 2D mesh, use dims=(1, 0) to shard across both axes
    logger.info(f"Converting counter to TTNN: {experts_tok_counter.shape=}, {experts_tok_counter.dtype=}")
    logger.info(f"  Counter values: {experts_tok_counter=}")
    mesh_mapper_2d = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(1, 0),  # Shard tensor dim 1 across mesh rows, tensor dim 0 across mesh cols
    )
    tt_experts_tok_counter = ttnn.from_torch(
        experts_tok_counter,
        mesh_mapper=mesh_mapper_2d,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    logger.info(f"  TTNN counter shape: {tt_experts_tok_counter.shape}")
    ttnn.visualize_tensor(tt_experts_tok_counter)  # , header="Experts Token Counter")

    # Initialize TTNN combine module
    tt_combine_module = TtCombineModule(
        mesh_device=mesh_device,
        num_chips_sp=num_chips_sp,
        num_ep_ranks=num_chips_rep,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
    )

    # Run TTNN combine
    logger.info("Running TTNN combine...")
    tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_experts_tok_counter)
    logger.info("Combine complete!")

    logger.info(f"Combine output shape: {tt_output.shape}")

    # Convert TTNN output back to torch
    mesh_composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(
            dims=[1, 0],  # Axis 0: replicated; Axis 1: shard on tensor dim 0
        ),
    )

    y = ttnn.to_torch(tt_output, mesh_composer=mesh_composer, dtype=torch.bfloat16)

    # Host-side reduction
    # Output shape after mesh composition with dims=[1, 0]:
    # (num_chips_rep, num_chips_sp, 1, seq_len_per_chip, num_experts_per_tok, hidden_dim)
    # Note: Even for 1D mesh with num_chips_rep=1, the first dimension is still there
    logger.info(f"Before reduction: {y.shape=}")
    y = y.squeeze(-4)  # Remove extra dimension (the "1") added for 2D mesh composition
    logger.info(f"After squeeze: {y.shape=}")
    # y shape is now: (num_chips_rep, num_chips_sp, seq_len_per_chip, num_experts_per_tok, hidden_dim)

    # Verify round-trip correctness
    # NOTE: Current combine kernel does NOT all-reduce across EP ranks.
    # Each EP rank's output only contains data for tokens that EP rank processed.
    # Output positions not written by local combine contain uninitialized garbage.
    # We validate per (chip, token, topk) using the EP rank that actually processed it.
    logger.info("Verifying round-trip correctness (per EP rank that processed each token)...")

    experts_per_rank = n_routed_experts // num_chips_rep
    all_match = True
    matches = 0
    total_slots = 0
    mismatches = []

    for chip_id in range(num_chips_sp):
        for token_id in range(seq_len_per_chip):
            for topk_idx in range(num_experts_per_tok):
                total_slots += 1

                # Determine which EP rank processed this (chip, token, topk)
                expert_id = indices[chip_id, token_id, topk_idx].item()
                ep_rank = expert_id // experts_per_rank

                # Input token
                x_data = x[chip_id, token_id]
                # Output from the EP rank that processed this token
                y_data = y[ep_rank, chip_id, token_id, topk_idx]

                if torch.allclose(x_data, y_data, atol=1e-2, rtol=1e-2):
                    matches += 1
                else:
                    all_match = False
                    max_diff = torch.max(torch.abs(x_data.float() - y_data.float())).item()
                    mismatches.append((ep_rank, chip_id, token_id, topk_idx, max_diff))

    logger.info(f"Matches: {matches}/{total_slots} ({100.0*matches/total_slots:.2f}%)")

    if not all_match:
        logger.warning(f"Found {len(mismatches)} mismatches. Showing first 10:")
        for i, (ep_rank, chip_id, token_id, topk_idx, max_diff) in enumerate(mismatches[:10]):
            x_sample = x[chip_id, token_id, :5]
            y_sample = y[ep_rank, chip_id, token_id, topk_idx, :5]
            logger.error(
                f"  [{i}] Mismatch at ep_rank={ep_rank}, chip={chip_id}, token={token_id}, topk={topk_idx}: "
                f"max_diff={max_diff:.6f}"
            )
            logger.error(f"      x[:5]={x_sample}")
            logger.error(f"      y[:5]={y_sample}")

    assert all_match, f"Round-trip mismatch! {matches}/{total_slots} slots matched. Check logs for details."

    logger.info("✅ TTNN dispatch→combine round-trip matches input!")
