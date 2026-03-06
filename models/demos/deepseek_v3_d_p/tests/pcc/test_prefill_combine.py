"""
Test for TTNN MoE prefill combine operation in isolation.

This test verifies that the TTNN combine operation produces the same output as the
PyTorch reference implementation when combining expert outputs back to token positions.
Uses torch-generated dispatch inputs to isolate the combine operation.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.common import (
    compute_constants,
    create_expert_dispatch_table,
    create_fabric_router_config,
    get_gate_outputs,
    initialize_predictable_test_inputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, capacity_factor",
    [
        (32, 7 * 1024, 16, 4, 2),
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
def test_ttnn_combine(
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
    """Test TTNN combine operation in isolation using torch reference inputs."""

    num_devices = mesh_device.get_num_devices()

    if mesh_device.shape[0] > 1 and mesh_device.shape[1] > 1:
        sp_axis = 0
        num_chips_sp = mesh_device.shape[sp_axis]
        num_chips_rep = mesh_device.shape[1]
    else:
        num_chips_sp = mesh_device.get_num_devices()
        num_chips_rep = 1
        sp_axis = 0 if mesh_device.shape[0] > 1 else 1

    logger.info(f"Testing with {mesh_device.shape=}, {num_devices=} {num_chips_sp=} {num_chips_rep=}")
    ttnn.visualize_mesh_device(mesh_device)

    signpost(
        f"Combine {mesh_device=} {num_devices=} {num_chips_sp=} {num_chips_rep=} {seq_len_per_chip=} {hidden_dim=} "
        f"{n_routed_experts=} {num_experts_per_tok=} {capacity_factor=} {use_predictable_data=} {num_links=} {topology=}"
    )
    print("\n")

    # Compute configuration
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_devices, capacity_factor
    )
    logger.info(f"{experts_per_chip=}, {metadata_len=}, {max_dispatched_tokens_per_expert=}")

    # Step 1: Generate initial inputs using torch
    # For 2D mesh, generate different weights per EP rank
    if use_predictable_data:
        x, weights, indices = initialize_predictable_test_inputs(
            num_chips_sp,
            seq_len_per_chip,
            hidden_dim,
            n_routed_experts,
            num_experts_per_tok,
            max_dispatched_tokens_per_expert,
            num_ep_ranks=num_chips_rep,
        )
        logger.info("Using PREDICTABLE test data for debugging")
    else:
        x, weights, indices = initialize_test_inputs(
            num_chips_sp,
            seq_len_per_chip,
            hidden_dim,
            n_routed_experts,
            num_experts_per_tok,
            max_dispatched_tokens_per_expert,
            seed=42,
            num_ep_ranks=num_chips_rep,
        )
        logger.info("Using RANDOM test data")

    # Compute gate outputs before dispatch (same for all EP ranks since indices are shared)
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

    # Initialize torch dispatch module with num_ep_ranks support
    torch_dispatch_module = TorchDispatchModule(
        num_chips=num_chips_sp,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        num_ep_ranks=num_chips_rep,
        expert_dispatch_table=expert_dispatch_table,
    )

    # Run dispatch for each EP rank with rank-specific weights
    dispatched_buffer, dispatched_metadata = torch_dispatch_module(x, weights, indices, chip_to_n_routed_expert_offset)

    logger.info("Torch dispatch outputs (OG):")
    logger.info(f"  {dispatched_buffer.shape=}")
    logger.info(f"  {dispatched_metadata.shape=}")

    # Transform logical chip IDs to linearized coords
    # metadata[..., 0] contains the destination logical chip ID
    for r in range(num_chips_rep):
        # dest_linearized = dest_logical * num_chips_rep + replica_index
        dispatched_metadata[r, :, :, :, 0] = dispatched_metadata[r, :, :, :, 0] * num_chips_rep + r

    # Use different sharding: shard both dimensions
    mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(1, 0),  # Shard tensor dim 1 across mesh rows, tensor dim 0 across mesh cols
    )

    tt_dispatched_buffer = ttnn.from_torch(
        dispatched_buffer,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )

    tt_dispatched_metadata = ttnn.from_torch(
        dispatched_metadata,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    tt_experts_tok_counter = ttnn.from_torch(
        experts_tok_counter,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    torch_combine = TorchCombineModule(
        num_chips_sp=num_chips_sp,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        num_ep_ranks=num_chips_rep,
    )

    torch_output = torch_combine(dispatched_buffer, dispatched_metadata, experts_tok_counter)
    logger.info(f"Torch combine output shape: {torch_output.shape}")

    # Step 5: Run ttnn combine
    tt_combine = TtCombineModule(
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

    tt_output = tt_combine(
        tt_dispatched_buffer,
        tt_dispatched_metadata,
        tt_experts_tok_counter,
    )

    logger.info(f"TTNN combine output shape: {tt_output.shape}")

    # Step 6: Convert ttnn output to torch for comparison
    mesh_composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(
            dims=[1, 0],  # Axis 0: replicated; Axis 1: shard on tensor dim 0
        ),
    )
    logger.warning(f"{torch_output.shape=}")
    logger.warning(f"{tt_output.shape=}")

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=mesh_composer,
        dtype=torch.bfloat16,
    )
    logger.warning(f"{tt_output_torch.shape=}")

    # Step 7: Compute PCC and verify correctness
    logger.info("Computing PCC between torch and ttnn combine outputs...")

    assert (
        tt_output_torch.shape[0] == num_chips_rep
    ), f"Mismatch in replicated dimension: expected {num_chips_rep}, got {tt_output_torch.shape[0]}"
    assert (
        tt_output_torch.shape[1] == num_chips_sp
    ), f"Mismatch in sharded dimension: expected {num_chips_sp}, got {tt_output_torch.shape[1]}"

    # Quick sanity check of first elements
    logger.info(f"Sample torch output [0, 0, 0, :5]: {torch_output[0, 0, 0, :5]}")
    logger.info(f"Sample ttnn output [0, 0, 0, 0, :5]:  {tt_output_torch[0, 0, 0, 0, :5]}")
    if num_chips_sp > 1:
        logger.info(f"Sample torch output [1, 0, 0, :5]: {torch_output[1, 0, 0, :5]}")
        logger.info(f"Sample ttnn output [0, 1, 0, 0, :5]:  {tt_output_torch[0, 1, 0, 0, :5]}")

    # Detailed per-chip, per-token, per-expert comparison
    # NOTE: Current combine kernel does NOT all-reduce across EP ranks.
    # Each EP rank's output only contains data for tokens that EP rank processed.
    # Output positions not written by local combine contain uninitialized garbage.
    # This comparison only checks the EP rank that actually processed each token.
    data_ok = True
    mismatches = []
    matches = 0
    total_slots = 0

    experts_per_rank = n_routed_experts // num_chips_rep
    logger.info("Comparing combine output slots (per EP rank that processed each token)...")
    for chip_id in range(num_chips_sp):
        for token_id in range(seq_len_per_chip):
            for topk_idx in range(num_experts_per_tok):
                total_slots += 1

                # Determine which EP rank processed this (chip, token, topk)
                expert_id = indices[chip_id, token_id, topk_idx].item()
                ep_rank = expert_id // experts_per_rank

                torch_data = torch_output[chip_id, token_id, topk_idx]
                ttnn_data = tt_output_torch[ep_rank, chip_id, token_id, topk_idx]

                if torch.allclose(torch_data, ttnn_data, atol=1e-2, rtol=1e-2):
                    matches += 1
                else:
                    data_ok = False
                    max_diff = torch.max(torch.abs(torch_data.float() - ttnn_data.float())).item()
                    mismatches.append((ep_rank, chip_id, token_id, topk_idx, max_diff))

    # Report statistics
    logger.info(f"Matches: {matches}/{total_slots} ({100.0*matches/total_slots:.2f}%)")

    if not data_ok:
        # Show first 10 mismatches in detail
        logger.warning(f"Found {len(mismatches)} mismatches. Showing first 10:")
        for i, (ep_rank, chip_id, token_id, topk_idx, max_diff) in enumerate(mismatches[:10]):
            torch_sample = torch_output[chip_id, token_id, topk_idx, :5]
            ttnn_sample = tt_output_torch[ep_rank, chip_id, token_id, topk_idx, :5]
            logger.error(
                f"  [{i}] Mismatch at ep_rank={ep_rank}, chip={chip_id}, token={token_id}, topk={topk_idx}: "
                f"max_diff={max_diff:.6f}"
            )
            logger.error(f"      torch[:5]={torch_sample}")
            logger.error(f"      ttnn[:5]={ttnn_sample}")

        # Show per-chip statistics
        logger.info("\nPer-chip statistics:")
        for chip_id in range(num_chips_sp):
            chip_mismatches = [m for m in mismatches if m[1] == chip_id]
            chip_total = seq_len_per_chip * num_experts_per_tok
            chip_matches = chip_total - len(chip_mismatches)
            logger.info(
                f"  Chip {chip_id}: {chip_matches}/{chip_total} matches " f"({100.0*chip_matches/chip_total:.2f}%)"
            )

    # Assert all data matches
    assert data_ok, f"Combine data mismatch! {matches}/{total_slots} slots matched. Check logs for details."

    logger.info("✅ TTNN combine operation matches torch reference!")
