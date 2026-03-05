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
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips_sp, capacity_factor
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

    # Step 2: Run torch dispatch to generate combine inputs (once per EP rank)
    def create_torch_dispatch():
        return TorchDispatchModule(
            num_chips=num_chips_sp,
            experts_per_chip=experts_per_chip,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            seq_len_per_chip=seq_len_per_chip,
            hidden_dim=hidden_dim,
        )

    # Compute gate outputs before dispatch (same for all EP ranks since indices are shared)
    chip_to_n_routed_expert_offset, experts_tok_counter, cum_sum = get_gate_outputs(
        indices,
        num_chips_sp,
        n_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
    )

    # Run dispatch for each EP rank with rank-specific weights
    dispatched_buffer_per_rank = []
    dispatched_metadata_per_rank = []
    experts_tok_counter_per_rank = []
    for r in range(num_chips_rep):
        torch_dispatch = create_torch_dispatch()
        dispatched_buffer, dispatched_metadata = torch_dispatch(x, weights[r], indices, chip_to_n_routed_expert_offset)
        dispatched_buffer_per_rank.append(dispatched_buffer)
        dispatched_metadata_per_rank.append(dispatched_metadata)
        experts_tok_counter_per_rank.append(experts_tok_counter)  # Same for all ranks
        logger.info(f"Torch dispatch rank {r}: {dispatched_buffer.shape=}, {dispatched_metadata.shape=}")

    logger.info("Torch dispatch outputs (4D):")
    logger.info(f"  {dispatched_buffer_per_rank[0].shape=}")
    logger.info(f"  {dispatched_metadata_per_rank[0].shape=}")
    logger.info(f"  {experts_tok_counter_per_rank[0].shape=}")

    # Keep original 4D tensors for torch reference
    dispatched_buffer_4d_per_rank = dispatched_buffer_per_rank
    dispatched_metadata_4d_per_rank = dispatched_metadata_per_rank

    # Reshape to 5D format to match actual dispatch device operation output
    # From: (num_chips, experts_per_chip, max_tok, dim) -> (num_chips, 1, experts_per_chip, max_tok, dim)
    dispatched_buffer_per_rank = [buf.unsqueeze(1) for buf in dispatched_buffer_per_rank]
    dispatched_metadata_per_rank = [meta.unsqueeze(1) for meta in dispatched_metadata_per_rank]

    logger.info("Reshaped to 5D format for TTNN:")
    logger.info(f"  {dispatched_buffer_per_rank[0].shape=}")
    logger.info(f"  {dispatched_metadata_per_rank[0].shape=}")

    # Step 3: Convert torch tensors to ttnn tensors
    # For 2D mesh, we need to transform metadata to use linearized coords instead of logical chip IDs
    # Each replica needs different metadata: dest_linearized = dest_logical * num_chips_rep + replica_index
    if num_chips_rep > 1:
        # Stack per-rank data into a single tensor
        # Current shape per rank after unsqueeze: (num_chips_sp, 1, experts_per_chip, max_tok, dim)
        # New shape: (num_chips_rep, num_chips_sp, 1, experts_per_chip, max_tok, dim)
        dispatched_buffer_expanded = torch.stack(dispatched_buffer_per_rank, dim=0)
        dispatched_metadata_expanded = torch.stack(dispatched_metadata_per_rank, dim=0)
        experts_tok_counter_expanded = torch.stack(experts_tok_counter_per_rank, dim=0)

        # Transform logical chip IDs to linearized coords
        # metadata[..., 0] contains the destination logical chip ID
        for r in range(num_chips_rep):
            # dest_linearized = dest_logical * num_chips_rep + replica_index
            dispatched_metadata_expanded[r, :, :, :, :, 0] = (
                dispatched_metadata_per_rank[r][:, :, :, :, 0] * num_chips_rep + r
            )

        logger.info(f"  Expanded for 2D mesh: {dispatched_buffer_expanded.shape=}")
        logger.info(f"  Expanded for 2D mesh: {dispatched_metadata_expanded.shape=}")
        logger.info(f"  Expanded for 2D mesh: {experts_tok_counter_expanded.shape=}")

        # Use different sharding: shard both dimensions
        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(1, 0),  # Shard tensor dim 1 across mesh rows, tensor dim 0 across mesh cols
        )

        tt_dispatched_buffer = ttnn.from_torch(
            dispatched_buffer_expanded,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )

        tt_dispatched_metadata = ttnn.from_torch(
            dispatched_metadata_expanded,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
        )

        tt_experts_tok_counter = ttnn.from_torch(
            experts_tok_counter_expanded,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
        )
    else:
        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(sp_axis, None),  # Shard on sp_axis, replicate on other axis
        )

        tt_dispatched_buffer = ttnn.from_torch(
            dispatched_buffer_per_rank[0],
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )

        tt_dispatched_metadata = ttnn.from_torch(
            dispatched_metadata_per_rank[0],
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
        )

        tt_experts_tok_counter = ttnn.from_torch(
            experts_tok_counter_per_rank[0],
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
        )

    # Step 4: Run torch combine for reference output (once per EP rank)
    torch_output_per_rank = []
    for r in range(num_chips_rep):
        torch_combine = TorchCombineModule(
            num_chips=num_chips_sp,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=seq_len_per_chip,
        )

        torch_output = torch_combine(
            dispatched_buffer_4d_per_rank[r],
            dispatched_metadata_4d_per_rank[r],
            experts_tok_counter_per_rank[r],
        )
        torch_output_per_rank.append(torch_output)
        logger.info(f"Torch combine rank {r} output shape: {torch_output.shape}")

    # Step 5: Run ttnn combine
    tt_combine = TtCombineModule(
        mesh_device=mesh_device,
        num_chips=num_chips_sp,
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
    logger.info(f"Sample torch output [0][0, 0, 0, :5]: {torch_output_per_rank[0][0, 0, 0, :5]}")
    logger.info(f"Sample ttnn output [0, 0, 0, 0, :5]:  {tt_output_torch[0, 0, 0, 0, :5]}")
    if num_chips_sp > 1:
        logger.info(f"Sample torch output [0][1, 0, 0, :5]: {torch_output_per_rank[0][1, 0, 0, :5]}")
        logger.info(f"Sample ttnn output [0, 1, 0, 0, :5]:  {tt_output_torch[0, 1, 0, 0, :5]}")

    # Detailed per-chip, per-token, per-expert comparison
    data_ok = True
    mismatches = []
    matches = 0
    total_slots = 0

    logger.info("Comparing ALL combine output slots...")
    for r in range(num_chips_rep):
        torch_output = torch_output_per_rank[r]
        for chip_id in range(num_chips_sp):
            for token_id in range(seq_len_per_chip):
                for topk_idx in range(num_experts_per_tok):
                    total_slots += 1
                    # torch uses logical chip ID (per-rank reference)
                    torch_data = torch_output[chip_id, token_id, topk_idx]
                    # ttnn has extra replicated dim at front
                    ttnn_data = tt_output_torch[r, chip_id, token_id, topk_idx]

                    if torch.allclose(torch_data, ttnn_data, atol=1e-2, rtol=1e-2):
                        matches += 1
                    else:
                        data_ok = False
                        max_diff = torch.max(torch.abs(torch_data - ttnn_data)).item()
                        mismatches.append((r, chip_id, token_id, topk_idx, max_diff))

    # Report statistics
    logger.info(f"Matches: {matches}/{total_slots} ({100.0*matches/total_slots:.2f}%)")

    if not data_ok:
        # Show first 10 mismatches in detail
        logger.warning(f"Found {len(mismatches)} mismatches. Showing first 10:")
        for i, (r, chip_id, token_id, topk_idx, max_diff) in enumerate(mismatches[:10]):
            torch_sample = torch_output_per_rank[r][chip_id, token_id, topk_idx, :5]
            ttnn_sample = tt_output_torch[r, chip_id, token_id, topk_idx, :5]
            logger.error(
                f"  [{i}] Mismatch at r={r}, chip={chip_id}, token={token_id}, topk={topk_idx}: "
                f"max_diff={max_diff:.6f}"
            )
            logger.error(f"      torch[:5]={torch_sample}")
            logger.error(f"      ttnn[:5]={ttnn_sample}")

        # Show per-chip statistics
        logger.info("\nPer-chip statistics:")
        for r in range(num_chips_rep):
            for chip_id in range(num_chips_sp):
                chip_mismatches = [m for m in mismatches if m[0] == r and m[1] == chip_id]
                chip_total = seq_len_per_chip * num_experts_per_tok
                chip_matches = chip_total - len(chip_mismatches)
                logger.info(
                    f"  r={r} Chip {chip_id}: {chip_matches}/{chip_total} matches "
                    f"({100.0*chip_matches/chip_total:.2f}%)"
                )

    # Assert all data matches
    assert data_ok, f"Combine data mismatch! {matches}/{total_slots} slots matched. Check logs for details."

    logger.info("✅ TTNN combine operation matches torch reference!")
