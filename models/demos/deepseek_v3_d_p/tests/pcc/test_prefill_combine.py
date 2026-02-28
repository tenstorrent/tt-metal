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
    initialize_predictable_test_inputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, capacity_factor",
    [
        (512, 7 * 1024, 16, 4, 2),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
        },
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (2, 1),  # SP=2, TP=1
        (4, 1),  # SP=4, TP=1
        (8, 1),  # SP=8, TP=1
    ],
    ids=["linear-2", "linear-4", "linear-8"],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("use_predictable_data", [True, False], ids=["predictable", "random"])
def test_ttnn_combine(
    mesh_device,
    seq_len_per_chip,
    hidden_dim,
    n_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    use_predictable_data,
):
    """Test TTNN combine operation in isolation using torch reference inputs."""

    num_devices = num_chips = mesh_device.get_num_devices()
    logger.info(f"Testing with mesh_shape={mesh_device.shape}, num_devices={num_devices}")
    ttnn.visualize_mesh_device(mesh_device)

    signpost(
        f"Combine {mesh_device=} {seq_len_per_chip=} {hidden_dim=} "
        f"{n_routed_experts=} {num_experts_per_tok=} {num_chips=} "
        f"{capacity_factor=} {use_predictable_data=}"
    )
    print("\n")

    # Compute configuration
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor
    )
    logger.info(f"{experts_per_chip=}, {metadata_len=}, {max_dispatched_tokens_per_expert=}")

    # Step 1: Generate initial inputs using torch
    if use_predictable_data:
        x, weights, indices = initialize_predictable_test_inputs(
            num_chips,
            seq_len_per_chip,
            hidden_dim,
            n_routed_experts,
            num_experts_per_tok,
            max_dispatched_tokens_per_expert,
        )
        logger.info("Using PREDICTABLE test data for debugging")
    else:
        x, weights, indices = initialize_test_inputs(
            num_chips,
            seq_len_per_chip,
            hidden_dim,
            n_routed_experts,
            num_experts_per_tok,
            max_dispatched_tokens_per_expert,
            seed=42,
        )
        logger.info("Using RANDOM test data")

    # Step 2: Run torch dispatch to generate combine inputs
    torch_dispatch = TorchDispatchModule(
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
    )

    dispatched_buffer, dispatched_metadata, experts_tok_counter = torch_dispatch(x, weights, indices)

    logger.info("Torch dispatch outputs:")
    logger.info(f"  {dispatched_buffer.shape=}")
    logger.info(f"  {dispatched_metadata.shape=}")
    logger.info(f"  {experts_tok_counter.shape=}")

    # Step 3: Convert torch tensors to ttnn tensors
    mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None),  # Shard on dim 0, replicate on dim 1
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

    # Step 4: Run torch combine for reference output
    torch_combine = TorchCombineModule(
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
    )

    torch_output = torch_combine(
        dispatched_buffer,
        dispatched_metadata,
        experts_tok_counter,
    )

    logger.info(f"Torch combine output shape: {torch_output.shape}")

    # Step 5: Run ttnn combine
    tt_combine = TtCombineModule(
        mesh_device=mesh_device,
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
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
            dims=[0, 1],  # Axis 0: shard on tensor dim 0; Axis 1: replicated
        ),
    )

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=mesh_composer,
        dtype=torch.bfloat16,
    )

    # Step 7: Compute PCC and verify correctness
    logger.info("Computing PCC between torch and ttnn combine outputs...")

    # Quick sanity check of first elements
    logger.info(f"Sample torch output [0, 0, 0, :5]: {torch_output[0, 0, 0, :5]}")
    logger.info(f"Sample ttnn output [0, 0, 0, :5]:  {tt_output_torch[0, 0, 0, :5]}")
    if num_chips > 1:
        logger.info(f"Sample torch output [1, 0, 0, :5]: {torch_output[1, 0, 0, :5]}")
        logger.info(f"Sample ttnn output [1, 0, 0, :5]:  {tt_output_torch[1, 0, 0, :5]}")

    # Detailed per-chip, per-token, per-expert comparison
    data_ok = True
    mismatches = []
    matches = 0
    total_slots = 0

    logger.info("Comparing ALL combine output slots...")
    for chip_id in range(num_chips):
        for token_id in range(seq_len_per_chip):
            for topk_idx in range(num_experts_per_tok):
                total_slots += 1
                torch_data = torch_output[chip_id, token_id, topk_idx]
                ttnn_data = tt_output_torch[chip_id, token_id, topk_idx]

                if torch.allclose(torch_data, ttnn_data, atol=1e-2, rtol=1e-2):
                    matches += 1
                else:
                    data_ok = False
                    max_diff = torch.max(torch.abs(torch_data - ttnn_data)).item()
                    mismatches.append((chip_id, token_id, topk_idx, max_diff))

    # Report statistics
    logger.info(f"Matches: {matches}/{total_slots} ({100.0*matches/total_slots:.2f}%)")

    if not data_ok:
        # Show first 10 mismatches in detail
        logger.warning(f"Found {len(mismatches)} mismatches. Showing first 10:")
        for i, (chip_id, token_id, topk_idx, max_diff) in enumerate(mismatches[:10]):
            torch_sample = torch_output[chip_id, token_id, topk_idx, :5]
            ttnn_sample = tt_output_torch[chip_id, token_id, topk_idx, :5]
            logger.error(
                f"  [{i}] Mismatch at chip={chip_id}, token={token_id}, topk={topk_idx}: " f"max_diff={max_diff:.6f}"
            )
            logger.error(f"      torch[:5]={torch_sample}")
            logger.error(f"      ttnn[:5]={ttnn_sample}")

        # Show per-chip statistics
        logger.info("\nPer-chip statistics:")
        for chip_id in range(num_chips):
            chip_mismatches = [m for m in mismatches if m[0] == chip_id]
            chip_total = seq_len_per_chip * num_experts_per_tok
            chip_matches = chip_total - len(chip_mismatches)
            logger.info(
                f"  Chip {chip_id}: {chip_matches}/{chip_total} matches " f"({100.0*chip_matches/chip_total:.2f}%)"
            )

    # Assert all data matches
    assert data_ok, f"Combine data mismatch! {matches}/{total_slots} slots matched. Check logs for details."

    logger.info("✅ TTNN combine operation matches torch reference!")
