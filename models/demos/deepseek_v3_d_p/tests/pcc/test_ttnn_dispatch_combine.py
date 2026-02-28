"""
Test for end-to-end TTNN MoE dispatch→combine round-trip.

This test verifies that tokens dispatched to experts using TTNN dispatch and then
combined back using TTNN combine produce the original input after host-side reduction,
validating the full round-trip through TTNN dispatch and combine operations.
"""

import pytest
import torch
from loguru import logger

import ttnn
from tracy import signpost

from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.common import (
    compute_constants,
    initialize_test_inputs,
    initialize_predictable_test_inputs,
    create_fabric_router_config,
)


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor",
    [
        (512, 7168, 16, 4, 2, 2),
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
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("use_predictable_data", [True, False], ids=["predictable", "random"])
def test_ttnn_dispatch_combine(
    mesh_device,
    seq_len_per_chip,
    hidden_dim,
    n_routed_experts,
    num_experts_per_tok,
    num_chips,
    capacity_factor,
    use_predictable_data,
):
    """Test end-to-end TTNN dispatch→combine round-trip with host reduction."""
    signpost(
        f"TTNN Dispatch+Combine {mesh_device=} {seq_len_per_chip=} {hidden_dim=} "
        f"{n_routed_experts=} {num_experts_per_tok=} {num_chips=} "
        f"{capacity_factor=} {use_predictable_data=}"
    )
    print("\n")

    # Step 1: Compute configuration constants
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor
    )
    logger.info(f"{experts_per_chip=}, {metadata_len=}, {max_dispatched_tokens_per_expert=}")

    # Step 2: Verify mesh device configuration
    num_devices = mesh_device.get_num_devices()
    assert num_chips == num_devices, f"num_chips {num_chips} must match number of devices in mesh {num_devices}"
    mesh_shape = mesh_device.shape
    logger.info(f"Testing with mesh_shape={mesh_shape}, num_devices={num_devices}")
    ttnn.visualize_mesh_device(mesh_device)

    # Step 3: Generate test inputs
    if use_predictable_data:
        x, weights, indices = initialize_predictable_test_inputs(
            num_chips=num_chips,
            seq_len_per_chip=seq_len_per_chip,
            hidden_dim=hidden_dim,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        )
        logger.info("Using PREDICTABLE test data for debugging")
    else:
        x, weights, indices = initialize_test_inputs(
            num_chips=num_chips,
            seq_len_per_chip=seq_len_per_chip,
            hidden_dim=hidden_dim,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            seed=42,
        )
        logger.info("Using RANDOM test data")

    logger.info(f"Input shapes: {x.shape=}, {weights.shape=}, {indices.shape=}")

    # Step 4: Convert torch tensors to TTNN tensors
    mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None),  # Shard on dim 0, replicate on dim 1
    )

    tt_x = ttnn.from_torch(
        x, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_weights = ttnn.from_torch(
        weights, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
    )

    # Step 5: Initialize TTNN dispatch module
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
    )

    # Step 6: Run TTNN dispatch
    logger.info("Running TTNN dispatch...")
    tt_dispatched_buffer, tt_metadata, experts_tok_counter, offsets, cum_sum = tt_dispatch_module(
        tt_x, tt_weights, tt_indices
    )
    ttnn.synchronize_device(mesh_device)
    logger.info("Dispatch complete!")

    logger.info(f"Dispatch outputs: {experts_tok_counter.shape=}")
    logger.info(f"  {experts_tok_counter=}")
    logger.info(f"  {offsets.shape=}, {offsets=}")
    logger.info(f"  {cum_sum.shape=}, {cum_sum=}")

    # Step 6b: Convert counter to TTNN tensor for combine module
    logger.info(f"Converting counter to TTNN: {experts_tok_counter.shape=}, {experts_tok_counter.dtype=}")
    logger.info(f"  Counter values: {experts_tok_counter=}")
    tt_experts_tok_counter = ttnn.from_torch(
        experts_tok_counter,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    logger.info(f"  TTNN counter shape: {tt_experts_tok_counter.shape}")
    ttnn.visualize_tensor(tt_experts_tok_counter, header="Experts Token Counter")

    # Step 7: Initialize TTNN combine module
    tt_combine_module = TtCombineModule(
        mesh_device=mesh_device,
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
    )

    # Step 8: Run TTNN combine
    logger.info("Running TTNN combine...")
    tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_experts_tok_counter)
    logger.info("Combine complete!")

    logger.info(f"Combine output shape: {tt_output.shape}")

    # Step 9: Convert TTNN output back to torch
    mesh_composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(
            dims=[0, 1],  # Axis 0: shard on tensor dim 0; Axis 1: replicated
        ),
    )

    y = ttnn.to_torch(tt_output, mesh_composer=mesh_composer, dtype=torch.bfloat16)

    # Step 10: Host-side reduction
    logger.info(f"Before reduction: {y.shape=}")
    y = y / num_experts_per_tok  # Average contributions from multiple experts
    y = y.sum(dim=2)  # Sum across expert dimension
    logger.info(f"After reduction: {y.shape=}")

    # Step 11: Verify round-trip correctness
    logger.info("Verifying round-trip correctness...")
    logger.info(f"Sample input x[0, 0, :5]: {x[0, 0, :5]}")
    logger.info(f"Sample output y[0, 0, :5]: {y[0, 0, :5]}")
    if num_chips > 1:
        logger.info(f"Sample input x[1, 0, :5]: {x[1, 0, :5]}")
        logger.info(f"Sample output y[1, 0, :5]: {y[1, 0, :5]}")

    max_diff = torch.max(torch.abs(x - y)).item()
    logger.info(f"Maximum absolute difference: {max_diff}")

    assert torch.allclose(
        x, y, atol=1e-6
    ), f"Expected output to match input, but got max diff {max_diff}"

    logger.info("✅ TTNN dispatch→combine round-trip matches input!")
