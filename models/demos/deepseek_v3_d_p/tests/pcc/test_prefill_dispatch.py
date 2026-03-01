"""
Test for TTNN MoE prefill dispatch operation in isolation.

This test verifies that the TTNN dispatch operation produces the same output as the
PyTorch reference implementation when dispatching tokens to experts.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.reference.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.common import (
    compute_constants,
    create_fabric_router_config,
    initialize_predictable_test_inputs,
    initialize_test_inputs,
)
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
        (
            (2, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
        ),
        (
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
        ),
        (
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
        ),
        (
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Ring,
        ),
    ],
    ids=["linear-2", "linear-4", "linear-8", "ring-8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("use_predictable_data", [True, False], ids=["predictable", "random"])
@pytest.mark.parametrize("verbose", [False])
def test_ttnn_dispatch(
    mesh_device,
    seq_len_per_chip,
    hidden_dim,
    n_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    num_links,
    topology,
    use_predictable_data,
    verbose,
):
    """Test TTNN dispatch operation against PyTorch reference."""
    num_devices = num_chips = mesh_device.get_num_devices()
    logger.info(f"Testing with mesh_shape={mesh_device.shape}, num_devices={num_devices}")
    ttnn.visualize_mesh_device(mesh_device)

    signpost(
        f"Dispatch {mesh_device=} {seq_len_per_chip=} {hidden_dim=} {n_routed_experts=} {num_experts_per_tok=} {num_chips=} {capacity_factor=} {use_predictable_data=} {num_links=} {topology=}"
    )
    print("\n")

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor
    )
    logger.info(f"{experts_per_chip=}, {metadata_len=}, {max_dispatched_tokens_per_expert=}")

    # Initialize inputs using helper function
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

    mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None),
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

    # Initialize dispatch modules
    dispatch_module = TorchDispatchModule(
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
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

    # Forward pass through dispatch modules
    logger.info(f"{x.shape=}")
    logger.info(f"{weights.shape=}")
    logger.info(f"{indices.shape=}")
    dispatched, metadata, experts_counter = dispatch_module(x, weights, indices)

    tt_dispatched, tt_metadata, counter, offsets, cum_sum = tt_dispatch_module(tt_x, tt_weights, tt_indices)

    # Convert TTNN outputs to torch for comparison
    mesh_composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(
            dims=[0, 1],  # Axis 0: shard on tensor dim 0; Axis 1: replicated
        ),
    )

    tt_out_dispatched = ttnn.to_torch(tt_dispatched, mesh_composer=mesh_composer, dtype=torch.float32)
    tt_out_metadata = ttnn.to_torch(tt_metadata, mesh_composer=mesh_composer)

    # Quick sanity check of first elements
    logger.info(f"{tt_out_dispatched[0][0][0][0]=} | {tt_out_dispatched[1][0][0][0]=}")
    logger.info(f"{dispatched[0][0][0][0]=} | {dispatched[1][0][0][0]=}")
    logger.info(f"{tt_out_metadata[0][0][0][0:4]=} | {tt_out_metadata[1][0][0][0:4]=}")
    logger.info(f"{metadata[0][0][0][0:4]=} | {metadata[1][0][0][0:4]=}")
    logger.info(f"{counter.shape=}, {counter=}")
    logger.info(f"{offsets.shape=}, {offsets=}")
    logger.info(f"{cum_sum.shape=}, {cum_sum=}")

    # Verify dispatched data matches reference
    data_ok = True
    metadata_ok = True
    logger.warning("Comparing ALL dispatched buffer slots (including remote dispatch)...")
    for dst_chip_id in range(num_chips):
        for expert_id in range(experts_per_chip):
            count = counter[dst_chip_id, expert_id].item()
            out = tt_out_dispatched[dst_chip_id, expert_id, :count, :]
            ref = dispatched[dst_chip_id, expert_id, :count, :]
            if torch.allclose(out, ref, atol=1e-6):
                logger.info(f"✅ Data {dst_chip_id=} {expert_id=} {count=}")
            else:
                logger.error(f"❌ Data {dst_chip_id=} {expert_id=} {count=}")
                data_ok = False
                if verbose:
                    for slot in range(count):
                        torch_data = dispatched[dst_chip_id, expert_id, slot]
                        kernel_data = tt_out_dispatched[dst_chip_id, expert_id, slot]
                        data_match = torch.allclose(torch_data, kernel_data, atol=1e-6)
                        if not data_match:
                            logger.error(
                                f"    Slot {slot}: Data mismatch at chip={dst_chip_id}, expert={expert_id}, slot={slot}: "
                                f"{torch_data=}, {kernel_data=}"
                            )

    logger.info("Comparing ALL dispatched metadata slots (including remote dispatch)...")
    for dst_chip_id in range(num_chips):
        for expert_id in range(experts_per_chip):
            count = counter[dst_chip_id, expert_id].item()
            out = tt_out_metadata[dst_chip_id, expert_id, :count, :4]
            ref = metadata[dst_chip_id, expert_id, :count, :4]
            if torch.allclose(out, ref, atol=1e-6):
                logger.info(f"✅ Metadata {dst_chip_id=} {expert_id=} {count=}")
            else:
                logger.error(f"❌ Metadata {dst_chip_id=} {expert_id=} {count=}")
                metadata_ok = False
                if verbose:
                    for slot in range(count):
                        torch_data = metadata[dst_chip_id, expert_id, slot, :4]
                        kernel_data = tt_out_metadata[dst_chip_id, expert_id, slot, :4]
                        data_match = torch.allclose(torch_data, kernel_data, atol=1e-6)
                        if not data_match:
                            logger.error(
                                f"    Slot {slot}: Metadata mismatch at chip={dst_chip_id}, expert={expert_id}, slot={slot}: "
                                f"{torch_data=}, {kernel_data=}"
                            )
    assert data_ok and metadata_ok, f"Some slots did not match! {data_ok=} {metadata_ok=} Check logs for details."
    logger.info("✅ TTNN dispatch operation matches torch reference!")
