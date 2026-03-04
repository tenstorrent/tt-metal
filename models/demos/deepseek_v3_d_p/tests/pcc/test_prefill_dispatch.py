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

# =====
# mesh 4x2
#
# ---------
# | X0  | X0 |
# | W0  | W0 |
# | I0  | I0 |
# ------------
# | X1  | X1 |
# | W1  | W1 |
# | I1  | I1 |
# ------------
# | X2  | X2 |
# | W2  | W2 |
# | I2  | I2 |
# ------------
# | X3  | X3 |
# | W3  | W3 |
# | I3  | I3 |
# ------------
#                   MeshDevice(rows=4, cols=2)
# ┌──────────────────────────────┬──────────────────────────────┐
# │          Dev. ID: 4          │          Dev. ID: 6          │
# │            (0, 0)            │            (0, 1)            │
# │       LinMeshCoord=0         │       LinMeshCoord=1         │
# │       LogicalCoord=0         |       LogicalCoord=0         │
# │                              │                              │
# ├──────────────────────────────┼──────────────────────────────┤
# │          Dev. ID: 2          │          Dev. ID: 3          │
# │            (1, 0)            │            (1, 1)            │
# │       LinMeshCoord=2         │       LinMeshCoord=3         │
# │       LogicalCoord=1         |       LogicalCoord=1         │
# │                              │                              │
# ├──────────────────────────────┼──────────────────────────────┤
# │          Dev. ID: 1          │          Dev. ID: 0          │
# │            (2, 0)            │            (2, 1)            │
# │       LinMeshCoord=4         │       LinMeshCoord=5         │
# │       LogicalCoord=2         |       LogicalCoord=2         │
# │                              │                              │
# ├──────────────────────────────┼──────────────────────────────┤
# │          Dev. ID: 5          │          Dev. ID: 7          │
# │            (3, 0)            │            (3, 1)            │
# │       LinMeshCoord=6         │       LinMeshCoord=7         │
# │       LogicalCoord=3         |       LogicalCoord=3         │
# │                              │                              │
# └──────────────────────────────┴──────────────────────────────┘
# Dev. ID is physical mapping
# LinMeshCoord is used for fabric transfers
# LogicalCoord is coordinate in withing a2a dispatch group


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, capacity_factor",
    [
        (32, 7168, 16, 4, 2),
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
@pytest.mark.parametrize("verbose", [True])
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
        f"Dispatch {mesh_device=} {num_devices=} {num_chips_sp=} {num_chips_rep=} {seq_len_per_chip=} {hidden_dim=} {n_routed_experts=} {num_experts_per_tok=} {capacity_factor=} {use_predictable_data=} {num_links=} {topology=}"
    )
    print("\n")

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips_sp, capacity_factor
    )
    logger.info(f"{experts_per_chip=}, {metadata_len=}, {max_dispatched_tokens_per_expert=}")

    # Initialize inputs using helper function
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

    # x and indices: replicated across EP ranks
    mesh_mapper_replicated = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(sp_axis, None),
    )

    # weights: sharded across BOTH axes (different per EP rank) for 2D mesh
    # weights shape: (num_ep_ranks, num_chips_sp, seq_len, num_experts_per_tok)
    # For 1D mesh (num_chips_rep=1): squeeze the EP rank dimension and use replicated mapper
    # For 2D mesh: shard both axes
    if num_chips_rep > 1:
        # For sp_axis=0: mesh axis 0 (rows) = num_chips_sp, mesh axis 1 (cols) = num_ep_ranks
        #   dims = (1, 0): shard tensor dim 1 (num_chips_sp) on mesh axis 0, tensor dim 0 (num_ep_ranks) on mesh axis 1
        # For sp_axis=1: mesh axis 0 (rows) = num_ep_ranks, mesh axis 1 (cols) = num_chips_sp
        #   dims = (0, 1): shard tensor dim 0 (num_ep_ranks) on mesh axis 0, tensor dim 1 (num_chips_sp) on mesh axis 1
        if sp_axis == 0:
            weights_dims = (1, 0)
        else:
            weights_dims = (0, 1)
        mesh_mapper_weights = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=weights_dims,
        )
        weights_for_ttnn = weights
    else:
        # For 1D mesh, squeeze the num_ep_ranks dimension since it's 1
        mesh_mapper_weights = mesh_mapper_replicated
        weights_for_ttnn = weights.squeeze(0)  # Remove the num_ep_ranks=1 dimension

    tt_x = ttnn.from_torch(
        x, mesh_mapper=mesh_mapper_replicated, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_weights = ttnn.from_torch(
        weights_for_ttnn,
        mesh_mapper=mesh_mapper_weights,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=mesh_mapper_replicated, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
    )

    logger.warning(f"{x.shape=}, {weights.shape=}, {indices.shape=}")
    logger.warning(f"{tt_x.shape=}, {tt_weights.shape=}, {tt_indices.shape=}")
    # ttnn.visualize_tensor(tt_x)
    # ttnn.visualize_tensor(tt_weights)
    # ttnn.visualize_tensor(tt_indices)

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
    )

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

    # Forward pass through TTNN dispatch
    logger.info(f"{x.shape=}")
    logger.info(f"{weights.shape=}")
    logger.info(f"{indices.shape=}")

    tt_dispatched, tt_metadata, counter, offsets, cum_sum = tt_dispatch_module(tt_x, tt_weights, tt_indices)

    # Run torch reference for all EP ranks at once
    torch_dispatched, torch_metadata, torch_counter = torch_dispatch_module(x, weights, indices)
    logger.info(f"Torch dispatch: {torch_dispatched.shape=}, {torch_metadata.shape=}")

    # Convert TTNN outputs to torch for comparison
    mesh_composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(
            dims=[1, 0],  # Axis 0: shard on tensor dim 0; Axis 1: replicated
        ),
    )
    logger.warning(f"{torch_dispatched[0].shape=} {torch_metadata[0].shape=}")
    logger.warning(f"{tt_dispatched.shape=} {tt_metadata.shape=}")
    tt_out_dispatched = ttnn.to_torch(tt_dispatched, mesh_composer=mesh_composer, dtype=torch.float32)
    tt_out_metadata = ttnn.to_torch(tt_metadata, mesh_composer=mesh_composer)
    logger.warning(f"{tt_out_dispatched.shape=} {tt_out_metadata.shape=}")

    assert (
        tt_out_dispatched.shape[0] == num_chips_rep
    ), f"Mismatch in replicated dimension: expected {num_chips_rep}, got {tt_out_dispatched.shape[0]}"
    assert (
        tt_out_dispatched.shape[1] == num_chips_sp
    ), f"Mismatch in sharded dimension: expected {num_chips_sp}, got {tt_out_dispatched.shape[1]}"

    # Quick sanity check of first elements
    logger.info(f"{tt_out_dispatched[0][0][0][0][0]=} | {tt_out_dispatched[0][1][0][0][0]=}")
    logger.info(f"{torch_dispatched[0][0][0][0][0]=} | {torch_dispatched[0][1][0][0][0]=}")
    logger.info(f"{tt_out_metadata[0][0][0][0][0:4]=} | {tt_out_metadata[0][1][0][0][0:4]=}")
    logger.info(f"{torch_metadata[0][0][0][0][0:4]=} | {torch_metadata[0][1][0][0][0:4]=}")
    logger.info(f"{counter.shape=}, {counter=}")
    logger.info(f"{offsets.shape=}, {offsets=}")
    logger.info(f"{cum_sum.shape=}, {cum_sum=}")

    # Verify dispatched data matches reference (each EP rank against its torch reference)
    data_ok = True
    metadata_ok = True
    logger.warning("Comparing ALL dispatched buffer slots (including remote dispatch)...")
    for r in range(num_chips_rep):
        dispatched = torch_dispatched[r]
        metadata = torch_metadata[r]
        for dst_chip_id in range(num_chips_sp):
            for expert_id in range(experts_per_chip):
                count = counter[dst_chip_id, expert_id].item()
                out = tt_out_dispatched[r, dst_chip_id, expert_id, :count, :]
                ref = dispatched[dst_chip_id, expert_id, :count, :]
                if torch.allclose(out, ref, atol=1e-6):
                    logger.info(f"✅ {r} Data {dst_chip_id=} {expert_id=} {count=}")
                else:
                    logger.error(f"❌ {r} Data {dst_chip_id=} {expert_id=} {count=}")
                    data_ok = False
                    if verbose:
                        for slot in range(count):
                            torch_data = dispatched[dst_chip_id, expert_id, slot]
                            kernel_data = tt_out_dispatched[r, dst_chip_id, expert_id, slot]
                            data_match = torch.allclose(torch_data, kernel_data, atol=1e-6)
                            if not data_match:
                                logger.error(
                                    f"    Slot {slot}: Data mismatch at chip={dst_chip_id}, expert={expert_id}, slot={slot}: "
                                    f"{torch_data=}, {kernel_data=}"
                                )

    logger.info("Comparing ALL dispatched metadata slots (including remote dispatch)...")
    for r in range(num_chips_rep):
        dispatched = torch_dispatched[r]
        metadata = torch_metadata[r]
        for dst_chip_id in range(num_chips_sp):
            for expert_id in range(experts_per_chip):
                count = counter[dst_chip_id, expert_id].item()
                out = tt_out_metadata[r, dst_chip_id, expert_id, :count, 1:4]
                ref = metadata[dst_chip_id, expert_id, :count, 1:4]

                # torch computes "logical sender chip id"
                # while ttnn embeds real linearized mesh coord
                out_linearized_mesh_coord = tt_out_metadata[r, dst_chip_id, expert_id, :count, 0]
                ref_linearized_mesh_coord = r + metadata[dst_chip_id, expert_id, :count, 0] * num_chips_rep

                # Compare weights (metadata[4]):
                # TTNN stores raw bfloat16 bits as uint16 in int32 - convert to bfloat16
                out_weight_bf16 = (
                    tt_out_metadata[r, dst_chip_id, expert_id, :count, 4].to(torch.int16).view(torch.bfloat16)
                )
                # Torch stores bfloat16 value directly
                ref_weight_bf16 = metadata[dst_chip_id, expert_id, :count, 4].to(torch.int16).view(torch.bfloat16)

                metadata_match = torch.allclose(out, ref, atol=1e-6)
                coord_match = torch.allclose(out_linearized_mesh_coord, ref_linearized_mesh_coord, atol=1e-6)
                weight_match = torch.allclose(out_weight_bf16, ref_weight_bf16, atol=1e-3)

                if metadata_match and coord_match and weight_match:
                    logger.info(f"✅ {r} Metadata {dst_chip_id=} {expert_id=} {count=}")
                else:
                    logger.error(
                        f"❌ {r} Metadata {dst_chip_id=} {expert_id=} {count=} ({metadata_match=}, {coord_match=}, {weight_match=})"
                    )
                    metadata_ok = False
                    if verbose:
                        for slot in range(count):
                            torch_data = metadata[dst_chip_id, expert_id, slot, :4]
                            kernel_data = tt_out_metadata[r, dst_chip_id, expert_id, slot, :4]
                            data_match = torch.allclose(torch_data, kernel_data, atol=1e-6)
                            if not data_match:
                                logger.error(
                                    f"    Slot {slot}: Metadata mismatch at chip={dst_chip_id}, expert={expert_id}, slot={slot}: "
                                    f"{ref_linearized_mesh_coord[slot].item()}, {out_linearized_mesh_coord[slot].item()}, "
                                    f"{torch_data=}, {kernel_data=}"
                                )
                            if not weight_match:
                                logger.error(
                                    f"    Slot {slot}: Weight mismatch: ref={ref_weight_bf16[slot].item()}, out={out_weight_bf16[slot].item()}"
                                )
    assert data_ok and metadata_ok, f"Some slots did not match! {data_ok=} {metadata_ok=} Check logs for details."
    logger.info("✅ TTNN dispatch operation matches torch reference!")
