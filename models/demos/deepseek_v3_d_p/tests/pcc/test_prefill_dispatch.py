# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

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
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    compute_constants,
    create_expert_dispatch_table,
    create_fabric_router_config,
    extract_mesh_config,
    get_combine_output_mesh_composer,
    get_dispatch_input_mesh_mapper,
    get_gate_outputs,
    initialize_predictable_test_inputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    assert_output_shape,
    validate_dispatch_buffer,
    validate_dispatch_metadata,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_expert_dispatch_table, log_validation_results

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
    "seq_len_per_chip, hidden_dim, num_routed_experts, num_experts_per_tok, capacity_factor",
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
            id="linear-2-1link",
        ),
        pytest.param(
            (2, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 1), topology="linear"),
            id="linear-2-2link",
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
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Ring,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="ring"),
            id="ring-4-1link",
        ),
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Ring,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="ring"),
            id="ring-4-2link",
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
            id="linear-8-1link",
        ),
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="linear-8-2link",
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
            id="ring-8-1link",
        ),
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Ring,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="ring"),
            id="ring-8-2link",
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
@pytest.mark.parametrize("verbose", [False])
def test_ttnn_dispatch(
    mesh_device,
    seq_len_per_chip,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    num_links,
    topology,
    use_predictable_data,
    verbose,
):
    """Test TTNN dispatch operation against PyTorch reference."""
    num_devices = mesh_device.get_num_devices()

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.info(f"Testing with {mesh_device.shape=}, {num_devices=} {dispatch_group_size=} {num_dispatch_groups=}")
    ttnn.visualize_mesh_device(mesh_device)

    signpost(
        f"Dispatch {mesh_device=} {num_devices=} {dispatch_group_size=} {num_dispatch_groups=} {seq_len_per_chip=} {hidden_dim=} {num_routed_experts=} {num_experts_per_tok=} {capacity_factor=} {use_predictable_data=} {num_links=} {topology=}"
    )

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, num_routed_experts, num_experts_per_tok, num_devices, capacity_factor
    )
    logger.info(f"{experts_per_chip=}, {metadata_len=}, {max_dispatched_tokens_per_expert=}")

    # Initialize inputs using helper function
    # For 2D mesh, generate different weights per EP rank
    if use_predictable_data:
        x, weights, indices = initialize_predictable_test_inputs(
            dispatch_group_size=dispatch_group_size,
            seq_len_per_chip=seq_len_per_chip,
            hidden_dim=hidden_dim,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            num_dispatch_groups=num_dispatch_groups,
        )
        logger.info("Using PREDICTABLE test data for debugging")
    else:
        x, weights, indices = initialize_test_inputs(
            dispatch_group_size=dispatch_group_size,
            seq_len_per_chip=seq_len_per_chip,
            hidden_dim=hidden_dim,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            seed=42,
            num_dispatch_groups=num_dispatch_groups,
        )
        logger.info("Using RANDOM test data")

    logger.debug(f"Input shapes: {x.shape=}, {weights.shape=}, {indices.shape=}")

    # x and indices: sharded across SP axis, replicated across EP ranks
    mesh_mapper_replicated = get_dispatch_input_mesh_mapper(mesh_device, sp_axis)

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

    # Create expert dispatch table
    expert_dispatch_table = create_expert_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    log_expert_dispatch_table(
        expert_dispatch_table=expert_dispatch_table,
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        num_routed_experts=num_routed_experts,
    )

    # Initialize torch dispatch module with num_dispatch_groups support
    torch_dispatch_module = TorchDispatchModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        num_dispatch_groups=num_dispatch_groups,
        expert_dispatch_table=expert_dispatch_table,
    )

    tt_dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
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
    expert_offsets, expert_token_counts, cum_sum = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
    )

    # Forward pass through TTNN dispatch
    tt_expert_offsets = TtDispatchModule.shard_expert_offsets(mesh_device, expert_offsets)
    tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)

    tt_dispatched, tt_metadata = tt_dispatch_module(
        tt_x, tt_weights, tt_indices, tt_expert_offsets, tt_expert_dispatch_table
    )

    # Run torch reference for all EP ranks at once
    torch_dispatched, torch_metadata = torch_dispatch_module(x, weights, indices, expert_offsets)

    # Convert TTNN outputs to torch for comparison
    mesh_composer = get_combine_output_mesh_composer(mesh_device)
    tt_out_dispatched = ttnn.to_torch(tt_dispatched, mesh_composer=mesh_composer, dtype=torch.float32)
    tt_out_metadata = ttnn.to_torch(tt_metadata, mesh_composer=mesh_composer)

    assert_output_shape(tt_out_dispatched, num_dispatch_groups, dispatch_group_size, "dispatched buffer")

    # Quick sanity check of first elements (verbose mode only)
    if verbose:
        logger.debug(f"{tt_out_dispatched[0][0][0][0][0]=} | {tt_out_dispatched[0][1][0][0][0]=}")
        logger.debug(f"{torch_dispatched[0][0][0][0][0]=} | {torch_dispatched[0][1][0][0][0]=}")
        logger.debug(f"{tt_out_metadata[0][0][0][0][0:4]=} | {tt_out_metadata[0][1][0][0][0:4]=}")
        logger.debug(f"{torch_metadata[0][0][0][0][0:4]=} | {torch_metadata[0][1][0][0][0:4]=}")
        logger.debug(f"{expert_token_counts.shape=}, {expert_token_counts=}")
        logger.debug(f"{expert_offsets.shape=}, {expert_offsets=}")
        logger.debug(f"{cum_sum.shape=}, {cum_sum=}")

    # Verify dispatched data matches reference (each EP rank against its torch reference)
    buffer_result = validate_dispatch_buffer(
        torch_dispatched,
        tt_out_dispatched,
        expert_token_counts,
        expert_dispatch_table,
        num_dispatch_groups,
        dispatch_group_size,
        experts_per_chip,
        verbose=verbose,
    )

    metadata_result = validate_dispatch_metadata(
        torch_metadata,
        tt_out_metadata,
        expert_token_counts,
        expert_dispatch_table,
        num_dispatch_groups,
        dispatch_group_size,
        experts_per_chip,
        verbose=verbose,
    )

    # Log summaries and visualization
    log_validation_results(
        results=[buffer_result, metadata_result],
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        title="Dispatch Validation Results",
    )

    assert (
        buffer_result.passed and metadata_result.passed
    ), f"Some slots did not match! buffer={buffer_result.passed} metadata={metadata_result.passed} Check logs for details."
    logger.info("✅ TTNN dispatch operation matches torch reference!")
