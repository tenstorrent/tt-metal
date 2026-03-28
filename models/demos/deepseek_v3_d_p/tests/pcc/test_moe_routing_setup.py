# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for TTNN MoE prefill dispatch operation in isolation.

This test verifies that the TTNN dispatch operation produces the same output as the
PyTorch reference implementation when dispatching tokens to experts.
"""

import pytest
from loguru import logger
from tracy import signpost

import ttnn

# from models.demos.deepseek_v3_d_p.reference.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_gate_outputs,
    initialize_predictable_test_inputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup

# from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    compare_exact,
    validate_composed,
    validate_replication,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_expert_dispatch_table, log_validation_results


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, capacity_factor",
    [
        (3200, 7168, 64, 2, 2),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
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
def test_prep_dispatch_combine(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    num_links,
    topology,
    use_predictable_data,
):
    """Test TTNN dispatch operation against PyTorch reference."""
    num_devices = mesh_device.get_num_devices()

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.debug(f"Testing with {mesh_device.shape=}, {num_devices=} {dispatch_group_size=} {num_dispatch_groups=}")
    ttnn.visualize_mesh_device(mesh_device)

    signpost(
        f"prep dispatch/combine {mesh_device=} {num_devices=} {dispatch_group_size=} {num_dispatch_groups=} {seq_len_per_chip=} {emb_dim=} {num_routed_experts=} {num_experts_per_tok=} {capacity_factor=} {use_predictable_data=} {num_links=} {topology=}"
    )

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, num_routed_experts, num_experts_per_tok, num_devices, dispatch_group_size, capacity_factor
    )

    logger.debug(f"{experts_per_chip=}, {metadata_len=}, {max_dispatched_tokens_per_expert=}")

    # Initialize inputs using helper function
    # For 2D mesh, generate different weights per EP rank
    if use_predictable_data:
        x, weights, indices = initialize_predictable_test_inputs(
            dispatch_group_size=dispatch_group_size,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            num_dispatch_groups=num_dispatch_groups,
        )
        logger.debug("Using PREDICTABLE test data for debugging")
    else:
        x, weights, indices = initialize_test_inputs(
            dispatch_group_size=dispatch_group_size,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            seed=42,
            num_dispatch_groups=num_dispatch_groups,
        )
        logger.debug("Using RANDOM test data")

    logger.debug(f"Input shapes: {x.shape=}, {weights.shape=}, {indices.shape=}")

    # x and indices: replicated across EP ranks
    mesh_mapper_replicated = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(sp_axis, None),
    )

    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=mesh_mapper_replicated, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.uint16
    )

    # Create expert dispatch table
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
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

    # Compute gate outputs (offsets and token counts) before dispatch
    expert_offsets, expert_token_counts, per_device_expert_counter = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    tt_gate_outputs = TtMoERoutingSetup(
        mesh_device=mesh_device, expert_dispatch_table=expert_dispatch_table, num_links=num_links
    )

    tt_expert_offsets, tt_expert_token_counts, tt_per_device_expert_counter = tt_gate_outputs(
        ttnn_top_k_experts_indices=tt_indices,
        num_routed_experts=num_routed_experts,
        seq_len_per_chip=seq_len_per_chip,
        num_experts_per_tok=num_experts_per_tok,
    )

    # Both TTNN and reference outputs are now in sparse format:
    # Shape: (num_dispatch_groups, dispatch_group_size, num_routed_experts)
    # Compose across cols (dispatch groups) and rows (dispatch_group_size)
    tt_expert_offsets = ttnn.unsqueeze_to_4D(tt_expert_offsets)
    tt_expert_token_counts = ttnn.unsqueeze_to_4D(tt_expert_token_counts)

    ep_composer = get_ep_mesh_composer(mesh_device)
    # squeeze(2) removes only the singleton dim from unsqueeze_to_4D, preserving [groups, chips, experts]
    host_expert_offsets = ttnn.to_torch(tt_expert_offsets, mesh_composer=ep_composer).squeeze(2)
    host_expert_token_counts = ttnn.to_torch(tt_expert_token_counts, mesh_composer=ep_composer).squeeze(2)

    # Validate replication of expert_token_counts within dispatch groups (all chips see same totals)
    replication_result = validate_replication(host_expert_token_counts, name="counts_replication")

    # Validate values match torch reference
    offsets_result = validate_composed(
        host_expert_offsets.int(),
        expert_offsets.int(),
        num_dispatch_groups,
        dispatch_group_size,
        compare_exact,
        name="expert_offsets",
    )
    counts_result = validate_composed(
        host_expert_token_counts.int(),
        expert_token_counts.int(),
        num_dispatch_groups,
        dispatch_group_size,
        compare_exact,
        name="expert_token_counts",
    )

    log_validation_results(
        results=[offsets_result, counts_result],
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        title="Routing Setup Validation",
    )

    for r in [replication_result, offsets_result, counts_result]:
        r.assert_passed(f"{r.name} validation failed")
