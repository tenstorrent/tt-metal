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

# from models.demos.deepseek_v3_d_p.reference.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    extract_mesh_config,
    get_gate_outputs,
    initialize_predictable_test_inputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.moe_gate_prefill2d import MoERoutingSetup

# from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_expert_dispatch_table


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
    )

    tt_gate_outputs = MoERoutingSetup(
        mesh_device=mesh_device, expert_dispatch_table=expert_dispatch_table, num_links=num_links
    )

    tt_expert_offsets, tt_expert_token_counts, tt_per_device_expert_counter = tt_gate_outputs(
        ttnn_top_k_experts_indices=tt_indices,
        dispatch_group_size=dispatch_group_size,
        num_routed_experts=num_routed_experts,
        experts_per_chip=experts_per_chip,
        seq_len_per_chip=seq_len_per_chip,
        num_experts_per_tok=num_experts_per_tok,
    )

    # tt_expert_offsets and tt_expert_token_counts are replicated across columns; and sparse across rows;
    # expert_offsets, and expert_token_counts on torch are dense acrros rows;

    # Compose across cols and rows
    # - Validate replication on cols
    # - Sparse -> dense on rows to validate everything
    tt_expert_offsets = ttnn.unsqueeze_to_4D(tt_expert_offsets)
    tt_expert_token_counts = ttnn.unsqueeze_to_4D(tt_expert_token_counts)

    composer = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(dims=[1, 0]))
    host_expert_offsets = ttnn.to_torch(tt_expert_offsets, mesh_composer=composer)
    host_expert_token_counts = ttnn.to_torch(tt_expert_token_counts, mesh_composer=composer)

    tensors_to_validate = [
        ("expert_offsets", host_expert_offsets, expert_offsets),
        ("expert_token_counts", host_expert_token_counts, expert_token_counts),
    ]

    # Validate replication within dispatch groups (across cols)
    for name, host_tensor, _ in tensors_to_validate:
        for i in range(num_dispatch_groups):
            ref = host_tensor[i][0]
            for j in range(1, dispatch_group_size):
                if not torch.allclose(host_tensor[i][j], ref, atol=0, rtol=0):
                    raise AssertionError(
                        f"Replication mismatch in {name} for dispatch group {i}, row {j}. "
                        f"Expected {ref}, got {host_tensor[i][j]}"
                    )

    # Validate sparse -> dense across dispatch groups (rows)
    for name, host_tensor, expected in tensors_to_validate:
        tt_dense = sum(host_tensor[i][0].int() for i in range(num_dispatch_groups))
        if not torch.allclose(tt_dense.flatten(), expected.int().flatten(), atol=0, rtol=0):
            raise AssertionError(f"Sparse->dense mismatch for {name}. Expected {expected}, got {tt_dense}")
