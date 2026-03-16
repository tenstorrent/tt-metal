# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

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
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    compute_constants,
    create_expert_dispatch_table,
    create_fabric_router_config,
    extract_mesh_config,
    get_combine_counter_mesh_mapper,
    get_combine_output_mesh_composer,
    get_gate_outputs,
    initialize_predictable_test_inputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    assert_output_shape,
    log_combine_mismatch_details,
    log_per_chip_statistics,
    validate_combine_output,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_expert_dispatch_table, log_validation_results


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, num_routed_experts, num_experts_per_tok, capacity_factor",
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
def test_ttnn_combine(
    mesh_device,
    seq_len_per_chip,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    num_links,
    topology,
    use_predictable_data,
):
    """Test TTNN combine operation in isolation using torch reference inputs."""

    num_devices = mesh_device.get_num_devices()

    # Log fabric config
    logger.debug(f"Fabric max payload size: {ttnn.get_tt_fabric_max_payload_size_bytes()}")

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.debug(f"Testing with {mesh_device.shape=}, {num_devices=} {dispatch_group_size=} {num_dispatch_groups=}")
    ttnn.visualize_mesh_device(mesh_device)

    signpost(
        f"Combine {mesh_device=} {num_devices=} {dispatch_group_size=} {num_dispatch_groups=} {seq_len_per_chip=} {hidden_dim=} "
        f"{num_routed_experts=} {num_experts_per_tok=} {capacity_factor=} {use_predictable_data=} {num_links=} {topology=}"
    )

    # Compute configuration
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, num_routed_experts, num_experts_per_tok, num_devices, dispatch_group_size, capacity_factor
    )
    logger.debug(f"{experts_per_chip=}, {metadata_len=}, {max_dispatched_tokens_per_expert=}")

    # Step 1: Generate initial inputs using torch
    # For 2D mesh, generate different weights per EP rank
    if use_predictable_data:
        x, weights, indices = initialize_predictable_test_inputs(
            dispatch_group_size,
            seq_len_per_chip,
            hidden_dim,
            num_routed_experts,
            num_experts_per_tok,
            max_dispatched_tokens_per_expert,
            num_dispatch_groups=num_dispatch_groups,
        )
        logger.debug("Using PREDICTABLE test data for debugging")
    else:
        x, weights, indices = initialize_test_inputs(
            dispatch_group_size,
            seq_len_per_chip,
            hidden_dim,
            num_routed_experts,
            num_experts_per_tok,
            max_dispatched_tokens_per_expert,
            seed=42,
            num_dispatch_groups=num_dispatch_groups,
        )
        logger.debug("Using RANDOM test data")

    # Compute gate outputs before dispatch (same for all EP ranks since indices are shared)
    expert_offsets, expert_token_counts, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
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

    # Run dispatch for each EP rank with rank-specific weights
    dispatched_buffer, dispatched_metadata = torch_dispatch_module(x, weights, indices, expert_offsets)

    # Transform logical chip IDs to linearized coords
    # metadata[..., 0] contains the destination logical chip ID
    for r in range(num_dispatch_groups):
        # dest_linearized = dest_logical * num_dispatch_groups + replica_index
        dispatched_metadata[r, :, :, :, 0] = dispatched_metadata[r, :, :, :, 0] * num_dispatch_groups + r

    # Use different sharding: shard both dimensions
    mesh_mapper = get_combine_counter_mesh_mapper(mesh_device)

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

    tt_expert_token_counts = ttnn.from_torch(
        expert_token_counts,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    torch_combine = TorchCombineModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        num_dispatch_groups=num_dispatch_groups,
    )

    torch_output = torch_combine(dispatched_buffer, dispatched_metadata, expert_token_counts)

    # Step 5: Run ttnn combine
    tt_combine = TtCombineModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        init_zeros=True,
    )

    tt_output = tt_combine(
        tt_dispatched_buffer,
        tt_dispatched_metadata,
        tt_expert_token_counts,
    )

    # Step 6: Convert ttnn output to torch for comparison
    mesh_composer = get_combine_output_mesh_composer(mesh_device)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=mesh_composer,
        dtype=torch.bfloat16,
    )

    # Step 7: Verify correctness
    assert_output_shape(tt_output_torch, num_dispatch_groups, dispatch_group_size, "combine output")

    # Validate combine output (EP-rank aware)
    # NOTE: Current combine kernel does NOT all-reduce across EP ranks.
    # Each EP rank's output only contains data for tokens that EP rank processed.
    # Output positions not written by local combine contain uninitialized garbage.
    # This comparison only checks the EP rank that actually processed each token.
    result = validate_combine_output(
        torch_output,
        tt_output_torch,
        indices,
        num_dispatch_groups,
        num_routed_experts,
        verbose=True,
        expert_dispatch_table=expert_dispatch_table,
        expert_token_counts=expert_token_counts,
        experts_per_chip=experts_per_chip,
    )

    log_validation_results(
        results=[result],
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        title="Combine Validation Results",
    )

    if not result.passed:
        log_combine_mismatch_details(result.mismatches, torch_output, tt_output_torch)
        log_per_chip_statistics(result.mismatches, dispatch_group_size, seq_len_per_chip, num_experts_per_tok)

    result.assert_passed("Combine data mismatch")

    logger.debug("✅ TTNN combine operation matches torch reference!")
