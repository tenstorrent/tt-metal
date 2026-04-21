# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PoC test for MoE prefill combine using subgroups on an 8x1 Blackhole LoudBox mesh.

Mirrors test_prefill_dispatch_subgroups.py but for the combine op: the 8-chip mesh is
partitioned into two 4x1 subgroups. Each subgroup runs combine independently and must
produce, per chip, the same output as a standalone 4-chip combine would produce on the
tiled input.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    get_ep_mesh_composer,
    get_gate_outputs,
    get_max_payload_size,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import validate_combine_output


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, capacity_factor",
    [
        (128, 7 * 1024, 16, 4, 2),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology, num_dispatch_subgroups, dispatch_group_size",
    [
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            },
            1,
            ttnn.Topology.Linear,
            2,
            4,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="subgroups-2x4-linear-1link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_ttnn_combine_subgroups(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    num_links,
    topology,
    num_dispatch_subgroups,
    dispatch_group_size,
):
    torch.manual_seed(42)

    num_devices = mesh_device.get_num_devices()
    assert num_devices == dispatch_group_size * num_dispatch_subgroups
    num_dispatch_groups = 1
    sp_axis = 0

    logger.info(
        f"Combine subgroups test: mesh_shape={mesh_device.shape} "
        f"num_dispatch_subgroups={num_dispatch_subgroups} dispatch_group_size={dispatch_group_size}"
    )

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        dispatch_group_size,
        dispatch_group_size,
        capacity_factor,
    )

    # Single 4-chip torch reference; replicated across both subgroups.
    x_single, weights_single, indices_single = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
    )

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    expert_offsets_single, expert_token_counts_single, _ = get_gate_outputs(
        indices_single,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    # Torch dispatch runs once on the 4-chip reference to produce the dispatched_buffer
    # and metadata inputs to combine.
    torch_dispatch_module = TorchDispatchModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_dispatch_groups=num_dispatch_groups,
        expert_dispatch_table=expert_dispatch_table,
    )
    dispatched_buffer_single, dispatched_metadata_single = torch_dispatch_module(
        x_single, weights_single, indices_single, expert_offsets_single
    )

    # Tile the dispatched_buffer/metadata across the device axis so each subgroup receives
    # an identical 4-chip payload. Shapes: (num_dispatch_groups, dispatch_group_size,
    # experts_per_chip, max_tokens, emb_dim) => expand dim 1 from 4 to 8.
    dispatched_buffer = dispatched_buffer_single.repeat(1, num_dispatch_subgroups, 1, 1, 1)
    dispatched_metadata = dispatched_metadata_single.repeat(1, num_dispatch_subgroups, 1, 1, 1)

    expert_token_counts = expert_token_counts_single.repeat(1, num_dispatch_subgroups, 1)

    mesh_mapper_ep = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(1, 0))
    tt_dispatched_buffer = ttnn.from_torch(
        dispatched_buffer,
        mesh_mapper=mesh_mapper_ep,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_dispatched_metadata = ttnn.from_torch(
        dispatched_metadata,
        mesh_mapper=mesh_mapper_ep,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    tt_expert_token_counts = ttnn.from_torch(
        expert_token_counts,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(1, 0)),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    torch_combine_module = TorchCombineModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        num_dispatch_groups=num_dispatch_groups,
    )
    torch_output = torch_combine_module(
        dispatched_buffer_single, dispatched_metadata_single, expert_token_counts_single
    )

    tt_combine_module = TtCombineModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        init_zeros=True,
        num_dispatch_subgroups=num_dispatch_subgroups,
    )
    tt_output = tt_combine_module(tt_dispatched_buffer, tt_dispatched_metadata, tt_expert_token_counts)

    # Same composer pattern as the existing combine PCC test — output shape becomes
    # (num_dispatch_groups=1, dispatch_group_size_total=8, seq_len_per_chip, num_experts_per_tok, emb_dim).
    mesh_composer = get_ep_mesh_composer(mesh_device)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer, dtype=torch.bfloat16)

    # Split along the chips axis (tensor dim 1) into per-subgroup slices of shape
    # (1, 4, seq_len_per_chip, num_experts_per_tok, emb_dim) — matching the 4-chip torch reference.
    split_chunks = torch.chunk(tt_output_torch, num_dispatch_subgroups, dim=1)
    for sg_idx, tt_sub_output in enumerate(split_chunks):
        logger.info(f"Validating combine subgroup {sg_idx} (shape {tt_sub_output.shape})")
        result = validate_combine_output(
            torch_output,
            tt_sub_output,
            indices_single,
            num_dispatch_groups,
            num_routed_experts,
            verbose=False,
            expert_dispatch_table=expert_dispatch_table,
            expert_token_counts=expert_token_counts_single,
            experts_per_chip=experts_per_chip,
        )
        result.assert_passed(f"Subgroup {sg_idx} combine data mismatch")
        logger.info(f"✅ subgroup {sg_idx} matched the 4-chip torch reference")

    logger.info("✅ All combine subgroups matched independently")
