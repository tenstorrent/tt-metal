# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
PoC test: end-to-end TTNN dispatch -> TTNN combine on an 8x1 Blackhole LoudBox with
two 4-chip dispatch subgroups.

Unlike test_ttnn_moe_subgroups.py, this test does NOT go through TtMoe's gate path.
The offsets and token counts are pre-computed in torch with subgroup-local semantics
(via get_gate_outputs on the 4-chip input, tiled across subgroups). This isolates the
TTNN dispatch <-> TTNN combine pipeline from the gate's offset_cumsum op, which runs
on cluster_axis=0 and currently crosses subgroup boundaries.

If this test passes, it confirms that offset_cumsum is the blocker for the full
TtMoe.forward path; the fix is to add num_dispatch_subgroups to offset_cumsum using
the same pattern applied to dispatch and combine in Step 1.
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
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
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
def test_ttnn_dispatch_combine_subgroups(
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
        f"TTNN dispatch -> combine subgroups test: mesh={mesh_device.shape} "
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

    # 4-chip torch reference inputs.
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

    # Per-subgroup offsets + token counts (torch-computed, bypassing offset_cumsum).
    expert_offsets_single, expert_token_counts_single, _ = get_gate_outputs(
        indices_single,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    # Tile inputs to 8 chips (each subgroup sees the same 4-chip payload).
    x = x_single.repeat(num_dispatch_subgroups, 1, 1)
    weights = weights_single.repeat(num_dispatch_subgroups, 1, 1)
    indices = indices_single.repeat(num_dispatch_subgroups, 1, 1)
    expert_offsets = expert_offsets_single.repeat(1, num_dispatch_subgroups, 1)
    expert_token_counts = expert_token_counts_single.repeat(1, num_dispatch_subgroups, 1)

    input_shard_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(sp_axis, None))
    tt_x = ttnn.from_torch(
        x, mesh_mapper=input_shard_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_weights = ttnn.from_torch(
        weights, mesh_mapper=input_shard_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=input_shard_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
    )
    tt_expert_offsets = ttnn.from_torch(
        expert_offsets,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(1, 0)),
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
    tt_expert_dispatch_table = ttnn.from_torch(
        expert_dispatch_table,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(None, 0)),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # Run TTNN dispatch with subgroup scoping.
    tt_dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        num_dispatch_subgroups=num_dispatch_subgroups,
    )
    tt_dispatched_buffer, tt_dispatched_metadata = tt_dispatch_module(
        tt_x, tt_weights, tt_indices, tt_expert_offsets, tt_expert_dispatch_table
    )

    # Run TTNN combine on the TTNN dispatch outputs (the path that hangs in TtMoe.forward).
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

    # Torch reference: run torch dispatch + torch combine on the single-subgroup input.
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

    # Compose 8-chip TTNN output and validate each subgroup against the 4-chip torch ref.
    mesh_composer = get_ep_mesh_composer(mesh_device)
    tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer, dtype=torch.bfloat16)
    logger.info(f"tt_output_torch shape: {tt_output_torch.shape}")

    slices = torch.chunk(tt_output_torch, num_dispatch_subgroups, dim=1)
    for sg_idx, tt_sub_output in enumerate(slices):
        logger.info(f"Validating subgroup {sg_idx} (shape {tt_sub_output.shape})")
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

    logger.info("✅ TTNN dispatch -> TTNN combine subgroups path verified")
