# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PoC test for MoE prefill dispatch using subgroups on an 8x1 Blackhole LoudBox mesh.

Partitions the 8-chip mesh into two 4x1 dispatch subgroups. The dispatch op must see
only the 4 chips of its own subgroup — no fabric communication, no shared semaphores,
and no expert routing should cross the subgroup boundary.

Correctness strategy:
- Build an 8-chip input by tiling a 4-chip torch input twice along the device axis,
  so each subgroup receives an identical copy.
- Run a torch reference dispatch once over the 4-chip input.
- Assert that the TTNN output on chips 0..3 matches the torch reference, AND that the
  output on chips 4..7 independently matches the same torch reference.
"""

import pytest
import torch
from loguru import logger

import ttnn
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
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import validate_dispatch_buffer, validate_dispatch_metadata


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, capacity_factor",
    [
        (32, 7168, 16, 4, 2),
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
def test_ttnn_dispatch_subgroups(
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
    assert num_devices == dispatch_group_size * num_dispatch_subgroups, (
        f"mesh has {num_devices} devices but subgroups say "
        f"{dispatch_group_size} x {num_dispatch_subgroups} = {dispatch_group_size * num_dispatch_subgroups}"
    )
    num_dispatch_groups = 1  # 1D mesh: a single dispatch group per subgroup (no EP axis)
    sp_axis = 0

    logger.info(
        f"Subgroups test: mesh_shape={mesh_device.shape} num_devices={num_devices} "
        f"num_dispatch_subgroups={num_dispatch_subgroups} dispatch_group_size={dispatch_group_size}"
    )

    # We pass num_devices_for_constants == dispatch_group_size because the op's kernel sees
    # exactly `dispatch_group_size` chips within its subgroup — experts_per_chip is computed
    # against that local size.
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        dispatch_group_size,
        dispatch_group_size,
        capacity_factor,
    )

    # One 4-chip torch reference, replicated across both subgroups.
    x_single, weights_single, indices_single = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
    )

    # Tile across the device axis so each subgroup receives the same 4-chip pattern.
    x = x_single.repeat(num_dispatch_subgroups, 1, 1)
    weights = weights_single.repeat(num_dispatch_subgroups, 1, 1)
    indices = indices_single.repeat(num_dispatch_subgroups, 1, 1)

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    # Shard inputs along the device axis (rows) and replicate across cols (cols == 1 here).
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

    # Compute expert offsets from the single-subgroup indices (replicated across both subgroups).
    expert_offsets_single, expert_token_counts_single, _ = get_gate_outputs(
        indices_single,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    # expert_offsets shape: (num_dispatch_groups, dispatch_group_size, num_routed_experts) =>
    # tile along dim 1 so chips 0..3 and 4..7 get the same per-chip offset rows.
    expert_offsets = expert_offsets_single.repeat(1, num_dispatch_subgroups, 1)

    tt_expert_offsets = ttnn.from_torch(
        expert_offsets,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(1, 0)),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # expert_dispatch_table is (num_dispatch_groups, num_routed_experts); replicate across
    # mesh rows (dispatch axis) and shard along cols (== 1 here, so effectively replicated).
    tt_expert_dispatch_table = ttnn.from_torch(
        expert_dispatch_table,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(None, 0)),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

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
    torch_dispatched, torch_metadata = torch_dispatch_module(
        x_single, weights_single, indices_single, expert_offsets_single
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
        emb_dim=emb_dim,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        num_dispatch_subgroups=num_dispatch_subgroups,
    )
    tt_dispatched, tt_metadata = tt_dispatch_module(
        tt_x, tt_weights, tt_indices, tt_expert_offsets, tt_expert_dispatch_table
    )

    # Compose back using the same pattern the existing dispatch PCC test uses: mesh axis 0
    # (rows = chips) composes into tensor dim 1, so the output is (1, 8, ...) matching the
    # torch reference's (num_dispatch_groups, dispatch_group_size, ...) layout.
    mesh_composer = get_ep_mesh_composer(mesh_device)
    tt_out_dispatched = ttnn.to_torch(tt_dispatched, mesh_composer=mesh_composer, dtype=torch.float32)
    tt_out_metadata = ttnn.to_torch(tt_metadata, mesh_composer=mesh_composer)

    # Split the 8-chip output (along the chips axis = tensor dim 1) into two 4-chip slices.
    split_chunks_dispatched = torch.chunk(tt_out_dispatched, num_dispatch_subgroups, dim=1)
    split_chunks_metadata = torch.chunk(tt_out_metadata, num_dispatch_subgroups, dim=1)

    for sg_idx, (tt_sub_dispatched, tt_sub_metadata) in enumerate(zip(split_chunks_dispatched, split_chunks_metadata)):
        logger.info(f"Validating subgroup {sg_idx} (shape {tt_sub_dispatched.shape})")
        buffer_result = validate_dispatch_buffer(
            torch_dispatched,
            tt_sub_dispatched,
            expert_token_counts_single,
            expert_dispatch_table,
            num_dispatch_groups,
            dispatch_group_size,
            experts_per_chip,
            verbose=False,
        )
        metadata_result = validate_dispatch_metadata(
            torch_metadata,
            tt_sub_metadata,
            expert_token_counts_single,
            expert_dispatch_table,
            num_dispatch_groups,
            dispatch_group_size,
            experts_per_chip,
            verbose=False,
        )
        assert (
            buffer_result.passed and metadata_result.passed
        ), f"Subgroup {sg_idx} mismatch: buffer={buffer_result.passed} metadata={metadata_result.passed}"
        logger.info(f"✅ subgroup {sg_idx} matched the 4-chip torch reference")

    logger.info("✅ All dispatch subgroups matched independently")
