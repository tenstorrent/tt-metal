# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC test for combine_fabric2d overlapped INSIDE hybrid_routed_expert_ffn.

The two-op forward runs the routed expert, waits for it, then runs combine. This drives the
merged op instead: one dispatch, the routed expert on rows 2..9 and combine on rows 0..1 of the
same grid, with a per-expert semaphore handoff standing in for the dispatch boundary. What is
being graded is that handoff -- if it lets combine read an expert's rows early, the output is
wrong in exactly the places the routed expert had not finished writing.

The reference is therefore the WHOLE chain, torch dispatch -> torch expert -> torch combine, not
combine alone. Grading only the routed-expert output would pass with the gate removed entirely.

OPT-IN. Every case is skipped unless TT_RUN_COMBINE_OVERLAP is set, because this folder is run
unfiltered by the bh_p150 and bh_p300 OP_TESTS legs and the failure mode of a wrong handoff is a
hung grid, not a failed assert.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import fabric_to_device_params
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_ep_mesh_mapper,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    assert_output_shape,
    log_combine_mismatch_details,
    validate_combine_output,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

pytestmark = pytest.mark.skipif(
    os.environ.get("TT_RUN_COMBINE_OVERLAP") is None,
    reason="opt-in: a wrong expert-ready handoff hangs the grid, and OP_TESTS runs this folder unfiltered",
)

# Non-zero, so BOTH halves are carried: experts at or below this land on the fused half, the rest
# on the unified one. 320 is what the models that carry a threshold use. Two passes is also the
# only configuration that exercises PRIOR in the expert-ready counter -- with one pass PRIOR is 0
# and the term is never read.
_TOKEN_THRESHOLD = 320

# The proxy meshes an 8-card box can actually open, from the combine test's target_meshes. The
# 8x4 full-model mesh is deliberately absent: it needs 32 chips.
# Two links per direction, matching the full-model case. On this mesh both links reach distinct
# workers, so nothing here drives the relocation combine's placement falls back on.
_NUM_LINKS = 2

# Combine drives its dispatch axis as a ring in both directions, so only a fabric that wraps that
# axis can carry it. A 4x2 FABRIC_2D mesh leaves axis 0 linear and is rejected outright.
_PROXY_MESHES = (((8, 1), ttnn.FabricConfig.FABRIC_2D_TORUS_Y, "ring"),)


# (seq_len_per_chip, dispatch_buffer_capacity_factor). 128 is a quick shape check; 640 is where
# the handoff is actually under load -- more tokens per expert means the routed-expert half spends
# longer per expert, which is what lets combine arrive first and makes the gate load-bearing.
_SCENARIOS = ((128, 4), (640, 8))


def _proxy_params():
    params = []
    for mesh, fabric_cfg, topo in _PROXY_MESHES:
        num_chips = mesh[0] * mesh[1]
        model = DeepSeekV3Config()
        # Scaled down to keep the case short, but a multiple of both the chip count (so each chip
        # hosts a whole number of experts) and of 16. The 16 is combine's: its untilizer reads the
        # control tables one expert-row of num_routed_experts*4 bytes at a time, straight into L1
        # at row_bytes strides, and a DRAM read needs a 64-byte-aligned L1 destination. At 8
        # experts the rows are 32 bytes and every odd one is misaligned.
        num_routed_experts = int(os.environ.get("TT_OVERLAP_EXPERTS", "32"))
        assert num_routed_experts % num_chips == 0 and num_routed_experts % 16 == 0
        for seq_len_per_chip, capacity_factor in _SCENARIOS:
            params.append(
                pytest.param(
                    mesh,
                    fabric_to_device_params(fabric_cfg),
                    per_axis_topology(fabric_cfg)[0],
                    seq_len_per_chip,
                    model.EMB_SIZE,
                    num_routed_experts,
                    max(2, model.NUM_EXPERTS_PER_TOKEN // 4),
                    capacity_factor,
                    marks=pytest.mark.requires_mesh_topology(mesh_shape=mesh, topology=topo),
                    id=f"dsv3-overlap-{topo}-{mesh[0]}x{mesh[1]}-seq{seq_len_per_chip}",
                )
            )
    return params


def _torch_routed_expert(
    dispatched_buffer,
    expert_token_counts,
    expert_region_offsets,
    torch_weights,
    experts_per_chip,
    emb_dim,
    hidden_dim,
):
    """The routed-expert forward in the dispatched buffer's own (group, chip, token, emb) layout.

    Rows outside an expert's active count are left as they came in: the device op does not write
    them either, and torch combine never reads them, because the same counts bound its walk.

    Which global expert sits at (group, chip, local) comes from ExpertMapping rather than from
    arithmetic here -- it is the same function TtRoutedExpert shards the weights with, so the
    reference and the device cannot disagree about whose weights an expert has.
    """
    out = dispatched_buffer.clone()
    num_dispatch_groups, dispatch_group_size = dispatched_buffer.shape[0], dispatched_buffer.shape[1]
    for group in range(num_dispatch_groups):
        for chip in range(dispatch_group_size):
            for local in range(experts_per_chip):
                global_expert = ExpertMapping.get_global_expert_idx(
                    group=group,
                    chip=chip,
                    local_expert=local,
                    experts_per_chip=experts_per_chip,
                    dispatch_group_size=dispatch_group_size,
                    num_dispatch_groups=num_dispatch_groups,
                )
                count = int(expert_token_counts[group, chip, global_expert])
                if count == 0:
                    continue
                start = int(expert_region_offsets[group, chip, global_expert])
                expert = TorchExpert(emb_dim, hidden_dim, torch_weights=torch_weights[global_expert])
                out[group, chip, start : start + count] = expert(dispatched_buffer[group, chip, start : start + count])
    return out


@pytest.mark.parametrize(
    "mesh_device, device_params, topology, seq_len_per_chip, emb_dim, num_routed_experts, "
    "num_experts_per_tok, dispatch_buffer_capacity_factor",
    _proxy_params(),
    indirect=["mesh_device", "device_params"],
)
def test_combine_overlapped_in_routed_expert(
    mesh_device,
    device_params,
    topology,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
):
    torch.manual_seed(42)

    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices,
        dispatch_group_size,
        dispatch_buffer_capacity_factor,
    )
    hidden_dim = DeepSeekV3Config().MOE_INTERMEDIATE_SIZE
    logger.debug(
        f"{mesh_device.shape=} {experts_per_chip=} {max_dispatched_tokens_per_expert=} {emb_dim=} {hidden_dim=}"
    )

    x, router_weights, indices = initialize_test_inputs(
        dispatch_group_size,
        seq_len_per_chip,
        emb_dim,
        num_routed_experts,
        num_experts_per_tok,
        max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
    )

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    dispatched_buffer, dispatched_metadata = TorchDispatchModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_dispatch_groups=num_dispatch_groups,
        expert_dispatch_table=expert_dispatch_table,
    )(x, router_weights, indices, expert_offsets)

    # Indexed by GLOBAL expert id: that is the order TtRoutedExpert's mesh distribution expects,
    # and the reference walks it through the same ExpertMapping.
    expert_weights = [
        {
            "gate_proj": torch.randn(hidden_dim, emb_dim) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim) * 0.02,
        }
        for _ in range(num_routed_experts)
    ]

    torch_expert_out = _torch_routed_expert(
        dispatched_buffer,
        expert_token_counts,
        expert_region_offsets,
        expert_weights,
        experts_per_chip,
        emb_dim,
        hidden_dim,
    )
    torch_output = TorchCombineModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        num_dispatch_groups=num_dispatch_groups,
    )(torch_expert_out, dispatched_metadata, expert_token_counts, expert_region_offsets)

    mesh_mapper = get_ep_mesh_mapper(mesh_device)
    counts_mapper = get_expert_token_counts_mesh_mapper(mesh_device)
    tt_x = ttnn.reshape(
        ttnn.from_torch(
            dispatched_buffer,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat16,
        ),
        (max_dispatch_buffer_token_size, emb_dim),
    )
    tt_metadata = ttnn.from_torch(
        dispatched_metadata,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    # UINT32, not the INT32 the combine-only test uses: the routed-expert half reads these as the
    # index vectors it fused extract/insert around and rejects anything else. Same four-byte pages
    # either way, so the combine half is indifferent.
    tt_counts = ttnn.squeeze(
        ttnn.from_torch(
            expert_token_counts,
            mesh_mapper=counts_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        ),
        0,
    )
    # Squeezed to 2D: the routed-expert half inherits ttnn::insert's rule that the index vector is
    # 1D, or 2D with a leading 1. The combine half reads it by page, so the rank does not reach it.
    tt_region_offsets = ttnn.squeeze(
        ttnn.from_torch(
            expert_region_offsets,
            mesh_mapper=counts_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        ),
        0,
    )
    # Replicated along the ring axis: every chip needs every origin chip's run boundaries for the
    # experts it hosts.
    tt_expert_offsets = ttnn.from_torch(
        expert_offsets,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 0)),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # The op takes this rather than allocating it, so it has to carry the spec combine's own op
    # would have produced; validate_arguments rejects anything else.
    tt_combine_output = ttnn.zeros(
        ttnn.Shape([1, 1, seq_len_per_chip, num_experts_per_tok, emb_dim]),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )

    # (group, chip, local) -> global expert id, sharded so each chip holds its own
    # (experts_per_chip,) row. Squeezed to 1D, which the extract/insert validators require.
    idx_table = ttnn.from_torch(
        ExpertMapping.create_global_expert_idx_table(
            experts_per_chip=experts_per_chip,
            dispatch_group_size=dispatch_group_size,
            num_dispatch_groups=num_dispatch_groups,
        ),
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint32,
    )
    idx_table = ttnn.squeeze(ttnn.squeeze(idx_table, 0), 0)

    tt_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=idx_table,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_dispatched_tokens_per_expert,
        torch_weights=expert_weights,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        activation=ttnn.RoutedExpertActivation.Silu,
    )

    ttnn.experimental.deepseek_prefill.hybrid_routed_expert_moe(
        tt_x,
        tt_region_offsets,
        tt_counts,
        idx_table,
        tt_expert.gate_projs,
        tt_expert.up_projs,
        tt_expert.down_projs,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        hybrid_token_threshold=_TOKEN_THRESHOLD,
        compute_kernel_config=tt_expert.compute_kernel_config,
        overlap_combine=True,
        dispatched_metadata=tt_metadata,
        expert_offsets=tt_expert_offsets,
        combine_output=tt_combine_output,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=_NUM_LINKS,
        topology=topology,
    )
    ttnn.synchronize_device(mesh_device)

    tt_output_torch = ttnn.to_torch(tt_combine_output, mesh_composer=get_ep_mesh_composer(mesh_device))
    assert_output_shape(tt_output_torch, num_dispatch_groups, dispatch_group_size, "overlapped combine output")
    result = validate_combine_output(
        torch_output,
        tt_output_torch,
        indices,
        num_dispatch_groups,
        num_routed_experts,
        # The routed expert runs in bfp8, so the combined result cannot survive an allclose the
        # way the combine-only test's bf16 passthrough does.
        use_pcc=True,
        verbose=True,
        expert_dispatch_table=expert_dispatch_table,
        expert_token_counts=expert_token_counts,
        experts_per_chip=experts_per_chip,
    )
    if not result.passed:
        log_combine_mismatch_details(result.mismatches, torch_output, tt_output_torch)
    result.assert_passed("Overlapped combine data mismatch")
