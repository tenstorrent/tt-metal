# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Chained unit test for the post-combine tail of ``TTMoEDecode.forward()``:

    moe_compute output → deepseek_moe_post_combine_tilize → deepseek_moe_fast_reduce_nc_fused

i.e. steps 3 and 4 of the forward pipeline (see
``models/common/modules/moe/tt_moe_decode.py``). ``post_combine_tilize`` only changes
layout (ROW_MAJOR → TILE, into an L1 ND-shard), so the end-to-end golden is identical
to the fast-reduce golden applied to the (un-tilized) activation — we reuse
``_torch_golden_gated`` and the input generators from the standalone fast-reduce test,
and the exact default memory configs that ``TTMoEDecodeConfig`` feeds the two ops.

Input shapes are taken from ``models/common/modules/moe/configs/deepseek_v3_single_glx.yaml``
(batch_per_device, hidden_size, select_experts_k, num_routed_experts, num_shared_experts,
shared_expert_scale); the mesh is overridden to a Blackhole 1x8 (gated via skipif).
"""

import random

import pytest
import torch
from loguru import logger

import ttnn
from models.common.modules.moe.tt_moe_decode_config import (
    _default_fast_reduce_output_memory_config,
    _default_post_combine_tilize_memory_config,
)
from models.common.utility_functions import comp_pcc, is_blackhole
from tests.nightly.tg.ccl.moe.test_deepseek_moe_fast_reduce_nc_fused import (
    _gen_expert_mapping_linearized,
    _get_expert_indices,
    _torch_golden_gated,
)

# Scalars from models/common/modules/moe/configs/deepseek_v3_single_glx.yaml. Only the
# per-device tensor shapes are taken from the YAML; the mesh shape is overridden below
# to a 1x8 Blackhole mesh (the YAML itself targets an [8, 4] galaxy).
_DS_SINGLE_GLX = {
    "batch_per_device": 32,
    "hidden_size": 7168,
    "select_experts_k": 8,
    "num_routed_experts": 256,
    "num_shared_experts": 1,
    "shared_expert_scale": 1.0,
}

PCC_THRESHOLD = 0.999


@pytest.mark.skipif(not is_blackhole(), reason="1x8 mesh case targets a Blackhole 1x8 machine")
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [pytest.param((1, 8), (1, 8), id="1x8_grid_bh")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("cluster_axis", [0, 1])
def test_deepseek_moe_tilize_fast_reduce_chain(mesh_device, mesh_shape, cluster_axis):
    torch.manual_seed(2005)
    random.seed(2005)

    cfg = _DS_SINGLE_GLX
    hidden_size = cfg["hidden_size"]
    select_experts_k = cfg["select_experts_k"]
    num_shared_experts = cfg["num_shared_experts"]
    batch_per_device = cfg["batch_per_device"]
    shared_expert_scale = cfg["shared_expert_scale"]
    seq = 1

    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis]
    num_replicated_devices = mesh_shape[1 - cluster_axis]
    batch = batch_per_device * num_dispatch_devices
    experts = cfg["num_routed_experts"]
    assert (
        experts % num_devices == 0
    ), f"num_routed_experts ({experts}) must be divisible by num_devices ({num_devices})"

    # Activation reduction dim spans routed + trailing shared-expert slots; the shared
    # slots are scaled by ``shared_expert_scale`` (no per-token gating) — matching the op.
    effective_select_experts_k = select_experts_k + num_shared_experts
    shared_expert_scale_bf16 = torch.tensor(shared_expert_scale, dtype=torch.bfloat16)

    # The exact memory configs TTMoEDecodeConfig derives for these two ops.
    post_combine_tilize_output_memory_config = _default_post_combine_tilize_memory_config(
        effective_select_experts_k, hidden_size
    )
    assert (
        post_combine_tilize_output_memory_config is not None
    ), "expected a valid ND-shard tilize config for this (effective_experts_k, hidden_size)"
    fast_reduce_output_memory_config = _default_fast_reduce_output_memory_config()

    replicate_mapper = ttnn.replicate_tensor_to_mesh_mapper(mesh_device)

    # Global tensors. The activation is the (unsqueezed) moe_compute output; its reduction
    # dim spans routed + shared expert slots.
    torch_unsqueezed_global = (
        torch.rand((effective_select_experts_k, 1, batch, hidden_size), dtype=torch.bfloat16) - 0.5
    )
    torch_expert_scores_global = torch.rand((batch, 1, seq, select_experts_k), dtype=torch.bfloat16)
    torch_expert_scores_global = torch_expert_scores_global / torch_expert_scores_global.sum(dim=-1, keepdim=True)
    torch_expert_indices_global = _get_expert_indices(batch, experts, select_experts_k, seq)
    torch_expert_mapping = _gen_expert_mapping_linearized(experts, num_devices)

    per_device_goldens = []
    for m0 in range(mesh_shape[0]):
        for m1 in range(mesh_shape[1]):
            mesh_coord = (m0, m1)
            t0 = (m1 if cluster_axis == 1 else m0) * batch_per_device
            t1 = t0 + batch_per_device
            u_slice = torch_unsqueezed_global[:, :, t0:t1, :].contiguous()
            s_slice = torch_expert_scores_global[t0:t1, :, :, :].contiguous()
            ind_slice = torch_expert_indices_global[t0:t1, :, :, :].contiguous()
            per_device_goldens.append(
                _torch_golden_gated(
                    u_slice,
                    s_slice,
                    ind_slice,
                    torch_expert_mapping,
                    mesh_shape,
                    mesh_coord,
                    cluster_axis,
                    num_replicated_devices,
                    num_shared_experts,
                    shared_expert_scale_bf16,
                )
            )

    def _shard_dims(dim):
        return (dim, None) if cluster_axis == 0 else (None, dim)

    # Activation feeds post_combine_tilize as ROW_MAJOR (the op tilizes it). Sharded on the
    # token dim across the dispatch axis, exactly like the moe_compute output in forward().
    tt_activation_rm = ttnn.from_torch(
        torch_unsqueezed_global,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape, dims=_shard_dims(2)),
    )
    tt_scores_dram = ttnn.from_torch(
        torch_expert_scores_global,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape, dims=_shard_dims(0)),
    )
    tt_expert_indices = ttnn.from_torch(
        torch_expert_indices_global,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape, dims=_shard_dims(0)),
    )
    tt_expert_mapping = ttnn.from_torch(
        torch_expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate_mapper,
    )

    # Op 1: ROW_MAJOR activation → TILE, L1 ND-shard.
    tt_tilized_compute_output = ttnn.experimental.deepseek_moe_post_combine_tilize(
        tt_activation_rm,
        output_memory_config=post_combine_tilize_output_memory_config,
    )

    # Op 2: per-(t, k) on-axis-gated score-weighted reduce over the expert dim, with the
    # trailing shared-expert slots scaled by shared_expert_scale and summed in.
    tt_fused_outputs = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
        tt_tilized_compute_output,
        tt_expert_indices,
        tt_expert_mapping,
        reduce_dim=0,
        split_size=int(hidden_size // num_replicated_devices),
        cluster_axis=cluster_axis,
        output_memory_config=fast_reduce_output_memory_config,
        scores_tensor=tt_scores_dram,
        num_shared_experts=num_shared_experts,
        shared_expert_scale=shared_expert_scale,
    )

    for cidx, tt_out_list in enumerate(tt_fused_outputs):
        for didx, tt_out in enumerate(ttnn.get_device_tensors(tt_out_list)):
            tt_host = ttnn.to_torch(tt_out, dtype=torch.bfloat16)
            ref = per_device_goldens[didx][cidx]
            ok, msg = comp_pcc(ref, tt_host, pcc=PCC_THRESHOLD)
            logger.info(f"virtual_dev={didx} chunk={cidx}: {msg}")
            assert ok, f"virtual_dev={didx} chunk={cidx} failed: {msg}"

    ttnn.deallocate(tt_activation_rm)
    ttnn.deallocate(tt_tilized_compute_output)
    ttnn.deallocate(tt_scores_dram)
    ttnn.deallocate(tt_expert_indices)
    ttnn.deallocate(tt_expert_mapping)
    for t in tt_fused_outputs:
        ttnn.deallocate(t)
