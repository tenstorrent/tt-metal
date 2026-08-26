# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE (>=8 chips): the ring-joint SDPA over the block-cyclic SP KV cache, vs torch.

This isolates ``tt/attention/dense_sp.py::dense_sp_attention`` from the rest of the block: Q, K and V
are placed directly, so a failure here is the ring op or the cache layout, not the projections.

The grouping is the point. At TP=4 each chip carries **2 KV heads and 24 Q heads**, so the op runs in
its grouped-GQA mode (``NKH == NVH < NQH && NQH % NKH == 0`` — see
``ring_joint_sdpa_device_operation.cpp:759-765``). Every other model in the repo drives this op with
either 1 KV head per chip or MLA's single latent, so this ratio is new.

The ring gathers KV across the SP axis internally via online softmax — there is no explicit
AllGather of K/V — so the reference is a plain full-sequence causal SDPA and the test checks that
each chip's Q shard sees the whole prefix.

Run:  pytest models/demos/mistral_medium_d_p/tests/unit/test_ring_joint_sp_vs_ref.py -k 2x4
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.mistral_medium_d_p.tt.attention import allocate_kv_cache, write_kv_chunk
from models.demos.mistral_medium_d_p.tt.attention.dense_sp import dense_sp_attention

from ..test_factory import mesh_setup, parametrize_mesh_with_fabric
from .shapes import HEAD_DIM, N_KV, N_Q, per_chip


def _torch_causal_gqa(q, k, v, n_q, n_kv):
    """[1, n_q, S, hd] x [1, n_kv, S, hd] -> [1, n_q, S, hd], dense causal, GQA-expanded."""
    rep = n_q // n_kv
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    scores = (q @ k.transpose(-1, -2)) * (HEAD_DIM**-0.5)
    S = q.shape[2]
    scores = scores + torch.triu(torch.full((S, S), float("-inf"), dtype=scores.dtype), diagonal=1)
    return torch.softmax(scores, dim=-1) @ v


@parametrize_mesh_with_fabric(mesh_shapes=[(2, 4), (8, 4)])
@pytest.mark.parametrize("chunk_local", [128], ids=["c128"])
def test_ring_joint_sp_vs_ref(mesh_device, device_params, chunk_local, reset_seeds):
    """One cache-backed ring-joint SDPA call, 2 KV / 24 Q heads per chip, vs torch."""
    rows, cols = tuple(mesh_device.shape)
    mesh_config, ccl = mesh_setup(mesh_device)
    sp, tp, sp_axis, tp_axis = mesh_config.sp, mesh_config.tp, mesh_config.sp_axis, mesh_config.tp_axis
    pc = per_chip(tp)
    n_q_local, n_kv_local = pc["n_q"], pc["n_kv"]

    C = chunk_local
    chunk_global = sp * C
    # Cache capacity strictly greater than the first chunk so the op sees short-Q / long-KV, which
    # is what the cache-backed ring reader requires (an equal-sized one-shot is rejected).
    cache_global = 2 * chunk_global

    torch.manual_seed(0)
    q_full = torch.randn(1, N_Q, chunk_global, HEAD_DIM, dtype=torch.bfloat16) * 0.3
    k_full = torch.randn(1, N_KV, chunk_global, HEAD_DIM, dtype=torch.bfloat16) * 0.3
    v_full = torch.randn(1, N_KV, chunk_global, HEAD_DIM, dtype=torch.bfloat16) * 0.3

    ref = _torch_causal_gqa(q_full.float(), k_full.float(), v_full.float(), N_Q, N_KV)

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=cache_global,
        sp_axis=sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        n_kv_local=n_kv_local,
    )

    # Block-cyclic chip order for chunk 0, shared by the KV write and the Q placement.
    positions = rotated_chip_positions(0, sp, C)
    idx = torch.tensor([positions[c][r] for c in range(sp) for r in range(C)], dtype=torch.long)

    def shard_heads_and_seq(t, dtype=ttnn.bfloat8_b):
        dims = [None, None]
        dims[sp_axis] = 2
        dims[tp_axis] = 1
        return ttnn.from_torch(
            t[:, :, idx, :],
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=tuple(dims)),
        )

    write_kv_chunk(
        kv_cache,
        shard_heads_and_seq(k_full),
        shard_heads_and_seq(v_full),
        slot_idx=0,
        layer_idx=0,
        kv_actual=0,
        sp_axis=sp_axis,
    )
    ttnn.synchronize_device(mesh_device)

    tt_q = shard_heads_and_seq(q_full, dtype=ttnn.bfloat16)

    grid = mesh_device.compute_with_storage_grid_size()
    out_tt = dense_sp_attention(
        tt_q,
        kv_cache.k,
        kv_cache.v,
        None,
        None,
        kv_actual=0,
        logical_n=chunk_global,
        n_kv=N_KV,
        cache_global=cache_global,
        head_dim=HEAD_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),  # carve the CCL column
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        ),
        # fp32_dest_acc_en=False is required by the ring op's streaming compute.
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        ),
        scale=HEAD_DIM**-0.5,
        cluster_axis=sp_axis,
        slot_idx=0,
        layer_idx=0,
        num_layers=1,
        write_chunk=False,  # already written above
    )

    # Output is [1, n_q_local, C, hd] per chip: heads on the TP cols, this chunk's seq on the SP rows.
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1
    got = ttnn.to_torch(
        out_tt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=(rows, cols))
    ).float()
    # Undo the block-cyclic ordering to compare in natural position order.
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(chunk_global)
    got = got[:, :, inv, :]

    passing, pcc = comp_pcc(ref, got, 0.99)
    logger.info(f"ring-joint SDPA SP={sp} TP={tp} ({n_q_local}Q/{n_kv_local}KV per chip): {pcc}")
    assert passing, f"ring-joint SDPA PCC fail: {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(2, 4)])
def test_ring_requires_bf8_cache(mesh_device, device_params, reset_seeds, expect_error):
    """The sliding/cache-backed ring path and its gather buffers are bf8; a bf16 cache must fail loud."""
    mesh_config, ccl = mesh_setup(mesh_device)
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=64 * mesh_config.sp,
        sp_axis=mesh_config.sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        n_kv_local=per_chip(mesh_config.tp)["n_kv"],
        cache_dtype=ttnn.bfloat16,
    )
    with expect_error(AssertionError, "bf8 KV cache"):
        dense_sp_attention(
            None,
            kv_cache.k,
            kv_cache.v,
            None,
            None,
            kv_actual=0,
            logical_n=64,
            n_kv=N_KV,
            cache_global=64 * mesh_config.sp,
            head_dim=HEAD_DIM,
            mesh_device=mesh_device,
            ccl_manager=ccl,
            program_config=None,
            compute_kernel_config=None,
            scale=1.0,
            cluster_axis=mesh_config.sp_axis,
            write_chunk=False,
        )
