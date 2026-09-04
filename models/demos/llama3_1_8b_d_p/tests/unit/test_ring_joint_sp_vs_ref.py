# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: the SP-sharded ring SDPA with LIVE Q/K/V (no cache), SP=8 x TP=4, vs a torch GQA golden.

De-risks the SP attention mechanism in isolation: that gathering K/V across the SP axis by online
softmax gives the same answer as unsharded attention. Q ``[1,32,S,128]`` and K/V ``[1,8,S,128]`` are
already-projected randoms, so this tests the op and the CCL wiring
(``ring_attention_ccl_semaphore_handles`` + ``ring_attention_ccl_core_grid_offset``: CCL workers in
the last compute column, SDPA on the carved grid) and nothing else.

Llama's GQA shape at TP=4 puts **2 KV heads on each chip** (both donors put 1), and the group size is
4 rather than 8 or 16 — so the op's K/V broadcast is exercised at a ratio neither donor covers.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

from ..test_factory import llama_config, make_ccl, make_mesh_config, parametrize_mesh_with_fabric

PCC = 0.99


def torch_gqa_causal(q, k, v, n_q, n_kv, head_dim):
    """fp32 GQA causal SDPA golden. q [1,n_q,S,HD], k/v [1,n_kv,S,HD]."""
    rep = n_q // n_kv
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = q.shape[2]
    scores = (q @ k.transpose(-1, -2)) * (head_dim**-0.5)
    causal = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
    return torch.softmax(scores + causal, dim=-1) @ v


def gather_sp_tp(mesh_device, out):
    """Reassemble ``[1, n_q, S, HD]``: per SP row concat the TP cols on heads, then rows on seq."""
    rows, cols = tuple(mesh_device.shape)
    dts = ttnn.get_device_tensors(out)
    row_t = [torch.cat([ttnn.to_torch(dts[r * cols + c]).float() for c in range(cols)], dim=1) for r in range(rows)]
    return torch.cat(row_t, dim=2)


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("seq_len", [512, 4096], ids=["s512", "s4096"])
def test_ring_joint_sp_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    cfg = llama_config()
    n_q, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    mesh_config = make_mesh_config(mesh_device)
    rows, cols = tuple(mesh_device.shape)
    sp_axis, tp_axis = mesh_config.sp_axis, mesh_config.tp_axis

    q = torch.randn(1, n_q, seq_len, hd) * 0.1
    k = torch.randn(1, n_kv, seq_len, hd) * 0.1
    v = torch.randn(1, n_kv, seq_len, hd) * 0.1
    reference = torch_gqa_causal(q.float(), k.float(), v.float(), n_q, n_kv, hd)

    ccl = make_ccl(mesh_device)

    qkv_dims = [None, None]
    qkv_dims[sp_axis] = 2  # sequence -> rows
    qkv_dims[tp_axis] = 1  # heads    -> cols

    def shard(t, dt=ttnn.bfloat16):
        return ttnn.from_torch(
            t,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=qkv_dims),
        )

    tt_q, tt_k, tt_v = shard(q), shard(k), shard(v)

    # Persistent gather buffers hold the FULL sequence per chip: heads on cols, seq replicated.
    pbuf_dims = [None, None]
    pbuf_dims[tp_axis] = 1

    def pbuf(n_heads):
        return ttnn.from_torch(
            torch.zeros(1, n_heads, seq_len, hd),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=pbuf_dims),
        )

    grid = mesh_device.compute_with_storage_grid_size()
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),  # carve out the CCL column
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    kernel_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )

    out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        None,
        None,
        None,
        persistent_output_buffer_k=pbuf(n_kv),
        persistent_output_buffer_v=pbuf(n_kv),
        joint_strategy="rear",
        logical_n=seq_len,
        program_config=prog,
        compute_kernel_config=kernel_cfg,
        dim=2,
        multi_device_global_semaphore=ccl.ring_attention_ccl_semaphore_handles,
        num_links=ccl.num_links,
        cluster_axis=sp_axis,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        ccl_core_grid_offset=ccl.ring_attention_ccl_core_grid_offset,
        use_column_major_ccl=True,
        is_causal=True,
        scale=hd**-0.5,
        is_balanced=False,
    )

    full = gather_sp_tp(mesh_device, out)
    passing, pcc = comp_pcc(reference, full, PCC)
    logger.info(f"ring_joint SP live q/k/v s={seq_len}: pcc={pcc}")
    assert passing, f"ring_joint SP PCC fail: {pcc}"
