# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
SP=8 × TP=4 ring-attention op validation for MiniMax-M3 dense attention.

De-risks the SP-1 attention mechanism IN ISOLATION (before wiring into attention/prefill.py): calls
`ttnn.transformer.ring_joint_scaled_dot_product_attention` on M3's GQA shapes
(Q[1,64,S,128], K/V[1,4,S,128]) sharded across the (8,4) galaxy — heads on the TP axis (cols),
sequence on the SP axis (rows) — and PCCs the gathered result against a torch GQA-causal SDPA golden.

This exercises the pieces we just added to CCLManager: ring_attention_ccl_semaphore_handles +
ring_attention_ccl_core_grid_offset (CCL workers in the last compute column, SDPA on the carved grid),
cluster_axis = sp_axis (rows). Models the call on deepseek_v3_d_p/tt/mla/mla.py:857 (plain GQA, no MLA
latent). RoPE/QKV-proj integration into the full Attention module (incl. SP-sharded RoPE positions) is
a SEPARATE next step — here Q/K/V are already-projected randoms so the op is tested on its own.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

NQ, NKV, HEAD_DIM = 64, 4, 128


def _torch_gqa_causal(q, k, v):
    """fp32 GQA causal SDPA golden. q[1,NQ,S,HD], k/v[1,NKV,S,HD]."""
    rep = NQ // NKV
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    s = q.shape[2]
    scores = (q @ k.transpose(-1, -2)) * (HEAD_DIM**-0.5)
    causal = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
    attn = torch.softmax(scores + causal, dim=-1)
    return attn @ v  # [1, NQ, S, HD]


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("seq_len", [512, 5120], ids=["s512", "s5120"])  # 5120 = real prefill (640/dev)
def test_ring_joint_sp_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """ring_joint GQA causal attention, SP=8 (rows) × TP=4 (cols), vs torch golden. Needs (8,4)."""
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4), "SP=8 × TP=4 layout expected"
    torch.manual_seed(0)

    q = torch.randn(1, NQ, seq_len, HEAD_DIM) * 0.1
    k = torch.randn(1, NKV, seq_len, HEAD_DIM) * 0.1
    v = torch.randn(1, NKV, seq_len, HEAD_DIM) * 0.1
    ref = _torch_gqa_causal(q.float(), k.float(), v.float())  # [1, NQ, S, HD]

    mesh_config = MeshConfig((rows, cols), tp=cols)
    sp_axis, tp_axis = mesh_config.sp_axis, mesh_config.tp_axis  # 0 (rows), 1 (cols)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    # Q/K/V: shard heads on TP cols (dim1), sequence on SP rows (dim2). dims indexed by mesh axis.
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

    # GQA: K and V both stay grouped (4 heads -> 1/chip). The patched ring_joint broadcasts V to the
    # query heads internally (nv = nq / (NQH/NVH)), symmetric to the K broadcast — no V inflation.
    tt_q, tt_k, tt_v = shard(q), shard(k), shard(v)

    # Persistent K/V buffers hold the gathered FULL sequence per chip: heads on cols, seq replicated.
    pbuf_dims = [None, None]
    pbuf_dims[tp_axis] = 1

    def pbuf(n_heads):
        return ttnn.from_torch(
            torch.zeros(1, n_heads, seq_len, HEAD_DIM),
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
        persistent_output_buffer_k=pbuf(NKV),
        persistent_output_buffer_v=pbuf(NKV),
        joint_strategy="rear",
        logical_n=seq_len,  # full sequence reconstructed across the ring
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
        scale=HEAD_DIM**-0.5,
        is_balanced=False,
    )

    # Reconstruct [1, NQ, S, HD]: per row, concat the 4 TP cols on the head dim; then concat rows on seq.
    dts = ttnn.get_device_tensors(out)
    row_tensors = []
    for r in range(rows):
        col_tensors = [ttnn.to_torch(dts[r * cols + c]).float() for c in range(cols)]  # each [1,16,S/8,128]
        row_tensors.append(torch.cat(col_tensors, dim=1))  # [1, NQ, S/8, 128]
    full = torch.cat(row_tensors, dim=2)  # [1, NQ, S, 128]

    passing, pcc = comp_pcc(ref, full, 0.99)
    logger.info(f"ring_joint SP=8 x TP=4 vs ref: pcc={pcc}, shape={tuple(full.shape)}")
    assert passing, f"ring_joint SP PCC fail: {pcc}"
