# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
THE LINK: ring_joint reading from the GQA chunked-KV cache (cache-read mode), SP=8 × TP=4 on (8,4).

Ties the two validated building blocks together — the GQA chunked-KV cache write
(test_kv_cache_gqa_sp_vs_ref) and the SP ring_joint op (test_ring_joint_sp_vs_ref) — into the actual
chunked-prefill attention: write K/V into the cache, then run
`ring_joint_scaled_dot_product_attention` with `kv_cache_batch_idx` + `kv_actual_isl` so it reads the
accumulated prefix FROM the cache (block-cyclic, SP-sharded) and runs causal online-softmax over it.
Grouped V (cache stays 4 heads → 1/chip; NO inflation). vs a full-causal-GQA torch golden.

Single chunk = whole sequence (kv_actual_isl=0, logical_n=S) — the minimal cache-read verification;
multi-chunk accumulation + SP-RoPE are follow-ups. This is the path that wires into prefill.py:124.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.minimax_m3.tt.attention.dense_sp import dense_sp_attention
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

NQ, NKV, HEAD_DIM = 64, 4, 128


def _torch_gqa_causal(q, k, v):
    rep = NQ // NKV
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = q.shape[2]
    scores = (q @ k.transpose(-1, -2)) * (HEAD_DIM**-0.5)
    causal = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
    return torch.softmax(scores + causal, dim=-1) @ v  # [1, NQ, S, HD]


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize(
    "n_chunks,chunk_local",
    [(2, 32), (2, 640)],  # 2x256 (quick) and 2x5120 — the REAL M3 prefill chunk (640/chip at SP=8)
    ids=["2x256", "2x5120"],
)
def test_ring_joint_cache_read_sp(mesh_device, device_params, n_chunks, chunk_local, reset_seeds):
    """ring_joint cache-read: last chunk's Q attends to the full cached prefix, SP=8 x TP=4, vs golden.

    Chunked mode requires Q.seq < cached K.seq, so we write n_chunks into the cache (= full prefix) and
    run attention for ONLY the LAST chunk's queries (kv_actual_isl = prefix-before-it, logical_n = full).
    """
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis, tp_axis = rows, cols, 0, 1
    C = chunk_local
    chunk_global = sp * C  # 256
    cache_global = n_chunks * chunk_global  # 512
    kv_actual_last = (n_chunks - 1) * chunk_global  # prefix length before the last chunk (256)

    torch.manual_seed(0)
    q = torch.randn(1, NQ, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k = torch.randn(1, NKV, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    v = torch.randn(1, NKV, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    ref_full = _torch_gqa_causal(q.float(), k.float(), v.float())  # [1, NQ, cache_global, HD]
    ref = ref_full[:, :, kv_actual_last:, :]  # golden for the LAST chunk's query positions

    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    # --- write all chunks into the GQA chunked-KV cache (block-cyclic) ---
    cache_k = init_kvpe_cache(HEAD_DIM, mesh_device, cache_global, list(mesh_device.shape), sp_axis, 1)
    cache_v = init_kvpe_cache(HEAD_DIM, mesh_device, cache_global, list(mesh_device.shape), sp_axis, 1)
    wr_dims = [None, None]
    wr_dims[sp_axis], wr_dims[tp_axis] = 2, 1

    def bc_index(kv_actual):
        pos = rotated_chip_positions(kv_actual, sp, C)
        return torch.tensor([pos[c][r] for c in range(sp) for r in range(C)], dtype=torch.long)

    def make_chunk(src, kv_actual):
        chunk = src[:, bc_index(kv_actual), :].reshape(1, NKV, chunk_global, HEAD_DIM)
        return ttnn.from_torch(
            chunk,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=wr_dims),
        )

    def write(cache, src, kv_actual):
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            make_chunk(src, kv_actual),
            slot_idx=0,
            layer_idx=0,
            num_layers=1,
            kv_actual_global=kv_actual,
            cluster_axis=sp_axis,
        )

    # Write the PRIOR chunks into the cache; the LAST chunk is written by dense_sp_attention below.
    for c in range(n_chunks - 1):
        kv_actual = c * chunk_global
        write(cache_k, k[0], kv_actual)
        write(cache_v, v[0], kv_actual)
    ttnn.synchronize_device(mesh_device)

    # --- Q = the LAST chunk's queries, block-cyclic within that chunk, sharded (seq rows, heads cols) ---
    last_idx = bc_index(kv_actual_last)  # global positions of the last chunk, block-cyclic
    q_bc = q[:, :, last_idx, :]  # [1, NQ, chunk_global, HD]
    q_dims = [None, None]
    q_dims[sp_axis], q_dims[tp_axis] = 2, 1
    tt_q = ttnn.from_torch(
        q_bc,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=q_dims),
    )

    grid = mesh_device.compute_with_storage_grid_size()
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=128,
        k_chunk_size=512,
        exp_approx_mode=False,  # Pavle's minimax3_gqa_causal_perf tuning
    )
    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )

    # dense_sp_attention writes the LAST chunk into the cache, then ring_joint cache-read: K/V come
    # from the cache (kv_cache_batch_idx=slot 0); kv_actual_isl = prefix before the last chunk,
    # logical_n = full valid prefix (Q attends causally over [0:logical_n]).
    out = dense_sp_attention(
        tt_q,
        cache_k,
        cache_v,
        make_chunk(k[0], kv_actual_last),
        make_chunk(v[0], kv_actual_last),
        kv_actual=kv_actual_last,
        logical_n=cache_global,
        n_kv=NKV,
        cache_global=cache_global,
        head_dim=HEAD_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        program_config=prog,
        compute_kernel_config=kcfg,
        scale=HEAD_DIM**-0.5,
        cluster_axis=sp_axis,
    )

    # out per chip [1, NQ/tp, C, HD], block-cyclic over the LAST chunk. Gather heads (cols) + seq (rows).
    dts = ttnn.get_device_tensors(out)
    row_t = []
    for r in range(rows):
        row_t.append(torch.cat([ttnn.to_torch(dts[r * cols + c]).float() for c in range(cols)], dim=1))  # [1,NQ,C,HD]
    full_bc = torch.cat(row_t, dim=2)  # [1, NQ, chunk_global, HD] block-cyclic over the last chunk
    # invert block-cyclic -> natural order WITHIN the last chunk (last_idx are global; subtract the offset)
    local_pos = last_idx - kv_actual_last  # positions in [0, chunk_global)
    inv = torch.empty(chunk_global, dtype=torch.long)
    inv[local_pos] = torch.arange(chunk_global)
    full = full_bc[:, :, inv, :]  # [1, NQ, chunk_global, HD] natural order (= positions kv_actual_last:cache_global)

    passing, pcc = comp_pcc(ref, full, 0.99)
    logger.info(f"ring_joint CACHE-READ SP=8 x TP=4 vs ref: pcc={pcc}")
    assert passing, f"cache-read PCC fail: {pcc}"
