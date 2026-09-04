# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: the ring SDPA reading K/V OUT of the block-cyclic KV cache, SP=8 x TP=4.

This is the mechanism chunked prefill depends on, and the link between the two blocks validated
separately: the KV-cache write (``test_kv_cache_write_vs_ref``) and the live-tensor ring SDPA
(``test_ring_joint_sp_vs_ref``). Prior chunks are written into the cache; then a SHORT Q (the last
chunk) attends the LONGER accumulated prefix via ``kv_cache_batch_idx`` + ``kv_actual_isl``.

It runs on **this package's** ``allocate_kv_cache`` — 2 KV heads per chip — rather than the donors'
``init_kvpe_cache`` single-head test cache. That is the point: it proves the cache-read path works at
Llama's TP=4 head split, which is the one structural thing neither donor exercises.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.llama3_1_8b_d_p.tt.attention.dense_sp import dense_sp_attention
from models.demos.llama3_1_8b_d_p.tt.attention.kv_cache import allocate_kv_cache

from ..test_factory import llama_config, make_ccl, make_mesh_config, parametrize_mesh_with_fabric
from .test_ring_joint_sp_vs_ref import gather_sp_tp, torch_gqa_causal

PCC = 0.99


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("n_chunks, chunk_local", [(2, 32), (2, 512)], ids=["2x256", "2x4096"])
def test_ring_joint_cache_read_sp_vs_ref(mesh_device, device_params, n_chunks, chunk_local, reset_seeds):
    """The last chunk's Q attends the full cached prefix; compared against unsharded GQA causal.

    Chunked mode requires Q.seq < cached K.seq, so ``n_chunks`` are written into the cache and
    attention runs for only the LAST chunk's queries: ``kv_actual_isl`` = prefix before it,
    ``logical_n`` = the full valid prefix.
    """
    cfg = llama_config()
    n_q, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    mesh_config = make_mesh_config(mesh_device)
    rows, cols = tuple(mesh_device.shape)
    sp, sp_axis, tp_axis = mesh_config.sp, mesh_config.sp_axis, mesh_config.tp_axis
    n_kv_local = n_kv // mesh_config.tp
    assert n_kv_local == 2, f"this test pins the 2-KV-heads-per-chip case; got {n_kv_local}"

    C = chunk_local
    chunk_global = sp * C
    cache_global = n_chunks * chunk_global
    kv_actual_last = (n_chunks - 1) * chunk_global

    q = torch.randn(1, n_q, cache_global, hd, dtype=torch.bfloat16) * 0.1
    k = torch.randn(1, n_kv, cache_global, hd, dtype=torch.bfloat16) * 0.1
    v = torch.randn(1, n_kv, cache_global, hd, dtype=torch.bfloat16) * 0.1
    ref_full = torch_gqa_causal(q.float(), k.float(), v.float(), n_q, n_kv, hd)
    reference = ref_full[:, :, kv_actual_last:, :]  # golden for the last chunk's query positions

    ccl = make_ccl(mesh_device)
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=cache_global,
        sp_axis=sp_axis,
        num_users=1,
        head_dim=hd,
        num_kv_heads_local=n_kv_local,
    )

    wr_dims = [None, None]
    wr_dims[sp_axis], wr_dims[tp_axis] = 2, 1

    def bc_index(kv_actual):
        """Global token positions of one chunk, in block-cyclic (per-chip) order."""
        pos = rotated_chip_positions(kv_actual, sp, C)
        return torch.tensor([pos[c][r] for c in range(sp) for r in range(C)], dtype=torch.long)

    def make_chunk(src, kv_actual):
        chunk = src[:, bc_index(kv_actual), :].reshape(1, n_kv, chunk_global, hd)
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

    # Write the PRIOR chunks; dense_sp_attention writes the last one below.
    for c in range(n_chunks - 1):
        kv_actual = c * chunk_global
        write(kv_cache.k, k[0], kv_actual)
        write(kv_cache.v, v[0], kv_actual)
    ttnn.synchronize_device(mesh_device)

    last_idx = bc_index(kv_actual_last)
    q_dims = [None, None]
    q_dims[sp_axis], q_dims[tp_axis] = 2, 1
    tt_q = ttnn.from_torch(
        q[:, :, last_idx, :],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=q_dims),
    )

    grid = mesh_device.compute_with_storage_grid_size()
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1, grid.y),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    kcfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )

    out = dense_sp_attention(
        tt_q,
        kv_cache.k,
        kv_cache.v,
        make_chunk(k[0], kv_actual_last),
        make_chunk(v[0], kv_actual_last),
        kv_actual=kv_actual_last,
        logical_n=cache_global,
        n_kv=n_kv,
        cache_global=cache_global,
        head_dim=hd,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        program_config=prog,
        compute_kernel_config=kcfg,
        scale=hd**-0.5,
        cluster_axis=sp_axis,
        slot_idx=0,
        layer_idx=0,
        num_layers=1,
        write_chunk=True,
    )

    full_bc = gather_sp_tp(mesh_device, out)  # block-cyclic over the last chunk
    # Invert the block-cyclic permutation within the last chunk.
    local_pos = last_idx - kv_actual_last
    inv = torch.empty(chunk_global, dtype=torch.long)
    inv[local_pos] = torch.arange(chunk_global)
    full = full_bc[:, :, inv, :]

    passing, pcc = comp_pcc(reference, full, PCC)
    logger.info(f"ring_joint CACHE-READ n_chunks={n_chunks} chunk_global={chunk_global}: pcc={pcc}")
    assert passing, f"cache-read PCC fail: {pcc}"
