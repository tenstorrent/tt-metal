# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ring_joint cache-read on (8,4), SP=8 x TP=4: write n_chunks of K/V into the GQA chunked-KV cache (block-cyclic,
SP-sharded), then run the last chunk's queries through dense_sp_attention with the host kv_cache_batch_idx /
kv_actual_isl so ring_joint reads the accumulated prefix from the cache, vs a full-causal GQA torch golden.
Grouped V (cache stays NKV heads, 1/chip). The chunk / gather helpers live in ring_joint_cache_read_helpers.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.minimax_m3.tt.attention.dense_sp import dense_sp_attention
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric
from .ring_joint_cache_read_helpers import (
    HEAD_DIM,
    NKV,
    NQ,
    PCC_BF8_CACHE,
    SP_AXIS,
    gather_chunk,
    make_kv_chunk,
    make_q_chunk,
    sdpa_configs,
    torch_gqa_causal,
)


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
    sp, sp_axis = rows, SP_AXIS
    C = chunk_local
    chunk_global = sp * C  # 256
    cache_global = n_chunks * chunk_global  # 512
    kv_actual_last = (n_chunks - 1) * chunk_global  # prefix length before the last chunk (256)

    torch.manual_seed(0)
    q = torch.randn(1, NQ, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    k = torch.randn(1, NKV, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    v = torch.randn(1, NKV, cache_global, HEAD_DIM, dtype=torch.bfloat16) * 0.1
    ref_full = torch_gqa_causal(q.float(), k.float(), v.float())  # [1, NQ, cache_global, HD]
    ref = ref_full[:, :, kv_actual_last:, :]  # golden for the LAST chunk's query positions

    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    # --- write all chunks into the GQA chunked-KV cache (block-cyclic) ---
    cache_k = init_kvpe_cache(HEAD_DIM, mesh_device, cache_global, list(mesh_device.shape), sp_axis, 1)
    cache_v = init_kvpe_cache(HEAD_DIM, mesh_device, cache_global, list(mesh_device.shape), sp_axis, 1)

    def write(cache, src, kv_actual):
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache,
            make_kv_chunk(src, kv_actual, mesh_device, C),
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
    tt_q = make_q_chunk(q, kv_actual_last, mesh_device, C)
    prog, kcfg = sdpa_configs(mesh_device)

    # dense_sp_attention writes the LAST chunk into the cache, then ring_joint cache-read: K/V come
    # from the cache (kv_cache_batch_idx=slot 0); kv_actual_isl = prefix before the last chunk,
    # logical_n = full valid prefix (Q attends causally over [0:logical_n]).
    out = dense_sp_attention(
        tt_q,
        cache_k,
        cache_v,
        make_kv_chunk(k[0], kv_actual_last, mesh_device, C),
        make_kv_chunk(v[0], kv_actual_last, mesh_device, C),
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

    full = gather_chunk(out, kv_actual_last, mesh_device, C)  # natural order over positions kv_actual_last:cache_global

    passing, pcc = comp_pcc(ref, full, PCC_BF8_CACHE)
    logger.info(f"ring_joint CACHE-READ SP=8 x TP=4 vs ref: pcc={pcc}")
    assert passing, f"cache-read PCC fail: {pcc}"
