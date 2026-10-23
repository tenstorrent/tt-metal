# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GQA chunked-KV cache, TP=4 × SP=8, on (8,4) — write + read-back PCC vs torch golden.

Verifies that DeepSeek's chunked-KV machinery works for M3's plain GQA (separate 4-head K and V),
NOT just MLA's single latent head. Under TP=4 each chip holds 1 KV head (4 heads / 4 cols), so the
per-chip cache is [1, 1, seq_local, head_dim] — exactly init_kvpe_cache's shape — and the chunk
writer `ttnn.experimental.deepseek_prefill.update_padded_kv_cache` should be head-agnostic (it
block-cyclic-shards the SEQUENCE on the SP axis; TP heads are orthogonal, one per column).

Layout: K cache + V cache each [1, NKV, seq_cache, 128], heads TP-sharded (cols → 1/chip), sequence
SP-sharded block-cyclic (rows), DRAM NdShard. Mirrors the DS verify recipe
(tests/op_unit_tests/test_deepseek_prefill_update_padded_kv_cache.py): write chunks → read back via
ConcatMesh2dToTensor(dims=(seq, heads)) → invert the block-cyclic layout → PCC vs the natural-order
torch K/V. If this passes, update_padded_kv_cache is reusable for the GQA dense chunked-KV cache.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

from ..test_factory import parametrize_mesh_with_fabric

NKV, HEAD_DIM = 4, 128


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("n_chunks,chunk_local", [(2, 32)], ids=["2x256"])
def test_kv_cache_gqa_sp(mesh_device, device_params, n_chunks, chunk_local, reset_seeds):
    """Write GQA K/V chunks into a TP+SP chunked-KV cache, read back, PCC vs natural-order torch."""
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4), "TP=4 x SP=8 layout expected"
    sp, tp = rows, cols
    sp_axis, tp_axis = 0, 1

    C = chunk_local
    chunk_global = sp * C
    cache_global = n_chunks * chunk_global
    cache_tokens_per_dev = cache_global // sp
    cum_total = cache_global  # chunk-aligned: every position is real

    torch.manual_seed(0)
    # Natural-order per-head K/V for all prefilled tokens.
    sent_k = torch.randn(NKV, cum_total, HEAD_DIM, dtype=torch.bfloat16)
    sent_v = torch.randn(NKV, cum_total, HEAD_DIM, dtype=torch.bfloat16)

    # K and V caches: per-chip [1, 1, seq_local, HEAD_DIM] (TP gives 1 head/chip), DRAM NdShard.
    cache_k = init_kvpe_cache(HEAD_DIM, mesh_device, cache_global, list(mesh_device.shape), sp_axis, 1)
    cache_v = init_kvpe_cache(HEAD_DIM, mesh_device, cache_global, list(mesh_device.shape), sp_axis, 1)

    # Chunk input sharding: sequence on SP rows (dim 2), heads on TP cols (dim 1).
    in_dims = [None, None]
    in_dims[sp_axis] = 2
    in_dims[tp_axis] = 1

    def write_chunk(cache, sent, kv_actual):
        # Block-cyclic chip-concat order of this chunk's global positions (mirrors the writer).
        positions = rotated_chip_positions(kv_actual, sp, C)
        idx = torch.tensor([positions[c][r] for c in range(sp) for r in range(C)], dtype=torch.long)
        chunk = sent[:, idx, :].reshape(1, NKV, chunk_global, HEAD_DIM)  # [1, NKV, chunk_global, 128]
        tt_in = ttnn.from_torch(
            chunk,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
        )
        ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
            cache, tt_in, slot_idx=0, layer_idx=0, num_layers=1, kv_actual_global=kv_actual, cluster_axis=sp_axis
        )

    for c in range(n_chunks):
        kv_actual = c * chunk_global
        write_chunk(cache_k, sent_k, kv_actual)
        write_chunk(cache_v, sent_v, kv_actual)
    ttnn.synchronize_device(mesh_device)

    # Read back: concat sequence over SP rows (dim 2) + heads over TP cols (dim 1).
    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 1

    def readback(cache):
        return ttnn.to_torch(
            cache,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat_dims), mesh_shape=(rows, cols)),
        ).to(
            torch.bfloat16
        )  # [1, NKV, cache_global, HEAD_DIM]

    host_k, host_v = readback(cache_k), readback(cache_v)

    # Invert the block-cyclic layout: natural pos p -> (chip, local row) -> dim-2 index in the cache.
    p = torch.arange(cum_total)
    chip = (p % chunk_global) // C
    local_row = (p // chunk_global) * C + (p % C)
    dim2_idx = chip * cache_tokens_per_dev + local_row

    worst = 1.0
    for h in range(NKV):
        ok_k, pcc_k = comp_pcc(sent_k[h], host_k[0, h, dim2_idx, :], 0.99)
        ok_v, pcc_v = comp_pcc(sent_v[h], host_v[0, h, dim2_idx, :], 0.99)
        logger.info(f"head {h}: K pcc={pcc_k} V pcc={pcc_v}")
        assert ok_k and ok_v, f"head {h} cache mismatch: K={pcc_k} V={pcc_v}"
    logger.info(f"GQA TP=4 x SP=8 chunked-KV cache ({n_chunks} chunks, {cache_global} tok): all heads PCC>=0.99")
