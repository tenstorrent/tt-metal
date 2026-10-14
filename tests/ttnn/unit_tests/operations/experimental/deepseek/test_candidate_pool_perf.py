# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device-time profile of the V4.1 two-level select at 16K and 128K compressed keys.

``test_candidate_pool_profile`` (one device), per context length and iteration, in this order:

1. ``fused_lightning_select_kv`` over the full cache (what a consumer layer costs without a pool),
2. the same op in candidate source mode (layer 20),
3. ``build_candidate_pool``,
4. ``fused_lightning_select_kv`` on the pool, gathering through its row table (layers 24, 28, 32, 36).

``test_key_split_profile`` (a 1x4 submesh of the 8x4 mesh): steps 1 and 4 again, then each of them
split over the 4 devices (``num_parts=4`` select, ``all_gather``, ``merge_select_kv``).

Each length runs twice; the first fills the program cache, so read the second (after the
``<length>-warm`` signpost). Run::

    python -m tracy -p -r -v -m pytest tests/ttnn/unit_tests/operations/experimental/deepseek/test_candidate_pool_perf.py
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from tracy import signpost

from tests.ttnn.unit_tests.operations.experimental.deepseek.test_fused_lightning_select_kv import (
    COMPRESS_RATE,
    PARENT_MESH,
    V4_HEAD_DIM,
    V4_INDEX_HEAD_DIM,
    V4_INDEX_HEADS,
    V4_INDEX_TOPK,
    V4_KV_HEADS,
    _paged_inputs,
    _replicated_rm_hs,
    _rm,
    _split_select,
    _tile,
    _tp_submesh,
)

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")

CANDIDATE_TOPK_BLOCKS = 2048
CANDIDATE_BLOCK_SIZE = 8
PAGE_BLOCK_SIZE = 64


@pytest.mark.parametrize("num_keys", [16 * 1024, 128 * 1024], ids=["16k", "128k"])
def test_candidate_pool_profile(device, num_keys) -> None:
    torch.manual_seed(0)
    heads, dim, k = V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, V4_INDEX_TOPK
    cur_pos = _rm(torch.tensor([num_keys * COMPRESS_RATE - 1], dtype=torch.int32), device, ttnn.int32)
    query = _replicated_rm_hs(torch.randn(1, heads, 1, dim), device)
    weights = _replicated_rm_hs(torch.randn(1, 1, 1, heads) * dim**-0.5, device)
    kv_cache, page_table = _paged_inputs(1, num_keys, PAGE_BLOCK_SIZE, V4_KV_HEADS, V4_HEAD_DIM, seed=0)
    # The model's caches: interleaved TILE index keys, row-major interleaved kv rows.
    key_tt = _tile(torch.randn(page_table.numel(), 1, PAGE_BLOCK_SIZE, dim), device)
    kv_tt = _rm(kv_cache, device, ttnn.bfloat16)
    page_table_tt = _rm(page_table, device, ttnn.int32)
    pool_blocks = CANDIDATE_TOPK_BLOCKS * CANDIDATE_BLOCK_SIZE // PAGE_BLOCK_SIZE
    identity = _rm(torch.arange(pool_blocks, dtype=torch.int32).reshape(1, pool_blocks), device, ttnn.int32)

    for iteration in ("cold", "warm"):
        signpost(f"{num_keys}-{iteration}")
        ttnn.experimental.deepseek.fused_lightning_select_kv(query, key_tt, weights, kv_tt, page_table_tt, cur_pos, k)
        ttnn.synchronize_device(device)
        _, _, blocks, pool_count = ttnn.experimental.deepseek.fused_lightning_select_kv(
            query,
            key_tt,
            weights,
            kv_tt,
            page_table_tt,
            cur_pos,
            k,
            candidate_topk_blocks=CANDIDATE_TOPK_BLOCKS,
            candidate_block_size=CANDIDATE_BLOCK_SIZE,
        )
        ttnn.synchronize_device(device)
        pool_keys, pool_rows = ttnn.experimental.deepseek.build_candidate_pool(
            blocks, pool_count, key_tt, page_table_tt, CANDIDATE_BLOCK_SIZE
        )
        ttnn.synchronize_device(device)
        ttnn.experimental.deepseek.fused_lightning_select_kv(
            query, pool_keys, weights, kv_tt, identity, cur_pos, k, num_keys_tensor=pool_count, row_table=pool_rows
        )
        ttnn.synchronize_device(device)

    count = int(ttnn.to_torch(pool_count).reshape(-1)[0])
    logger.info(f"{num_keys} keys: pool of {count}")
    assert count == min(num_keys, CANDIDATE_TOPK_BLOCKS * CANDIDATE_BLOCK_SIZE)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("num_keys", [16 * 1024, 128 * 1024], ids=["16k", "128k"])
def test_key_split_profile(mesh_device, num_keys) -> None:
    submesh = _tp_submesh(mesh_device)
    replicate = ttnn.ReplicateTensorToMesh(submesh)
    torch.manual_seed(0)
    heads, dim, k = V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, V4_INDEX_TOPK
    cur_pos = _rm(
        torch.tensor([num_keys * COMPRESS_RATE - 1], dtype=torch.int32), submesh, ttnn.int32, mesh_mapper=replicate
    )
    query = _replicated_rm_hs(torch.randn(1, heads, 1, dim), submesh, mesh_mapper=replicate)
    weights = _replicated_rm_hs(torch.randn(1, 1, 1, heads) * dim**-0.5, submesh, mesh_mapper=replicate)
    kv_cache, page_table = _paged_inputs(1, num_keys, PAGE_BLOCK_SIZE, V4_KV_HEADS, V4_HEAD_DIM, seed=0)
    key_tt = _tile(torch.randn(page_table.numel(), 1, PAGE_BLOCK_SIZE, dim), submesh, mesh_mapper=replicate)
    kv_tt = _rm(kv_cache, submesh, ttnn.bfloat16, mesh_mapper=replicate)
    page_table_tt = _rm(page_table, submesh, ttnn.int32, mesh_mapper=replicate)
    pool_blocks = CANDIDATE_TOPK_BLOCKS * CANDIDATE_BLOCK_SIZE // PAGE_BLOCK_SIZE
    identity = _rm(
        torch.arange(pool_blocks, dtype=torch.int32).reshape(1, pool_blocks), submesh, ttnn.int32, mesh_mapper=replicate
    )
    _, _, blocks, pool_count = ttnn.experimental.deepseek.fused_lightning_select_kv(
        query,
        key_tt,
        weights,
        kv_tt,
        page_table_tt,
        cur_pos,
        k,
        candidate_topk_blocks=CANDIDATE_TOPK_BLOCKS,
        candidate_block_size=CANDIDATE_BLOCK_SIZE,
    )
    pool_keys, pool_rows = ttnn.experimental.deepseek.build_candidate_pool(
        blocks, pool_count, key_tt, page_table_tt, CANDIDATE_BLOCK_SIZE
    )
    ttnn.synchronize_device(submesh)

    for iteration in ("cold", "warm"):
        signpost(f"split-{num_keys}-{iteration}")
        ttnn.experimental.deepseek.fused_lightning_select_kv(query, key_tt, weights, kv_tt, page_table_tt, cur_pos, k)
        ttnn.synchronize_device(submesh)
        ttnn.experimental.deepseek.fused_lightning_select_kv(
            query, pool_keys, weights, kv_tt, identity, cur_pos, k, num_keys_tensor=pool_count, row_table=pool_rows
        )
        ttnn.synchronize_device(submesh)
        _split_select(submesh, query, key_tt, weights, kv_tt, page_table_tt, cur_pos, k)
        ttnn.synchronize_device(submesh)
        _split_select(
            submesh,
            query,
            pool_keys,
            weights,
            kv_tt,
            identity,
            cur_pos,
            k,
            num_keys_tensor=pool_count,
            row_table=pool_rows,
        )
        ttnn.synchronize_device(submesh)
