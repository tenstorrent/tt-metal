# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``ttnn.experimental.deepseek.fused_lightning_select_kv``.

The op fuses the DeepSeek V4-Flash CSA lightning indexer
(``DeepSeekV4Indexer.select`` in models/experimental/deepseek_v4_flash/tt/attention_csa.py)
with the paged KV gather that follows it:

    scores[t]  = sum_h ReLU(q[h, Sq-1] . key_cache[t]) * w[Sq-1, h]
    scores[t]  = -inf for t >= valid_length
    indices    = topk(scores, k)                      (descending)
    out[b, :, j] = kv_cache[page_table[b, idx // block_size], :, idx % block_size]

The unfused model path puts the one real query on row 31 of a tile-padded 32-row
block. The fused op instead takes ``query`` ROW_MAJOR HEIGHT_SHARDED in L1 with one
full ``[B * Hi * Sq, D]`` replica per core (the ``matmul_decode`` rm_hs layout), so
the tests hand it just the real row. The op is decode-only: ``B == 1`` and ``Sq == 1``.

Run::

    pytest tests/ttnn/unit_tests/operations/experimental/deepseek/test_fused_lightning_select_kv.py
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")

TILE = 32
COMPRESS_RATE = 4

# DeepSeek V4-Flash CSA indexer / attention dimensions.
V4_INDEX_HEADS = 64
V4_INDEX_HEAD_DIM = 128
V4_INDEX_TOPK = 512
V4_KV_HEADS = 1
V4_HEAD_DIM = 512


def _tile(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _block_sharded_key_cache(key_cache: torch.Tensor, device) -> ttnn.Tensor:
    """Tiled DRAM ND-sharded ``key_cache`` with one ``[1, 1, block_size, D]`` block per shard.

    Each block is contiguous in a single DRAM bank, so the op reads a whole block in one NoC read.
    """
    _, heads, block_size, dim = key_cache.shape
    num_banks = device.dram_grid_size().x
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    nd_shard = ttnn.NdShardSpec(ttnn.Shape([1, heads, block_size, dim]), dram_grid, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.from_torch(
        key_cache.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.BufferType.DRAM, nd_shard),
    )


def _rm(t: torch.Tensor, device, dtype) -> ttnn.Tensor:
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _full_grid(device) -> ttnn.CoreRangeSet:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})


def _replicated_rm_hs(x: torch.Tensor, device) -> ttnn.Tensor:
    """``[B, H, S, W]`` -> ROW_MAJOR HEIGHT_SHARDED L1 with one full replica per core of the grid.

    Mirrors ``LinearDecode.to_replicated_rm_hs_activation``: repeat once per core along the
    outermost axis, then height-shard with shard ``[B * H * S, W]``. Used for both
    ``query`` ``[1, Hi, 1, D]`` (shard ``[Hi, D]``) and ``head_weights`` ``[1, 1, 1, Hi]``
    (shard ``[1, Hi]``). The logical batch becomes ``num_cores``.
    """
    grid = _full_grid(device)
    batch, heads, rows, width = x.shape
    mem_cfg = ttnn.create_sharded_memory_config(
        (batch * heads * rows, width),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    replicated = x.repeat(grid.num_cores(), 1, 1, 1)
    print(f"q replicated: {replicated.shape}, mem_cfg: {mem_cfg}")
    return ttnn.from_torch(
        replicated.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem_cfg,
    )


def _as_u32(t: torch.Tensor) -> torch.Tensor:
    """Host view of a uint32 device tensor. ``to_torch`` may return the sign bit as int32."""
    return t.to(torch.int64) & 0xFFFFFFFF


def _reference_scores(query_row: torch.Tensor, keys: torch.Tensor, weights_row: torch.Tensor) -> torch.Tensor:
    """``sum_h ReLU(q_h . k_t) * w_h`` over every key. ``query_row`` is ``[Hi, D]``, ``keys`` ``[T, D]``."""
    dots = torch.relu(query_row.float() @ keys.float().T)
    return (dots * weights_row.float()[:, None]).sum(dim=0)


def _paged_gather(kv_cache: torch.Tensor, page_table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """``kv_cache`` ``[num_blocks, Hkv, block_size, Dh]``, ``indices`` ``[B, k]`` -> ``[B, Hkv, k, Dh]``."""
    block_size = kv_cache.shape[2]
    rows = []
    for b in range(indices.shape[0]):
        idx = indices[b].to(torch.int64)
        blocks = page_table[b].to(torch.int64)[idx // block_size]
        rows.append(kv_cache[blocks, :, idx % block_size, :].permute(1, 0, 2))
    return torch.stack(rows)


def _paged_inputs(batch: int, length: int, block_size: int, kv_heads: int, head_dim: int, seed: int):
    """A shuffled block pool so logical block ``i`` of a user is not physical block ``i``."""
    g = torch.Generator().manual_seed(seed)
    blocks_per_user = length // block_size
    num_blocks = batch * blocks_per_user
    print(f"num_blocks: {num_blocks}, length: {length}, block_size: {block_size}")
    kv_cache = torch.randn(num_blocks, kv_heads, block_size, head_dim, generator=g).to(torch.bfloat16)
    page_table = torch.randperm(num_blocks, generator=g).reshape(batch, blocks_per_user).to(torch.int32)
    return kv_cache, page_table


def _page_keys(keys: torch.Tensor, page_table: torch.Tensor, block_size: int) -> torch.Tensor:
    """Logical ``[B, 1, T, D]`` keys -> ``[num_blocks, 1, block_size, D]`` cache laid out by ``page_table``."""
    batch, heads, length, dim = keys.shape
    blocks_per_user = length // block_size
    key_cache = torch.zeros(page_table.numel(), heads, block_size, dim, dtype=keys.dtype)
    for b in range(batch):
        logical = keys[b].reshape(heads, blocks_per_user, block_size, dim).permute(1, 0, 2, 3)
        key_cache[page_table[b].to(torch.int64)] = logical
    return key_cache


def _unpage_keys(key_cache: torch.Tensor, page_table: torch.Tensor) -> torch.Tensor:
    """``[num_blocks, 1, block_size, D]`` cache -> logical ``[B, 1, T, D]`` keys, inverse of ``_page_keys``."""
    _, heads, block_size, dim = key_cache.shape
    batch, blocks_per_user = page_table.shape
    logical = key_cache[page_table.to(torch.int64)]  # [B, blocks_per_user, heads, block_size, D]
    return logical.permute(0, 2, 1, 3, 4).reshape(batch, heads, blocks_per_user * block_size, dim)


def _run_op(device, query, key_cache, weights, kv_cache, page_table, cur_pos, valid, k):
    """``query`` ``[1, Hi, 1, D]`` and ``weights`` ``[1, 1, 1, Hi]``: the one real row the op scores.

    ``key_cache`` is ``[num_blocks, 1, block_size, D]``, paged through ``page_table`` like ``kv_cache``.
    """
    out, _ = _run_op_with_scores(device, query, key_cache, weights, kv_cache, page_table, cur_pos, valid, k)
    return out


def _run_op_with_scores(device, query, key_cache, weights, kv_cache, page_table, cur_pos, valid, k):
    """Same as ``_run_op`` but also returns the ``[T_max]`` fp32 index scores."""
    out, scores = ttnn.experimental.deepseek.fused_lightning_select_kv(
        _replicated_rm_hs(query, device),
        _block_sharded_key_cache(key_cache, device),
        _replicated_rm_hs(weights, device),
        _rm(kv_cache, device, ttnn.bfloat16),
        _rm(page_table, device, ttnn.int32),
        _rm(cur_pos, device, ttnn.int32),
        k,
        valid_length_tensor=_rm(torch.tensor([valid], dtype=torch.int32), device, ttnn.uint32),
    )
    assert list(out.shape) == [page_table.shape[0], kv_cache.shape[1], k, kv_cache.shape[3]]
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    assert list(scores.shape) == [page_table.shape[0], 1, 1, page_table.shape[1] * key_cache.shape[2]]
    assert scores.dtype == ttnn.float32
    return ttnn.to_torch(out), ttnn.to_torch(scores).reshape(-1)


@pytest.mark.parametrize(
    "heads, dim, length, block_size, num_keys",
    [
        (32, 128, 64, 64, 24),
        (V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, 128 * 1024, 64, 2048),
        (V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, 2048, 64, 1000),
    ],
    ids=["one_block", "v4_flash_full", "v4_flash_partial"],
)
def test_index_scores_match_reference(device, heads, dim, length, block_size, num_keys) -> None:
    """``scores[t] = sum_h ReLU(q_h . k_t) * w_h`` for every valid key, read through the page table."""
    torch.manual_seed(0)
    batch = 1
    cur_pos = torch.tensor([num_keys * COMPRESS_RATE - 1], dtype=torch.int32)

    # bf16-round on host so the reference sees exactly what the device reads.
    query = torch.randn(batch, heads, 1, dim).to(torch.bfloat16).float()
    # Signed weights: ReLU must happen before the head weighting.
    weights = (torch.randn(batch, 1, 1, heads) * dim**-0.5).to(torch.bfloat16).float()
    kv_cache, page_table = _paged_inputs(batch, length, block_size, V4_KV_HEADS, V4_HEAD_DIM, seed=2)
    key_cache = torch.randn(page_table.numel(), 1, block_size, dim).to(torch.bfloat16).float()
    keys = _unpage_keys(key_cache, page_table)

    expected = _reference_scores(query[0, :, 0], keys[0, 0], weights[0, 0, 0])[:num_keys]

    _, got = _run_op_with_scores(
        device, query, key_cache, weights, kv_cache, page_table, cur_pos, num_keys, k=min(16, num_keys)
    )
    got = got[:num_keys].float()

    # The score matmul runs custom_mm, which is LoFi-only, so the bound is LoFi-level rather than
    # the fp32 reference's.
    passing, message = comp_pcc(expected, got, pcc=0.999)
    max_err = (expected - got).abs().max().item()
    rel_err = max_err / expected.abs().max().item()
    logger.info(f"index scores: pcc {message}, max abs err {max_err:.4e} ({rel_err:.2%} of max |score|)")
    assert passing, f"index scores diverge from the reference: {message}"


def test_select_kv_respects_closed_windows(device) -> None:
    """Exact golden: well-separated scores, with the largest key past ``valid_length``.

    Key ``t`` scores ``t + 1`` for the real query row, so the expected top-k is the
    last ``k`` closed windows in descending order. Key 50 is set far above the
    rest but sits past ``valid_length`` and must not be selected.
    """
    batch, heads, dim, length, k, block_size = 1, 32, 128, 64, 16, 64
    cur_pos = torch.full((batch,), 24 * COMPRESS_RATE - 1, dtype=torch.int32)
    valid = int(cur_pos[0] + 1) // COMPRESS_RATE

    query = torch.zeros(batch, heads, TILE, dim)
    query[:, :, TILE - 1, 0] = 1.0
    weights = torch.zeros(batch, 1, TILE, heads)
    weights[:, 0, TILE - 1, :] = 1.0 / heads
    keys = torch.zeros(batch, 1, length, dim)
    keys[:, 0, :, 0] = torch.arange(1, length + 1, dtype=torch.float32)
    keys[:, 0, 50, 0] = 1000.0

    kv_cache, page_table = _paged_inputs(batch, length, block_size, kv_heads=1, head_dim=64, seed=0)
    print(f"page_table: {page_table}")
    expected_idx = []
    for b in range(batch):
        scores = _reference_scores(query[b, :, TILE - 1], keys[b, 0], weights[b, 0, TILE - 1])
        scores[valid:] = float("-inf")
        expected_idx.append(torch.topk(scores, k).indices)
    expected_idx = torch.stack(expected_idx)
    assert 50 not in expected_idx.tolist()
    expected = _paged_gather(kv_cache, page_table, expected_idx)

    real_row = slice(TILE - 1, TILE)
    key_cache = _page_keys(keys, page_table, block_size)
    got = _run_op(
        device, query[:, :, real_row], key_cache, weights[:, :, real_row], kv_cache, page_table, cur_pos, valid, k
    )

    assert torch.equal(got.to(torch.bfloat16), expected), "gathered rows differ from the exact top-k golden"


@pytest.mark.parametrize(
    "length, k, block_size",
    [
        (2048, 64, 64),
        (2048, V4_INDEX_TOPK, 64),
    ],
    ids=["small", "v4_flash"],
)
def test_select_kv_matches_indexer_pipeline(device, length, k, block_size) -> None:
    """V4-Flash shapes: the fused op must match the unfused model path.

    The model path is ``indexer_score_dsa`` -> ``topk_large_indices`` on device,
    exactly as ``DeepSeekV4Indexer.select`` runs it, followed by a torch paged
    gather of the selected rows. Using the device indices (not torch top-k) keeps
    bf16 score ties from turning into spurious mismatches.
    """
    torch.manual_seed(0)
    batch, heads, dim = 1, V4_INDEX_HEADS, V4_INDEX_HEAD_DIM
    cur_pos = torch.tensor([length * COMPRESS_RATE - 1], dtype=torch.int32)
    valid = int(cur_pos[0] + 1) // COMPRESS_RATE
    assert valid >= k

    query = torch.zeros(batch, heads, TILE, dim)
    query[:, :, TILE - 1] = torch.randn(batch, heads, dim)
    weights = torch.zeros(batch, 1, TILE, heads)
    weights[:, 0, TILE - 1] = torch.randn(batch, heads).abs() * dim**-0.5
    kv_cache, page_table = _paged_inputs(batch, length, block_size, V4_KV_HEADS, V4_HEAD_DIM, seed=1)
    print(f"q : {query.shape}")
    print(f"page_table: {page_table}")
    key_cache = torch.randn(page_table.numel(), 1, block_size, dim)
    # The unfused pipeline scores contiguous keys, so hand it the logical view of the same cache.
    keys = _unpage_keys(key_cache, page_table)
    valid_length = _rm(torch.tensor([valid], dtype=torch.int32), device, ttnn.uint32)
    scores = ttnn.experimental.indexer_score_dsa(
        _tile(query, device), _tile(keys, device), _tile(weights, device), chunk_start_idx=length - TILE
    )
    topk = ttnn.experimental.topk_large_indices(scores, k=k, valid_length_tensor=valid_length)
    row = ttnn.slice(topk, [0, 0, TILE - 1, 0], [batch, 1, TILE, k])
    pipeline_idx = _as_u32(ttnn.to_torch(row)).reshape(batch, k)
    expected = _paged_gather(kv_cache, page_table, pipeline_idx)

    # The pipeline needs a tile-padded 32-row block; the fused op takes just the real row.
    real_row = slice(TILE - 1, TILE)
    got = _run_op(
        device, query[:, :, real_row], key_cache, weights[:, :, real_row], kv_cache, page_table, cur_pos, valid, k
    )

    matches = (got.to(torch.bfloat16) == expected).all(dim=-1)
    logger.info(f"fused_lightning_select_kv: {int(matches.sum())}/{matches.numel()} rows match the indexer pipeline")
    passing, message = comp_pcc(expected.float(), got.float(), pcc=0.9999)
    assert passing, f"fused op diverges from the indexer pipeline: {message}"
    assert bool(matches.all()), "fused op selected different rows than the indexer pipeline"
