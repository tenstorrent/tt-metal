# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``ttnn.experimental.deepseek.fused_lightning_select_kv``.

The op fuses the DeepSeek V4-Flash CSA lightning indexer
(``DeepSeekV4Indexer.select`` in models/experimental/deepseek_v4_flash/tt/decode/attention_csa.py)
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


def _tile(t: torch.Tensor, device, mesh_mapper=None) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
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


def _rm(t: torch.Tensor, device, dtype, mesh_mapper=None) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def _full_grid(device) -> ttnn.CoreRangeSet:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})


def _replicated_rm_hs(x: torch.Tensor, device, mesh_mapper=None) -> ttnn.Tensor:
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
    return ttnn.from_torch(
        replicated.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mem_cfg,
        mesh_mapper=mesh_mapper,
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
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert list(scores.shape) == [page_table.shape[0], 1, 1, page_table.shape[1] * key_cache.shape[2]]
    assert scores.dtype == ttnn.float32
    return ttnn.to_torch(out), ttnn.to_torch(scores).reshape(-1)


@pytest.mark.parametrize(
    "heads, dim, length, block_size, num_keys",
    [
        (32, 128, 64, 64, 24),
        (V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, 128 * 1024, 64, 512),
        (V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, 2048, 64, 1000),
    ],
    ids=["one_block", "v4_flash_full", "v4_flash_partial"],
)
def test_index_scores_match_reference(device, heads, dim, length, block_size, num_keys) -> None:
    """``scores[t] = sum_h ReLU(q_h . k_t) * w_h`` for every valid key, read through the page table."""
    torch.manual_seed(0)
    batch = 1
    cur_pos = torch.tensor([length * COMPRESS_RATE - 1], dtype=torch.int32)

    # bf16-round on host so the reference sees exactly what the device reads.
    query = torch.randn(batch, heads, 1, dim).to(torch.bfloat16).float()
    # Signed weights: ReLU must happen before the head weighting.
    weights = (torch.randn(batch, 1, 1, heads) * dim**-0.5).to(torch.bfloat16).float()
    kv_cache, page_table = _paged_inputs(batch, length, block_size, V4_KV_HEADS, V4_HEAD_DIM, seed=2)
    key_cache = torch.randn(page_table.numel(), 1, block_size, dim).to(torch.bfloat16).float()
    keys = _unpage_keys(key_cache, page_table)

    expected = _reference_scores(query[0, :, 0], keys[0, 0], weights[0, 0, 0])[:num_keys]

    _, got = _run_op_with_scores(
        device, query, key_cache, weights, kv_cache, page_table, cur_pos, num_keys, k=min(V4_INDEX_TOPK, num_keys)
    )
    got = got[:num_keys].float()

    # The score matmul runs custom_mm, which is LoFi-only, so the bound is LoFi-level rather than
    # the fp32 reference's.
    passing, message = comp_pcc(expected, got, pcc=0.999)
    max_err = (expected - got).abs().max().item()
    rel_err = max_err / expected.abs().max().item()
    logger.info(f"index scores: pcc {message}, max abs err {max_err:.4e} ({rel_err:.2%} of max |score|)")
    assert passing, f"index scores diverge from the reference: {message}"


@pytest.mark.parametrize(
    "heads, dim, length, block_size, num_keys",
    [(V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, 128 * 1024, 64, 512)],
    ids=["v4_flash_full"],
)
def test_select_kv_rows_match_reference(device, heads, dim, length, block_size, num_keys) -> None:
    """At least 95% of the gathered kv rows are rows of the reference top-k, in any order.

    Same inputs as ``test_index_scores_match_reference[v4_flash_full]``. The op's top-k is
    histogram-based and bf16-scored, so near-threshold keys may differ from torch top-k.
    """
    torch.manual_seed(0)
    batch = 1
    k = min(V4_INDEX_TOPK, num_keys)
    cur_pos = torch.tensor([num_keys * COMPRESS_RATE - 1], dtype=torch.int32)

    query = torch.randn(batch, heads, 1, dim).to(torch.bfloat16).float()
    weights = (torch.randn(batch, 1, 1, heads) * dim**-0.5).to(torch.bfloat16).float()
    kv_cache, page_table = _paged_inputs(batch, length, block_size, V4_KV_HEADS, V4_HEAD_DIM, seed=2)
    key_cache = torch.randn(page_table.numel(), 1, block_size, dim).to(torch.bfloat16).float()
    keys = _unpage_keys(key_cache, page_table)

    scores = _reference_scores(query[0, :, 0], keys[0, 0], weights[0, 0, 0])[:num_keys]
    expected_idx = torch.topk(scores, k).indices.reshape(batch, k)
    expected = _paged_gather(kv_cache, page_table, expected_idx)[0, 0]  # [k, Dh]

    got = _run_op(device, query, key_cache, weights, kv_cache, page_table, cur_pos, num_keys, k)
    got = got.to(torch.bfloat16)[0, 0]  # [k, Dh]

    # A got row matches if it equals some reference row; each reference row is claimed at most once.
    equal = (got[:, None, :] == expected[None, :, :]).all(dim=-1)  # [k, k]
    claimed = torch.zeros(k, dtype=torch.bool)
    num_matched = 0
    for i in range(k):
        candidates = (equal[i] & ~claimed).nonzero()
        if candidates.numel() != 0:
            claimed[candidates[0, 0]] = True
            num_matched += 1
    match_ratio = num_matched / k
    logger.info(f"kv rows: {num_matched}/{k} ({match_ratio:.2%}) match the reference top-k")
    assert match_ratio >= 0.95, f"only {num_matched}/{k} ({match_ratio:.2%}) gathered kv rows match the reference"


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
    # The op emits the selected rows in ascending key order.
    expected_idx = torch.stack(expected_idx).sort(dim=-1).values
    assert 50 not in expected_idx.tolist()
    expected = _paged_gather(kv_cache, page_table, expected_idx)

    real_row = slice(TILE - 1, TILE)
    key_cache = _page_keys(keys, page_table, block_size)
    got = _run_op(
        device, query[:, :, real_row], key_cache, weights[:, :, real_row], kv_cache, page_table, cur_pos, valid, k
    )

    assert torch.equal(got.to(torch.bfloat16), expected), "gathered rows differ from the exact top-k golden"


@pytest.mark.parametrize("sliding_window, tail_rows", [(128, 0), (128, 32)], ids=["exact_fit", "with_tail"])
def test_select_kv_writes_into_output_tensor_at_offset(device, sliding_window, tail_rows) -> None:
    """``output_tensor`` + ``output_row_offset`` at V4-Flash CSA shapes.

    The selected rows land in a preallocated ``[1, 1, sliding_window + k + tail_rows, Dh]``
    buffer at row ``sliding_window`` (``exact_fit`` is the 640-row CSA combined buffer). The
    ring rows before it and any tail rows after it must keep their prior contents. As in
    ``test_select_kv_rows_match_reference``, the histogram top-k may pick different
    near-threshold keys than torch top-k, so at least 95% of the rows must match, in any order.
    """
    torch.manual_seed(0)
    batch, heads, dim, length, block_size = 1, V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, 128 * 1024, 64
    k, head_dim = V4_INDEX_TOPK, V4_HEAD_DIM
    num_keys = length
    cur_pos = torch.tensor([num_keys * COMPRESS_RATE - 1], dtype=torch.int32)

    query = torch.randn(batch, heads, 1, dim).to(torch.bfloat16).float()
    weights = (torch.randn(batch, 1, 1, heads) * dim**-0.5).to(torch.bfloat16).float()
    kv_cache, page_table = _paged_inputs(batch, length, block_size, V4_KV_HEADS, head_dim, seed=2)
    key_cache = torch.randn(page_table.numel(), 1, block_size, dim).to(torch.bfloat16).float()
    keys = _unpage_keys(key_cache, page_table)

    scores = _reference_scores(query[0, :, 0], keys[0, 0], weights[0, 0, 0])[:num_keys]
    expected_idx = torch.topk(scores, k).indices.reshape(batch, k)
    expected = _paged_gather(kv_cache, page_table, expected_idx)[0, 0]  # [k, Dh]

    total_rows = sliding_window + k + tail_rows
    prior = torch.randn(batch, V4_KV_HEADS, total_rows, head_dim).to(torch.bfloat16)
    out_buf = ttnn.from_torch(
        prior, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    out, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
        _replicated_rm_hs(query, device),
        _block_sharded_key_cache(key_cache, device),
        _replicated_rm_hs(weights, device),
        _rm(kv_cache, device, ttnn.bfloat16),
        _rm(page_table, device, ttnn.int32),
        _rm(cur_pos, device, ttnn.int32),
        k,
        valid_length_tensor=_rm(torch.tensor([num_keys], dtype=torch.int32), device, ttnn.uint32),
        output_tensor=out_buf,
        output_row_offset=sliding_window,
    )
    assert out.buffer_address() == out_buf.buffer_address(), "op did not return the provided output_tensor"
    got = ttnn.to_torch(out_buf).to(torch.bfloat16)[0, 0]  # [total_rows, Dh]

    assert torch.equal(got[:sliding_window], prior[0, 0, :sliding_window]), "sliding-window rows were overwritten"
    assert torch.equal(got[sliding_window + k :], prior[0, 0, sliding_window + k :]), "rows past the selection changed"

    # A written row matches if it equals some reference row; each reference row is claimed at most once.
    selected = got[sliding_window : sliding_window + k]
    equal = (selected[:, None, :] == expected[None, :, :]).all(dim=-1)  # [k, k]
    claimed = torch.zeros(k, dtype=torch.bool)
    num_matched = 0
    for i in range(k):
        candidates = (equal[i] & ~claimed).nonzero()
        if candidates.numel() != 0:
            claimed[candidates[0, 0]] = True
            num_matched += 1
    match_ratio = num_matched / k
    logger.info(f"output_tensor rows: {num_matched}/{k} ({match_ratio:.2%}) match the reference top-k")
    assert match_ratio >= 0.95, f"only {num_matched}/{k} ({match_ratio:.2%}) written kv rows match the reference"


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
    gather of the selected rows. At least 90% of the fused op's rows must be
    pipeline rows, in any order.
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
    # The op emits the selected rows in ascending key order.
    pipeline_idx = _as_u32(ttnn.to_torch(row)).reshape(batch, k).sort(dim=-1).values
    expected = _paged_gather(kv_cache, page_table, pipeline_idx)

    # The pipeline needs a tile-padded 32-row block; the fused op takes just the real row.
    real_row = slice(TILE - 1, TILE)
    got = _run_op(
        device, query[:, :, real_row], key_cache, weights[:, :, real_row], kv_cache, page_table, cur_pos, valid, k
    )

    # The fused op scores with LoFi custom_mm and the pipeline with indexer_score_dsa, so a few
    # near-threshold keys differ. Both lists are in key order, so one such difference shifts every
    # later row: compare as sets, each pipeline row claimed at most once.
    got = got.to(torch.bfloat16)[0, 0]  # [k, Dh]
    expected = expected[0, 0]
    equal = (got[:, None, :] == expected[None, :, :]).all(dim=-1)  # [k, k]
    claimed = torch.zeros(k, dtype=torch.bool)
    num_matched = 0
    for i in range(k):
        candidates = (equal[i] & ~claimed).nonzero()
        if candidates.numel() != 0:
            claimed[candidates[0, 0]] = True
            num_matched += 1
    match_ratio = num_matched / k
    logger.info(f"fused_lightning_select_kv: {num_matched}/{k} ({match_ratio:.2%}) rows match the indexer pipeline")
    assert match_ratio >= 0.9, f"only {num_matched}/{k} ({match_ratio:.2%}) rows match the indexer pipeline"


def test_num_keys_tensor_replaces_cur_pos(device) -> None:
    """``num_keys_tensor`` gives the same result as the ``cur_pos`` it stands for, whatever ``cur_pos`` holds."""
    torch.manual_seed(0)
    heads, dim, length, block_size, num_keys = V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, 2048, 64, 1500
    k = V4_INDEX_TOPK
    query = torch.randn(1, heads, 1, dim)
    weights = torch.randn(1, 1, 1, heads) * dim**-0.5
    kv_cache, page_table = _paged_inputs(1, length, block_size, V4_KV_HEADS, V4_HEAD_DIM, seed=3)
    key_cache = torch.randn(page_table.numel(), 1, block_size, dim)

    expected = _run_op(
        device,
        query,
        key_cache,
        weights,
        kv_cache,
        page_table,
        torch.tensor([num_keys * COMPRESS_RATE - 1], dtype=torch.int32),
        num_keys,
        k,
    )
    got, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
        _replicated_rm_hs(query, device),
        _block_sharded_key_cache(key_cache, device),
        _replicated_rm_hs(weights, device),
        _rm(kv_cache, device, ttnn.bfloat16),
        _rm(page_table, device, ttnn.int32),
        _rm(torch.tensor([0], dtype=torch.int32), device, ttnn.int32),
        k,
        valid_length_tensor=_rm(torch.tensor([num_keys], dtype=torch.int32), device, ttnn.uint32),
        num_keys_tensor=_rm(torch.tensor([num_keys], dtype=torch.int32), device, ttnn.int32),
    )
    assert torch.equal(ttnn.to_torch(got), expected), "num_keys_tensor changed the gathered rows"


@pytest.mark.parametrize(
    "length, num_keys, topk_blocks, zero_newest",
    [
        (2048, 1001, 2048, False),
        (2048, 2045, 64, True),
        (128 * 1024, 128 * 1024 - 3, 2048, False),
    ],
    ids=["all_blocks", "pinned_lowest", "v41_128k"],
)
def test_candidate_source_selects_blocks(device, length, num_keys, topk_blocks, zero_newest) -> None:
    """Candidate source mode: the ``topk_blocks`` best 8-key blocks, newest block always kept.

    Mirrors ``select_candidate_blocks`` in the V4.1 reference, applied to the op's own fp32 scores so
    that LoFi scoring differences do not count. Blocks tied at the cutoff may go either way. The
    gathered kv rows must be the same as without source mode. ``zero_newest`` zeroes the newest
    block's keys, so with non-negative head weights it scores lowest and is kept only by the pin.
    """
    torch.manual_seed(0)
    heads, dim, page_block_size, candidate_block_size = V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, 64, 8
    k = V4_INDEX_TOPK
    cur_pos = torch.tensor([num_keys * COMPRESS_RATE - 1], dtype=torch.int32)
    query = torch.randn(1, heads, 1, dim)
    weights = torch.randn(1, 1, 1, heads).abs() * dim**-0.5
    kv_cache, page_table = _paged_inputs(1, length, page_block_size, V4_KV_HEADS, V4_HEAD_DIM, seed=4)
    key_cache = torch.randn(page_table.numel(), 1, page_block_size, dim)
    num_blocks = -(-num_keys // candidate_block_size)
    newest = num_blocks - 1
    if zero_newest:
        keys = _unpage_keys(key_cache, page_table)
        keys[:, :, newest * candidate_block_size : num_keys] = 0
        key_cache = _page_keys(keys, page_table, page_block_size)

    expected_rows = _run_op(device, query, key_cache, weights, kv_cache, page_table, cur_pos, num_keys, k)
    rows, scores, blocks, pool_count = ttnn.experimental.deepseek.fused_lightning_select_kv(
        _replicated_rm_hs(query, device),
        _block_sharded_key_cache(key_cache, device),
        _replicated_rm_hs(weights, device),
        _rm(kv_cache, device, ttnn.bfloat16),
        _rm(page_table, device, ttnn.int32),
        _rm(cur_pos, device, ttnn.int32),
        k,
        valid_length_tensor=_rm(torch.tensor([num_keys], dtype=torch.int32), device, ttnn.uint32),
        candidate_topk_blocks=topk_blocks,
        candidate_block_size=candidate_block_size,
    )
    assert torch.equal(ttnn.to_torch(rows), expected_rows), "source mode changed the gathered kv rows"

    scores = ttnn.to_torch(scores).reshape(-1)[:num_keys].float()
    padded = torch.nn.functional.pad(scores, (0, num_blocks * candidate_block_size - num_keys), value=-torch.inf)
    block_scores = padded.view(num_blocks, candidate_block_size).amax(dim=-1)
    block_scores[newest] = torch.inf
    num_chosen = min(topk_blocks, num_blocks)
    cutoff = block_scores.topk(num_chosen).values[-1]

    blocks = ttnn.to_torch(blocks).reshape(-1).to(torch.int64)
    assert list(blocks.shape) == [topk_blocks]
    chosen = blocks[:num_chosen]
    assert bool((blocks[num_chosen:] == 0).all()), "entries past the chosen blocks must be zero"
    assert bool((chosen[1:] > chosen[:-1]).all()), "candidate blocks must be strictly ascending"
    assert int(chosen[-1]) == newest, "the newest block must be kept, and last"
    assert bool((block_scores[chosen] >= cutoff).all()), "a chosen block scores below the cutoff"
    above = set((block_scores > cutoff).nonzero().reshape(-1).tolist())
    missing = above - set(chosen.tolist())
    assert not missing, f"{len(missing)} blocks above the cutoff were not chosen"

    expected_count = (num_chosen - 1) * candidate_block_size + num_keys - newest * candidate_block_size
    got_count = int(ttnn.to_torch(pool_count).reshape(-1)[0])
    logger.info(f"candidate source: {num_chosen} of {num_blocks} blocks, pool_count {got_count}")
    assert got_count == expected_count, f"pool_count {got_count}, expected {expected_count}"


@pytest.mark.parametrize(
    "length, topk_blocks, num_chosen, pinned_keys",
    [
        (2048, 64, 64, 5),
        (2048, 64, 3, 8),
        (128 * 1024, 2048, 2048, 3),
    ],
    ids=["full", "few_blocks", "v41_128k"],
)
def test_build_candidate_pool_copies_blocks(device, length, topk_blocks, num_chosen, pinned_keys) -> None:
    """``build_candidate_pool`` copies the chosen blocks' keys in order and lists their kv cache rows,
    through a shuffled page table.

    Only the first ``ceil(pool_count / 8)`` blocks are written; the rest of the preallocated pool keeps
    its contents.
    """
    torch.manual_seed(0)
    dim, page_block_size, candidate_block_size = V4_INDEX_HEAD_DIM, 64, 8
    _, page_table = _paged_inputs(1, length, page_block_size, V4_KV_HEADS, 64, seed=5)
    key_cache = torch.randn(page_table.numel(), 1, page_block_size, dim).to(torch.bfloat16)
    num_blocks = length // candidate_block_size
    g = torch.Generator().manual_seed(6)
    older = torch.randperm(num_blocks - 1, generator=g)[: num_chosen - 1].sort().values
    chosen = torch.cat([older, torch.tensor([num_blocks - 1])])
    blocks = torch.zeros(1, topk_blocks, dtype=torch.int32)
    blocks[0, :num_chosen] = chosen.to(torch.int32)
    count = (num_chosen - 1) * candidate_block_size + pinned_keys

    keys = _unpage_keys(key_cache, page_table)[0, 0]  # [length, D]
    written = -(-count // candidate_block_size) * candidate_block_size
    key_idx = (chosen[:, None] * candidate_block_size + torch.arange(candidate_block_size)).reshape(-1)[:written]
    expected_rows = (
        page_table[0].to(torch.int64)[key_idx // page_block_size] * page_block_size + key_idx % page_block_size
    )

    pool_size = topk_blocks * candidate_block_size
    prior_keys = torch.randn(pool_size // page_block_size, 1, page_block_size, dim).to(torch.bfloat16)
    prior_rows = torch.randint(0, 1 << 20, (1, pool_size), dtype=torch.int32)
    pool_keys, pool_rows = ttnn.experimental.deepseek.build_candidate_pool(
        _rm(blocks, device, ttnn.int32),
        _rm(torch.tensor([count], dtype=torch.int32), device, ttnn.int32),
        _tile(key_cache, device),
        _rm(page_table, device, ttnn.int32),
        candidate_block_size,
        pool_keys_output=_tile(prior_keys, device),
        pool_rows_output=_rm(prior_rows, device, ttnn.int32),
    )
    got_keys = ttnn.to_torch(pool_keys).to(torch.bfloat16).reshape(-1, dim)
    got_rows = ttnn.to_torch(pool_rows).reshape(-1).to(torch.int64)
    assert torch.equal(got_keys[:count], keys[key_idx[:count]]), "pool keys differ from the chosen blocks' keys"
    assert torch.equal(got_rows[:written], expected_rows), "pool rows differ from the chosen blocks' kv cache rows"

    assert torch.equal(got_keys[written:], prior_keys.reshape(-1, dim)[written:]), "pool keys past the count changed"
    assert torch.equal(
        got_rows[written:], prior_rows.reshape(-1).to(torch.int64)[written:]
    ), "pool rows past the count changed"


def test_candidate_pool_chain_matches_reference(device) -> None:
    """Source select -> ``build_candidate_pool`` -> consumer select on the pool, against a torch reference.

    The consumer has its own query and head weights and must pick its top-k among the source's
    candidate keys only, as the V4.1 reference's level two does. As in the other gather tests, at
    least 95% of the gathered rows must be reference rows, in any order.
    """
    torch.manual_seed(0)
    heads, dim, page_block_size, candidate_block_size = V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, 64, 8
    length, num_keys, topk_blocks, k = 8192, 8189, 256, V4_INDEX_TOPK
    cur_pos = torch.tensor([num_keys * COMPRESS_RATE - 1], dtype=torch.int32)
    src_query = torch.randn(1, heads, 1, dim).to(torch.bfloat16).float()
    src_weights = (torch.randn(1, 1, 1, heads) * dim**-0.5).to(torch.bfloat16).float()
    dst_query = torch.randn(1, heads, 1, dim).to(torch.bfloat16).float()
    dst_weights = (torch.randn(1, 1, 1, heads) * dim**-0.5).to(torch.bfloat16).float()
    kv_cache, page_table = _paged_inputs(1, length, page_block_size, V4_KV_HEADS, V4_HEAD_DIM, seed=7)
    key_cache = torch.randn(page_table.numel(), 1, page_block_size, dim).to(torch.bfloat16)
    key_tt = _tile(key_cache, device)
    kv_tt = _rm(kv_cache, device, ttnn.bfloat16)
    page_table_tt = _rm(page_table, device, ttnn.int32)

    _, _, blocks, pool_count = ttnn.experimental.deepseek.fused_lightning_select_kv(
        _replicated_rm_hs(src_query, device),
        key_tt,
        _replicated_rm_hs(src_weights, device),
        kv_tt,
        page_table_tt,
        _rm(cur_pos, device, ttnn.int32),
        k,
        candidate_topk_blocks=topk_blocks,
        candidate_block_size=candidate_block_size,
    )
    pool_keys, pool_rows = ttnn.experimental.deepseek.build_candidate_pool(
        blocks, pool_count, key_tt, page_table_tt, candidate_block_size
    )
    pool_blocks = topk_blocks * candidate_block_size // page_block_size
    identity = torch.arange(pool_blocks, dtype=torch.int32).reshape(1, pool_blocks)
    got, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
        _replicated_rm_hs(dst_query, device),
        pool_keys,
        _replicated_rm_hs(dst_weights, device),
        kv_tt,
        _rm(identity, device, ttnn.int32),
        _rm(torch.tensor([0], dtype=torch.int32), device, ttnn.int32),
        k,
        num_keys_tensor=pool_count,
        row_table=pool_rows,
    )
    got = ttnn.to_torch(got).to(torch.bfloat16)[0, 0]  # [k, Dh]

    count = int(ttnn.to_torch(pool_count).reshape(-1)[0])
    chosen = ttnn.to_torch(blocks).reshape(-1).to(torch.int64)[: -(-count // candidate_block_size)]
    cand_idx = (chosen[:, None] * candidate_block_size + torch.arange(candidate_block_size)).reshape(-1)[:count]
    keys = _unpage_keys(key_cache, page_table)[0, 0]
    scores = _reference_scores(dst_query[0, :, 0], keys[cand_idx], dst_weights[0, 0, 0])
    expected_idx = cand_idx[torch.topk(scores, k).indices]
    expected = _paged_gather(kv_cache, page_table, expected_idx.reshape(1, k))[0, 0]

    equal = (got[:, None, :] == expected[None, :, :]).all(dim=-1)  # [k, k]
    claimed = torch.zeros(k, dtype=torch.bool)
    num_matched = 0
    for i in range(k):
        candidates = (equal[i] & ~claimed).nonzero()
        if candidates.numel() != 0:
            claimed[candidates[0, 0]] = True
            num_matched += 1
    match_ratio = num_matched / k
    logger.info(f"candidate chain: pool of {count} keys, {num_matched}/{k} ({match_ratio:.2%}) rows match")
    assert match_ratio >= 0.95, f"only {num_matched}/{k} ({match_ratio:.2%}) chained kv rows match the reference"


def test_pool_consumer_writes_new_kv_row(device) -> None:
    """A pool consumer given ``new_kv_row`` writes it at kv_cache row ``new_kv_row_index`` (the page table
    maps the pool, so the index is not paged) and leaves every other row alone, as a V4.1 consumer stage
    does with the compressed-KV row its source sent."""
    torch.manual_seed(0)
    heads, dim, page_block_size, candidate_block_size = V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, 64, 8
    num_keys, topk_blocks, k = 4096 - 3, 256, V4_INDEX_TOPK
    num_blocks = 4096 // page_block_size
    cur_pos = _rm(torch.tensor([num_keys * COMPRESS_RATE - 1], dtype=torch.int32), device, ttnn.int32)
    kv_cache = torch.randn(num_blocks, 1, page_block_size, V4_HEAD_DIM).to(torch.bfloat16)
    key_tt = _tile(torch.randn(num_blocks, 1, page_block_size, dim), device)
    kv_tt = _rm(kv_cache, device, ttnn.bfloat16)
    page_table_tt = _rm(torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks), device, ttnn.int32)
    query = _replicated_rm_hs(torch.randn(1, heads, 1, dim), device)
    weights = _replicated_rm_hs(torch.randn(1, 1, 1, heads) * dim**-0.5, device)

    _, _, blocks, pool_count = ttnn.experimental.deepseek.fused_lightning_select_kv(
        query,
        key_tt,
        weights,
        kv_tt,
        page_table_tt,
        cur_pos,
        k,
        candidate_topk_blocks=topk_blocks,
        candidate_block_size=candidate_block_size,
    )
    pool_keys, pool_rows = ttnn.experimental.deepseek.build_candidate_pool(
        blocks, pool_count, key_tt, page_table_tt, candidate_block_size
    )
    pool_blocks = topk_blocks * candidate_block_size // page_block_size
    new_row = torch.randn(1, 1, 1, V4_HEAD_DIM).to(torch.bfloat16)
    new_index = num_keys - 1
    ttnn.experimental.deepseek.fused_lightning_select_kv(
        query,
        pool_keys,
        weights,
        kv_tt,
        _rm(torch.arange(pool_blocks, dtype=torch.int32).reshape(1, pool_blocks), device, ttnn.int32),
        cur_pos,
        k,
        new_kv_row=_rm(new_row, device, ttnn.bfloat16),
        new_kv_row_index=_rm(torch.tensor([new_index], dtype=torch.int32), device, ttnn.int32),
        num_keys_tensor=pool_count,
        row_table=pool_rows,
    )
    expected = kv_cache.reshape(-1, V4_HEAD_DIM).clone()
    expected[new_index] = new_row.reshape(-1)
    got = ttnn.to_torch(kv_tt).to(torch.bfloat16).reshape(-1, V4_HEAD_DIM)
    assert torch.equal(got, expected), "the consumer did not write exactly the new row into kv_cache"


# The V4.1 model's mesh: 8 pipeline stages of 1x4 TP submeshes.
PARENT_MESH = (8, 4)
TP_SIZE = 4


def _tp_submesh(mesh_device):
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"needs an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    return mesh_device.create_submesh(ttnn.MeshShape(1, TP_SIZE), ttnn.MeshCoordinate(0, 0))


def _split_select(submesh, query, key_tt, weights, kv_tt, page_table_tt, cur_pos_tt, k, **kwargs) -> ttnn.Tensor:
    """Key-split select over the submesh's devices: each scores its part, then the candidates are
    all-gathered and merged on every device."""
    part_index = _rm(
        torch.arange(TP_SIZE, dtype=torch.int32),
        submesh,
        ttnn.int32,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )
    candidates, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
        query,
        key_tt,
        weights,
        kv_tt,
        page_table_tt,
        cur_pos_tt,
        k,
        num_parts=TP_SIZE,
        part_index_tensor=part_index,
        **kwargs,
    )
    gathered = ttnn.all_gather(candidates, dim=2, cluster_axis=1)
    return ttnn.experimental.deepseek.merge_select_kv(gathered, kv_tt, k)


def _per_device(t: ttnn.Tensor, submesh) -> list[torch.Tensor]:
    """The submesh devices' copies of ``t``, in device order."""
    return list(ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)).chunk(TP_SIZE, dim=0))


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize(
    "length, num_keys",
    [(2048, 100), (2048, 1000), (128 * 1024, 128 * 1024 - 3)],
    ids=["fewer_than_k", "small", "v41_128k"],
)
def test_key_split_matches_single_device(mesh_device, length, num_keys) -> None:
    """Scoring a quarter of the blocks on each of 4 devices, then all_gather + ``merge_select_kv``,
    gathers exactly the rows of the unsplit op, in the same order, on every device.

    ``fewer_than_k`` leaves two devices without keys and selects every key; ``small`` gives the last
    device a short part.
    """
    submesh = _tp_submesh(mesh_device)
    replicate = ttnn.ReplicateTensorToMesh(submesh)
    torch.manual_seed(0)
    heads, dim, page_block_size, k = V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, 64, V4_INDEX_TOPK
    query = _replicated_rm_hs(torch.randn(1, heads, 1, dim), submesh, mesh_mapper=replicate)
    weights = _replicated_rm_hs(torch.randn(1, 1, 1, heads) * dim**-0.5, submesh, mesh_mapper=replicate)
    kv_cache, page_table = _paged_inputs(1, length, page_block_size, V4_KV_HEADS, V4_HEAD_DIM, seed=8)
    key_tt = _tile(torch.randn(page_table.numel(), 1, page_block_size, dim), submesh, mesh_mapper=replicate)
    kv_tt = _rm(kv_cache, submesh, ttnn.bfloat16, mesh_mapper=replicate)
    page_table_tt = _rm(page_table, submesh, ttnn.int32, mesh_mapper=replicate)
    cur_pos_tt = _rm(
        torch.tensor([num_keys * COMPRESS_RATE - 1], dtype=torch.int32), submesh, ttnn.int32, mesh_mapper=replicate
    )

    expected, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
        query, key_tt, weights, kv_tt, page_table_tt, cur_pos_tt, k
    )
    got = _split_select(submesh, query, key_tt, weights, kv_tt, page_table_tt, cur_pos_tt, k)

    num_selected = min(k, num_keys)
    expected = _per_device(expected, submesh)
    for device_id, rows in enumerate(_per_device(got, submesh)):
        assert torch.equal(
            rows[..., :num_selected, :], expected[device_id][..., :num_selected, :]
        ), f"device {device_id}: the key split gathered different rows than the unsplit op"


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
def test_key_split_pool_consumer_matches_single_device(mesh_device) -> None:
    """A candidate-pool consumer split over 4 devices (pool keys by part, rows through the row table)
    gathers exactly the rows of the unsplit consumer."""
    submesh = _tp_submesh(mesh_device)
    replicate = ttnn.ReplicateTensorToMesh(submesh)
    torch.manual_seed(0)
    heads, dim, page_block_size, candidate_block_size = V4_INDEX_HEADS, V4_INDEX_HEAD_DIM, 64, 8
    length, num_keys, topk_blocks, k = 32 * 1024, 32 * 1024 - 5, 2048, V4_INDEX_TOPK
    src_query = _replicated_rm_hs(torch.randn(1, heads, 1, dim), submesh, mesh_mapper=replicate)
    src_weights = _replicated_rm_hs(torch.randn(1, 1, 1, heads) * dim**-0.5, submesh, mesh_mapper=replicate)
    dst_query = _replicated_rm_hs(torch.randn(1, heads, 1, dim), submesh, mesh_mapper=replicate)
    dst_weights = _replicated_rm_hs(torch.randn(1, 1, 1, heads) * dim**-0.5, submesh, mesh_mapper=replicate)
    kv_cache, page_table = _paged_inputs(1, length, page_block_size, V4_KV_HEADS, V4_HEAD_DIM, seed=9)
    key_tt = _tile(torch.randn(page_table.numel(), 1, page_block_size, dim), submesh, mesh_mapper=replicate)
    kv_tt = _rm(kv_cache, submesh, ttnn.bfloat16, mesh_mapper=replicate)
    page_table_tt = _rm(page_table, submesh, ttnn.int32, mesh_mapper=replicate)
    cur_pos_tt = _rm(
        torch.tensor([num_keys * COMPRESS_RATE - 1], dtype=torch.int32), submesh, ttnn.int32, mesh_mapper=replicate
    )

    _, _, blocks, pool_count = ttnn.experimental.deepseek.fused_lightning_select_kv(
        src_query,
        key_tt,
        src_weights,
        kv_tt,
        page_table_tt,
        cur_pos_tt,
        k,
        candidate_topk_blocks=topk_blocks,
        candidate_block_size=candidate_block_size,
    )
    pool_keys, pool_rows = ttnn.experimental.deepseek.build_candidate_pool(
        blocks, pool_count, key_tt, page_table_tt, candidate_block_size
    )
    pool_blocks = topk_blocks * candidate_block_size // page_block_size
    identity = _rm(
        torch.arange(pool_blocks, dtype=torch.int32).reshape(1, pool_blocks), submesh, ttnn.int32, mesh_mapper=replicate
    )
    expected, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
        dst_query,
        pool_keys,
        dst_weights,
        kv_tt,
        identity,
        cur_pos_tt,
        k,
        num_keys_tensor=pool_count,
        row_table=pool_rows,
    )
    got = _split_select(
        submesh,
        dst_query,
        pool_keys,
        dst_weights,
        kv_tt,
        identity,
        cur_pos_tt,
        k,
        num_keys_tensor=pool_count,
        row_table=pool_rows,
    )

    expected = _per_device(expected, submesh)
    for device_id, rows in enumerate(_per_device(got, submesh)):
        assert torch.equal(rows, expected[device_id]), f"device {device_id}: the split consumer gathered different rows"
