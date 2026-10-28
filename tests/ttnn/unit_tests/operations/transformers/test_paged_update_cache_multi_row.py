# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-row ``paged_update_cache``: one call writes ``S`` ROW_MAJOR rows per user.

The input is ``[1, B, S, D]`` height-sharded one user per core, and ``update_idxs`` holds one
position per row, user-major (entry ``b * S + s`` is row ``s`` of user ``b``; ``-1`` skips the row).
Rows are placed independently, so they may share a tile-row, straddle tiles, wrap a ring, or cross
page-table blocks.

Run::

    pytest tests/ttnn/unit_tests/operations/transformers/test_paged_update_cache_multi_row.py
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

HEAD_DIM = 128


def _rows_per_user(device, x: torch.Tensor) -> ttnn.Tensor:
    """``[1, B, S, D]`` -> ROW_MAJOR, height-sharded with shard ``[S, D]`` on ``B`` cores."""
    _, num_users, rows, dim = x.shape
    grid = ttnn.num_cores_to_corerangeset(num_users, device.compute_with_storage_grid_size(), True)
    shard_spec = ttnn.ShardSpec(grid, [rows, dim], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    return ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )


def _int32(values, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.as_tensor(values, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _tile_cache(t: torch.Tensor, dtype, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _read(cache: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(cache).float()


def _check(got: torch.Tensor, prior: torch.Tensor, x: torch.Tensor, targets, cache_dtype) -> None:
    """``targets[(b, s)] = (i0, i2)``: row ``s`` of user ``b`` must land at ``cache[i0, 0, i2]``; every
    other cache row must equal ``prior`` bit for bit."""
    written = torch.zeros(prior.shape[0], prior.shape[2], dtype=torch.bool)
    expected = prior.clone()
    for (b, s), (i0, i2) in targets.items():
        expected[i0, 0, i2] = x[0, b, s].float()
        written[i0, i2] = True
    untouched = ~written
    assert torch.equal(got[:, 0][untouched], prior[:, 0][untouched]), "untouched cache rows changed"
    if cache_dtype == ttnn.bfloat16:
        assert torch.equal(got[:, 0][written], expected[:, 0][written]), "written rows mismatch"
    else:
        passed, msg = comp_pcc(expected[:, 0][written], got[:, 0][written], pcc=0.99)
        assert passed, msg


def _non_paged_targets(positions_per_user):
    return {(b, s): (b, p) for b, positions in enumerate(positions_per_user) for s, p in enumerate(positions) if p >= 0}


def _run_non_paged(device, positions_per_user, cache_len, cache_dtype, use_index_tensor):
    num_users, rows = len(positions_per_user), len(positions_per_user[0])
    prior_src = torch.randn(num_users, 1, cache_len, HEAD_DIM).to(torch.bfloat16)
    cache = _tile_cache(prior_src, cache_dtype, device)
    prior = _read(cache)
    x = torch.randn(1, num_users, rows, HEAD_DIM).to(torch.bfloat16)
    flat = [p for positions in positions_per_user for p in positions]
    if use_index_tensor:
        ttnn.experimental.paged_update_cache(cache, _rows_per_user(device, x), update_idxs_tensor=_int32(flat, device))
    else:
        ttnn.experimental.paged_update_cache(cache, _rows_per_user(device, x), update_idxs=flat)
    _check(_read(cache), prior, x, _non_paged_targets(positions_per_user), cache_dtype)


@pytest.mark.parametrize("use_index_tensor", [True, False], ids=["index_tensor", "host_vector"])
@pytest.mark.parametrize(
    "positions, cache_len",
    [
        ([3, 4, 5, 6, 7], 128),
        ([30, 31, 32, 33, 34], 128),
        ([126, 127, 0, 1, 2], 128),
        ([77], 128),
        ([60, 61, 62, 63, 64, 65, 66], 128),
        (list(range(5, 37)), 128),
        (list(range(90, 123)), 128),
        ([200, 7, 64, 9, 31], 256),
    ],
    ids=["in_tile", "straddle", "ring_wrap", "s1", "s7", "s32", "s33", "scattered"],
)
def test_non_paged_batch1(device, positions, cache_len, use_index_tensor):
    torch.manual_seed(0)
    _run_non_paged(device, [positions], cache_len, ttnn.bfloat16, use_index_tensor)


def test_non_paged_skip_rows(device):
    """``-1`` skips just that row; the others of the same user still land."""
    torch.manual_seed(0)
    _run_non_paged(device, [[10, -1, 12, -1, 40]], 128, ttnn.bfloat16, use_index_tensor=True)


def test_non_paged_all_rows_skipped(device):
    torch.manual_seed(0)
    _run_non_paged(device, [[-1, -1, -1]], 64, ttnn.bfloat16, use_index_tensor=True)


@pytest.mark.parametrize("use_index_tensor", [True, False], ids=["index_tensor", "host_vector"])
def test_non_paged_bfloat8_b(device, use_index_tensor):
    torch.manual_seed(0)
    _run_non_paged(device, [[30, 31, 32, 33, 34]], 128, ttnn.bfloat8_b, use_index_tensor)


@pytest.mark.parametrize("use_index_tensor", [True, False], ids=["index_tensor", "host_vector"])
def test_non_paged_multi_user(device, use_index_tensor):
    """``B = 4``, ``S = 3``: each user writes its own cache batch."""
    torch.manual_seed(0)
    positions = [[5, 6, 7], [31, 32, 100], [127, 0, 1], [64, 96, 65]]
    _run_non_paged(device, positions, 128, ttnn.bfloat16, use_index_tensor)


def _paged_cache(device, num_blocks, block_size, cache_dtype):
    prior_src = torch.randn(num_blocks, 1, block_size, HEAD_DIM).to(torch.bfloat16)
    cache = _tile_cache(prior_src, cache_dtype, device)
    return cache, _read(cache)


@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_paged_batched_page_table(device, cache_dtype):
    """``B = 2``, ``S = 5`` through a shuffled ``[B, max_blocks]`` page table, crossing block boundaries."""
    torch.manual_seed(0)
    num_users, blocks_per_seq, block_size = 2, 4, 64
    num_blocks = num_users * blocks_per_seq
    page_table = torch.randperm(num_blocks).to(torch.int32).reshape(num_users, blocks_per_seq)
    cache, prior = _paged_cache(device, num_blocks, block_size, cache_dtype)
    positions = [[62, 63, 64, 65, 66], [125, 126, 127, 128, 129]]
    x = torch.randn(1, num_users, 5, HEAD_DIM).to(torch.bfloat16)
    ttnn.experimental.paged_update_cache(
        cache,
        _rows_per_user(device, x),
        update_idxs_tensor=_int32([p for row in positions for p in row], device),
        page_table=_int32(page_table, device),
    )
    targets = {
        (b, s): (int(page_table[b, p // block_size]), p % block_size)
        for b, row in enumerate(positions)
        for s, p in enumerate(row)
    }
    _check(_read(cache), prior, x, targets, cache_dtype)


@pytest.mark.parametrize("session", [0, 2])
def test_paged_page_table_row_tensor(device, session):
    """One session writes ``T = 5`` consecutive logical entries through its page-table row."""
    torch.manual_seed(0)
    num_sessions, blocks_per_session, block_size = 3, 8, 64
    num_blocks = num_sessions * blocks_per_session
    table = torch.randperm(num_blocks).to(torch.int32).reshape(num_sessions, blocks_per_session)
    cache, prior = _paged_cache(device, num_blocks, block_size, ttnn.bfloat16)
    positions = [126, 127, 128, 129, 130]
    x = torch.randn(1, 1, len(positions), HEAD_DIM).to(torch.bfloat16)
    ttnn.experimental.paged_update_cache(
        cache,
        _rows_per_user(device, x),
        update_idxs_tensor=_int32(positions, device),
        page_table=_int32(table, device),
        page_table_row_tensor=_int32([session], device),
    )
    targets = {(0, s): (int(table[session, p // block_size]), p % block_size) for s, p in enumerate(positions)}
    _check(_read(cache), prior, x, targets, ttnn.bfloat16)


def test_paged_cache_position_modulo(device):
    """Absolute positions past the bounded ring wrap per row before the page-table lookup."""
    torch.manual_seed(0)
    block_size, modulo, max_model_len = 32, 128, 256
    ring_blocks = modulo // block_size
    page_table = torch.zeros(1, max_model_len // block_size, dtype=torch.int32)
    page_table[0, :ring_blocks] = torch.randperm(ring_blocks).to(torch.int32)
    cache, prior = _paged_cache(device, ring_blocks, block_size, ttnn.bfloat16)
    positions = [254, 255, 256, 257, 258]
    x = torch.randn(1, 1, len(positions), HEAD_DIM).to(torch.bfloat16)
    ttnn.experimental.paged_update_cache(
        cache,
        _rows_per_user(device, x),
        update_idxs_tensor=_int32(positions, device),
        page_table=_int32(page_table, device),
        cache_position_modulo=modulo,
    )
    targets = {
        (0, s): (int(page_table[0, (p % modulo) // block_size]), (p % modulo) % block_size)
        for s, p in enumerate(positions)
    }
    _check(_read(cache), prior, x, targets, ttnn.bfloat16)


@pytest.mark.parametrize("use_index_tensor", [True, False], ids=["index_tensor", "host_vector"])
def test_program_cache_reuse(device, use_index_tensor):
    """Same shapes, new positions: the cache-hit path must re-patch every row's position."""
    torch.manual_seed(0)
    cache_len = 128
    prior_src = torch.randn(1, 1, cache_len, HEAD_DIM).to(torch.bfloat16)
    cache = _tile_cache(prior_src, ttnn.bfloat16, device)
    num_entries = None
    for positions in ([30, 31, 32, 33, 34], [126, 127, 0, 1, 2], [70, 3, 71, 99, 4]):
        prior = _read(cache)
        x = torch.randn(1, 1, len(positions), HEAD_DIM).to(torch.bfloat16)
        if use_index_tensor:
            ttnn.experimental.paged_update_cache(
                cache, _rows_per_user(device, x), update_idxs_tensor=_int32(positions, device)
            )
        else:
            ttnn.experimental.paged_update_cache(cache, _rows_per_user(device, x), update_idxs=positions)
        _check(_read(cache), prior, x, _non_paged_targets([positions]), ttnn.bfloat16)
        if num_entries is None:
            num_entries = device.num_program_cache_entries()
        assert device.num_program_cache_entries() == num_entries, "multi-row update recompiled on a cache hit"


def test_row_count_is_part_of_program_hash(device):
    """``[1, 1, 5, D]`` input with one index writes only row 0; with five indices, all five. Same tensor
    specs, different programs."""
    torch.manual_seed(0)
    prior_src = torch.randn(1, 1, 128, HEAD_DIM).to(torch.bfloat16)
    cache = _tile_cache(prior_src, ttnn.bfloat16, device)
    x = torch.randn(1, 1, 5, HEAD_DIM).to(torch.bfloat16)
    xt = _rows_per_user(device, x)

    prior = _read(cache)
    ttnn.experimental.paged_update_cache(cache, xt, update_idxs=[40])
    _check(_read(cache), prior, x, {(0, 0): (0, 40)}, ttnn.bfloat16)

    prior = _read(cache)
    positions = [50, 51, 52, 53, 54]
    ttnn.experimental.paged_update_cache(cache, xt, update_idxs=positions)
    _check(_read(cache), prior, x, _non_paged_targets([positions]), ttnn.bfloat16)


def _width_sharded_rm_cache(device, cache: torch.Tensor) -> ttnn.Tensor:
    height, width = cache.shape[0], cache.shape[-1]
    num_cores = width // 32
    grid = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), True)
    shard_spec = ttnn.ShardSpec(grid, [height, 32], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    return ttnn.from_torch(
        cache, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )


@pytest.mark.parametrize("use_index_tensor", [True, False], ids=["index_tensor", "host_vector"])
def test_rm_width_sharded_cache(device, use_index_tensor):
    """Writer-only L1 cache path: ``B = 2``, ``S = 3``; positions are linear rows of the packed cache."""
    torch.manual_seed(0)
    num_users, rows, height = 2, 3, 16
    prior = torch.randn(height, 1, 1, HEAD_DIM).to(torch.bfloat16)
    cache = _width_sharded_rm_cache(device, prior)
    x = torch.randn(1, num_users, rows, HEAD_DIM).to(torch.bfloat16)
    positions = [[0, 1, 2], [9, 15, 4]]
    flat = [p for row in positions for p in row]
    if use_index_tensor:
        ttnn.experimental.paged_update_cache(cache, _rows_per_user(device, x), update_idxs_tensor=_int32(flat, device))
    else:
        ttnn.experimental.paged_update_cache(cache, _rows_per_user(device, x), update_idxs=flat)
    expected = prior.clone()
    for b, row in enumerate(positions):
        for s, p in enumerate(row):
            expected[p, 0, 0] = x[0, b, s]
    assert torch.equal(ttnn.to_torch(cache).to(torch.bfloat16), expected)


def test_rm_width_sharded_cache_skip_row(device):
    torch.manual_seed(0)
    height = 8
    prior = torch.randn(height, 1, 1, HEAD_DIM).to(torch.bfloat16)
    cache = _width_sharded_rm_cache(device, prior)
    x = torch.randn(1, 1, 3, HEAD_DIM).to(torch.bfloat16)
    ttnn.experimental.paged_update_cache(
        cache, _rows_per_user(device, x), update_idxs_tensor=_int32([6, -1, 2], device)
    )
    expected = prior.clone()
    expected[6, 0, 0] = x[0, 0, 0]
    expected[2, 0, 0] = x[0, 0, 2]
    assert torch.equal(ttnn.to_torch(cache).to(torch.bfloat16), expected)


def test_tile_input_rejects_multiple_rows(device, expect_error):
    torch.manual_seed(0)
    cache = _tile_cache(torch.randn(1, 1, 128, HEAD_DIM).to(torch.bfloat16), ttnn.bfloat16, device)
    x = torch.randn(1, 1, 32, HEAD_DIM).to(torch.bfloat16)
    grid = ttnn.num_cores_to_corerangeset(1, device.compute_with_storage_grid_size(), True)
    shard_spec = ttnn.ShardSpec(grid, [32, HEAD_DIM], ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config)
    with expect_error(RuntimeError, "ROW_MAJOR input"):
        ttnn.experimental.paged_update_cache(cache, xt, update_idxs=[0, 1, 2, 3, 4])
