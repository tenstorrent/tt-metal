# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""paged_fused_update_cache(rows_per_user=T): one call writes each user's T rows and must leave the caches EXACTLY as T
sequential single-row calls do (the shipped spec-verify KV write), including 32-row tile straddles, page-block
straddles, -1 rows, a duplicated position (later row wins) and bf8 caches. Also checks a bf16 cache against torch and
that the multi-row program is cached by shape (positions are runtime). Case "past_table": rows whose position is past
the user's page table (virtual block >= its width, e.g. INT32_MAX) are skipped like -1, so the cache is the T
single-row calls' result with those rows dropped (the single-row kernels have no such check, so the reference gets -1
there); nothing outside those rows' tiles may change."""

import pytest
import torch
import ttnn

TILE = 32
NB = 4  # page-table blocks per user


def _row_cores(first, n, grid_x):
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(i % grid_x, i // grid_x), ttnn.CoreCoord(i % grid_x, i // grid_x))
            for i in range(first, first + n)
        }
    )


def _sharded(x_pad, first_core, n, rows, device):
    """[1, n, rows, HD] bf16 TILE, height sharded one user per core on cores [first_core, first_core + n)."""
    grid = device.compute_with_storage_grid_size()
    cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_row_cores(first_core, n, grid.x), [rows, x_pad.shape[-1]], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(x_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)


def _tiles(x, n, T, num_heads, head_dim):
    """x [n, T, num_heads, HD] -> [1, n, T*32, HD]: tile j of user u holds row j, head h on tile row h."""
    out = torch.zeros(1, n, T * TILE, head_dim)
    for j in range(T):
        out[0, :, j * TILE : j * TILE + num_heads] = x[:, j]
    return out


def _positions(n, T, block_size, case):
    """Per-row positions [n*T], user-major. Consecutive candidates p..p+T-1 per user (the verify layout)."""
    starts = [TILE - 2, block_size - 1, 5, 2 * block_size + 3, TILE - 1, block_size - 3, 0, 7][:n]
    idx = []
    for u, p in enumerate(starts):
        idx += [p + j for j in range(T)]
    if case == "skips" and n > 1:
        idx[0:T] = [-1] * T  # inactive user
        idx[T + 1] = -1  # one row of a live user
    if case == "dup" and T > 2:
        idx[2] = idx[1]  # duplicate position inside user 0: the later row must win
    if case == "past_table":
        nb_rows = NB * block_size  # first position past a user's table
        u = 1 if n > 1 else 0
        idx[u * T + T - 1] = nb_rows  # one block past the table
        idx[u * T + T - 2] = nb_rows + block_size * 1000 + 5  # far past it
        idx[(n - 1) * T] = 2**31 - 1  # INT32_MAX, e.g. a saturated uint32 -1
    return idx


def _ref_positions(idx, block_size):
    """What the T single-row calls get: rows past the table become -1 (the multi-row kernels skip them)."""
    return [p if p < NB * block_size else -1 for p in idx]


def _run(device, n, T, num_heads, head_dim, block_size, cache_dtype, case):
    torch.manual_seed(0)
    nb = NB  # blocks per user
    num_blocks = n * nb + 1  # + a never-written spare
    perm = torch.randperm(num_blocks - 1) + 1
    page_table = perm[: n * nb].reshape(n, nb).to(torch.int32)
    cache_shape = [num_blocks, num_heads, block_size, head_dim]
    k0 = torch.randn(cache_shape).bfloat16().float()
    v0 = torch.randn(cache_shape).bfloat16().float()
    xk = torch.randn(n, T, num_heads, head_dim).bfloat16().float()
    xv = torch.randn(n, T, num_heads, head_dim).bfloat16().float()
    idx = _positions(n, T, block_size, case)
    idx_ref = _ref_positions(idx, block_size)

    def fresh():
        return (
            ttnn.from_torch(k0, dtype=cache_dtype, layout=ttnn.TILE_LAYOUT, device=device),
            ttnn.from_torch(v0, dtype=cache_dtype, layout=ttnn.TILE_LAYOUT, device=device),
        )

    pt_tt = ttnn.from_torch(page_table, dtype=ttnn.int32, device=device)

    # Reference: T sequential single-row fused calls (what _write_kv_fused_groups issues today).
    k_ref, v_ref = fresh()
    for j in range(T):
        pos_j = ttnn.from_torch(torch.tensor(idx_ref[j::T], dtype=torch.int32), dtype=ttnn.int32, device=device)
        ks = _sharded(_tiles(xk[:, j : j + 1], n, 1, num_heads, head_dim), n, n, TILE, device)
        vs = _sharded(_tiles(xv[:, j : j + 1], n, 1, num_heads, head_dim), 0, n, TILE, device)
        ttnn.experimental.paged_fused_update_cache(k_ref, ks, v_ref, vs, update_idxs_tensor=pos_j, page_table=pt_tt)

    # Multi-row: one call.
    k_new, v_new = fresh()
    pos_all = ttnn.from_torch(torch.tensor(idx, dtype=torch.int32), dtype=ttnn.int32, device=device)
    ks = _sharded(_tiles(xk, n, T, num_heads, head_dim), n, n, T * TILE, device)
    vs = _sharded(_tiles(xv, n, T, num_heads, head_dim), 0, n, T * TILE, device)
    ttnn.experimental.paged_fused_update_cache(
        k_new, ks, v_new, vs, update_idxs_tensor=pos_all, page_table=pt_tt, rows_per_user=T
    )

    for name, ref, new, c0, x in (("K", k_ref, k_new, k0, xk), ("V", v_ref, v_new, v0, xv)):
        r, g = ttnn.to_torch(ref), ttnn.to_torch(new)
        assert torch.equal(r, g), f"{name}: multi-row cache differs from {T} sequential calls ({(r != g).sum()} values)"
        if cache_dtype == ttnn.bfloat16:
            exp = c0.clone()
            for u in range(n):
                for j in range(T):
                    p = idx_ref[u * T + j]
                    if p < 0:
                        continue
                    blk = int(page_table[u, p // block_size])
                    exp[blk, :, p % block_size] = x[u, j]
            assert torch.equal(exp, g), f"{name}: multi-row bf16 cache differs from torch"


@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bf8", "bf16"])
@pytest.mark.parametrize("block_size", [32, 64], ids=["block32", "block64"])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("n,T", [(1, 4), (8, 4), (4, 8), (8, 2)])
@pytest.mark.parametrize("case", ["straddle", "skips", "dup", "past_table"])
def test_multi_row_matches_sequential(device, n, T, head_dim, block_size, cache_dtype, case):
    _run(device, n, T, 1, head_dim, block_size, cache_dtype, case)


@pytest.mark.parametrize("num_heads", [2, 8])
def test_multi_row_multi_head(device, num_heads):
    _run(device, 4, 4, num_heads, 128, 64, ttnn.bfloat8_b, "straddle")


def test_multi_row_program_cache(device):
    _run(device, 8, 4, 1, 256, 64, ttnn.bfloat8_b, "straddle")
    entries = device.num_program_cache_entries()
    _run(device, 8, 4, 1, 256, 64, ttnn.bfloat8_b, "skips")  # same shapes, other positions: cache hit
    assert device.num_program_cache_entries() == entries
