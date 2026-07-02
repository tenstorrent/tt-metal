# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debug tests for all_gather. DO NOT DELETE.

Documents the page-grid model that the strided concat addressing (Refinement 2,
gather_dim != 0) depends on, and verifies the host out_page() remap formula
against a torch reference for every gather_dim.

The op is pure byte movement: it copies whole physical pages. For that to
reconstruct a concat-along-gather_dim, each shard's local page p must map to a
predictable output page. This test pins:

  * the page grid per layout: TILE = [B, C, Ht, Wt], RM = [B, C, H] (W is
    INSIDE the row-major page, not a page-grid axis);
  * the remap out_page(c, p) = high*(N*dim_j*inner) + (c*dim_j+mid)*inner + low.
"""

from math import prod

import pytest
import torch
import ttnn


def _page_grid(shape4, tile):
    B, C, H, W = shape4
    if tile:
        return [B, C, (H + 31) // 32, (W + 31) // 32]
    return [B, C, H]  # RM page == one W-row


def _dim_inner(shape4, tile, gd_neg):
    """(dim_j, inner_stride) in page units, or None if the gathered axis is
    intra-page (RM + gather_dim=-1 => sub-page, unsupported by pure page copy)."""
    grid = _page_grid(shape4, tile)
    logical_axis = 4 + gd_neg  # 0=B 1=C 2=H 3=W
    if not tile and logical_axis == 3:
        return None
    page_axis = logical_axis
    dim_j = grid[page_axis]
    inner = 1
    for a in range(page_axis + 1, len(grid)):
        inner *= grid[a]
    return dim_j, inner


def _out_page(c, p, dim_j, inner, N):
    block = dim_j * inner
    high = p // block
    rem = p % block
    mid = rem // inner
    low = rem % inner
    return high * (N * block) + (c * dim_j + mid) * inner + low


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("shape", [(1, 1, 48, 64), (2, 1, 32, 64), (1, 1, 96, 64), (1, 1, 32, 96), (1, 1, 64, 128)])
def test_page_model(device, layout, shape):
    """buffer_num_pages matches the predicted page grid for both layouts."""
    tile = layout == ttnn.TILE_LAYOUT
    t = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grid = _page_grid(shape, tile)
    assert t.buffer_num_pages() == prod(grid), (
        f"{'TILE' if tile else 'RM'} {shape}: num_pages={t.buffer_num_pages()} "
        f"!= prod(grid={grid})={prod(grid)}  (page_size={t.buffer_page_size()})"
    )


@pytest.mark.parametrize("gd_neg", [-4, -3, -2, -1])
def test_out_page_formula_matches_concat(gd_neg):
    """Pure-arithmetic check (no device): the whole-page remap reconstructs a
    torch concat-along-gather_dim, for TILE layout, N=3 devices, shape a tile
    grid so each 'page' is a scalar cell we can place and compare."""
    N = 3
    # Use a tiny page grid directly as the tensor (each element == one page).
    B, C, Ht, Wt = 2, 2, 2, 2
    tile = True
    shape4 = (B, C, Ht * 32, Wt * 32)  # H,W tile-aligned so grid = [B,C,Ht,Wt]
    di = _dim_inner(shape4, tile, gd_neg)
    assert di is not None
    dim_j, inner = di
    P = B * C * Ht * Wt

    # Per-device shard: page p holds value (c, p). Build the gathered output by
    # the remap, and independently by torch.cat over the page grid.
    grid_dims = [B, C, Ht, Wt]
    axis = 4 + gd_neg  # page-grid axis being gathered
    shards = [torch.arange(P).reshape(grid_dims) + c * 1000 for c in range(N)]
    ref = torch.cat(shards, dim=axis)  # ground-truth gathered page grid
    ref_flat = ref.reshape(-1)

    got = torch.full((N * P,), -1, dtype=torch.long)
    for c in range(N):
        for p in range(P):
            got[_out_page(c, p, dim_j, inner, N)] = shards[c].reshape(-1)[p]
    assert torch.equal(got, ref_flat), f"gd={gd_neg}: remap != concat\n got={got}\n ref={ref_flat}"
