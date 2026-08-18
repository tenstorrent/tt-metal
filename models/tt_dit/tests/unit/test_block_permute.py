# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Phase-1 verification of the 3-D block token permutation (host reorder machinery), no device.

Proves three things before the kernel (Phase 2) is touched:
  * the reorder is lossless (block order round-trips back to strided exactly),
  * the tensor reshape/permute agrees with the scalar coordinate map the kernel will port, and
  * the premise holds -- a block-order query chunk has a far smaller neighborhood box than the strided
    (strip) chunk it replaces.
"""
import pytest
import torch

from models.tt_dit.layers.block_permute import (
    block_to_token_index,
    from_block_order,
    padded_grid,
    to_block_order,
    token_to_block_coords,
)

GRID_BLOCK = [
    ((8, 16, 16), (4, 4, 4)),  # divides exactly, no padding
    ((13, 20, 14), (5, 8, 6)),  # every axis needs padding -> 15x24x18
    ((145, 272, 60), (5, 8, 12)),  # the real 6s per-shard grid + the zero-pad Phase-0 pick
]


def _nbr_shift_start(q, length, ker):
    ker = min(ker, length)
    half, last = ker // 2, length - ker
    return min(q - half if q >= half else 0, last)


def _window_box(coords, grid, kernel):
    """Volume of the bounding box of the inward-shifted windows over a set of physical coords."""

    def span(cs, length, ker):
        starts = [_nbr_shift_start(c, length, ker) for c in cs]
        return max(s + min(ker, length) for s in starts) - min(starts)

    axes = list(zip(*coords))
    return span(axes[0], grid[0], kernel[0]) * span(axes[1], grid[1], kernel[1]) * span(axes[2], grid[2], kernel[2])


@pytest.mark.parametrize("grid,block", GRID_BLOCK)
def test_round_trip_identity(grid, block):
    t, h, w = grid
    torch.manual_seed(0)
    x = torch.randn(2, 3, t * h * w, 5)  # (B, NH, S, HD)-shaped
    y = to_block_order(x, grid, block)
    (tp, hp, wp), _ = padded_grid(grid, block)
    assert tuple(y.shape) == (2, 3, tp * hp * wp, 5)
    torch.testing.assert_close(from_block_order(y, grid, block), x)


@pytest.mark.parametrize("grid,block", GRID_BLOCK)
def test_coord_map_is_a_bijection(grid, block):
    (tp, hp, wp), counts = padded_grid(grid, block)
    sp = tp * hp * wp
    seen = bytearray(sp)
    for t in range(tp):
        for h in range(hp):
            for w in range(wp):
                p = block_to_token_index(t, h, w, block, counts)
                assert 0 <= p < sp and not seen[p], f"collision/oob at {(t, h, w)} -> {p}"
                seen[p] = 1
                assert token_to_block_coords(p, block, counts) == (t, h, w)
    assert all(seen), "block order does not cover every padded token"


@pytest.mark.parametrize("grid,block", GRID_BLOCK[:2])
def test_tensor_permute_matches_coord_map(grid, block):
    t, h, w = grid
    (tp, hp, wp), counts = padded_grid(grid, block)
    # strided token id encoded as the value, so we can read back where each landed
    vals = torch.arange(t * h * w).reshape(1, t * h * w, 1).float()
    blk = to_block_order(vals, grid, block).reshape(-1)
    for t_, h_, w_ in [(0, 0, 0), (min(3, t - 1), min(5, h - 1), min(7, w - 1)), (t - 1, h - 1, w - 1)]:
        strided_id = (t_ * h + h_) * w + w_
        p = block_to_token_index(t_, h_, w_, block, counts)
        assert blk[p].item() == strided_id, f"token {(t_, h_, w_)} misplaced in block order"


@pytest.mark.parametrize("grid,block", GRID_BLOCK)
def test_block_chunk_box_beats_strip(grid, block):
    """A block-order chunk's box is far smaller than the strided-order chunk it replaces."""
    kernel = (11, 11, 11)
    (tp, hp, wp), counts = padded_grid(grid, block)
    q = block[0] * block[1] * block[2]  # q_chunk = one block

    # block chunk: the first block's coords
    block_coords = [token_to_block_coords(p, block, counts) for p in range(q)]
    block_box = _window_box(block_coords, (tp, hp, wp), kernel)

    # strided chunk: the first q tokens in (t outer, h mid, w inner) order
    gw, gh = grid[2], grid[1]
    strided_coords = [(i // (gw * gh), (i // gw) % gh, i % gw) for i in range(q)]
    strided_box = _window_box(strided_coords, grid, kernel)

    assert block_box < strided_box, f"block box {block_box} not < strip box {strided_box}"
    print(
        f"\n  grid={grid} block={block} q={q}: strip box={strided_box}  block box={block_box}  "
        f"({strided_box / block_box:.1f}x smaller)"
    )
