# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Phase-1b: the ttnn (device) block reorder matches the torch reference bit-for-bit and round-trips.

Validates the on-device reorder ops (reshape/pad/permute) before the Phase-2 kernel is wired in --
so any ttnn rank/layout surprise surfaces here, in isolation, not tangled with the kernel change."""
import pytest
import torch

import ttnn
from models.tt_dit.layers.block_permute import (
    from_block_order,
    from_block_order_tt,
    padded_grid,
    to_block_order,
    to_block_order_tt,
)

GRID_BLOCK = [
    ((8, 16, 16), (4, 4, 4)),  # no padding
    ((13, 20, 14), (5, 8, 6)),  # pads to 15x24x18
]


@pytest.mark.parametrize("grid,block", GRID_BLOCK)
def test_tt_to_block_matches_torch(*, device, grid, block):
    """Positions: unique values so any misplaced token is caught exactly."""
    t, h, w = grid
    nh = 2
    ref_in = torch.arange(1 * nh * t * h * w, dtype=torch.float32).reshape(1, nh, t * h * w, 1)
    want = to_block_order(ref_in, grid, block)

    tt_in = ttnn.from_torch(ref_in, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
    got = ttnn.to_torch(to_block_order_tt(tt_in, grid, block))

    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("grid,block", GRID_BLOCK)
def test_tt_round_trip_identity(*, device, grid, block):
    t, h, w = grid
    nh = 3
    torch.manual_seed(0)
    ref_in = torch.randn(1, nh, t * h * w, 8, dtype=torch.float32)

    tt = ttnn.from_torch(ref_in, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32)
    tt = to_block_order_tt(tt, grid, block)
    (tp, hp, wp), _ = padded_grid(grid, block)
    assert tuple(tt.shape) == (1, nh, tp * hp * wp, 8)
    out = ttnn.to_torch(from_block_order_tt(tt, grid, block))

    torch.testing.assert_close(out, ref_in)
    # sanity: the torch twin agrees end to end
    torch.testing.assert_close(from_block_order(to_block_order(ref_in, grid, block), grid, block), ref_in)
