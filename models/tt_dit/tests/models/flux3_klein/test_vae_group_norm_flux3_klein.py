# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only check of the reference encoder's GroupNorm blocking (normalization.bounded_group_norm_out_blocks with
vae_flux3_klein.GROUP_NORM_MAX_BLOCK_ROWS). No device.

A 1600x912 reference reaches the second down level as 400x228 = 2850 tile rows, which only a 6x5 grid splits evenly;
the op's auto choice then keeps 95-row blocks whose static CBs collide with the pipeline's resident L1. Those blocks
are split to at most GROUP_NORM_MAX_BLOCK_ROWS rows; shapes whose auto blocks already fit keep the auto choice (-1).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ....layers.normalization import bounded_group_norm_out_blocks
from ....models.vae.vae_flux3_klein import GROUP_NORM_MAX_BLOCK_ROWS


def _norm(grid_x: int, grid_y: int, num_virtual_cols: int) -> SimpleNamespace:
    """What bounded_group_norm_out_blocks reads from a GroupNorm: its grid for the input and its virtual columns."""
    grid = SimpleNamespace(x=grid_x, y=grid_y)
    return SimpleNamespace(_op_core_grid=lambda batch, spatial: grid, num_virtual_cols=num_virtual_cols)


@pytest.mark.parametrize(
    "grid, nvc, spatial, channels, expected",
    [
        ((6, 5), 2, 400 * 228, 64, 5),
        ((8, 8), 2, 256 * 256, 64, -1),
        ((8, 8), 1, 1024 * 1024, 32, -1),
        ((8, 8), 1, 2048 * 2048, 32, -1),
    ],
)
def test_bounded_group_norm_out_blocks(grid, nvc, spatial, channels, expected):
    norm = _norm(*grid, nvc)
    blocks = bounded_group_norm_out_blocks(norm, 1, spatial, channels, GROUP_NORM_MAX_BLOCK_ROWS)
    assert blocks == expected
    if blocks != -1:
        block_h = -(-spatial // 32) // ((grid[0] // nvc) * grid[1])
        assert block_h % blocks == 0 and block_h // blocks <= GROUP_NORM_MAX_BLOCK_ROWS
