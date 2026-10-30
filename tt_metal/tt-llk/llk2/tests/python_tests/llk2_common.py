# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What the LLK 2.0 tests share: tile geometry, Dest limits and stimulus layout helpers."""

from __future__ import annotations

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.tilize_untilize import tilize_block

TILE_DIM = 32  # llk2::default_tile_dim
TILE_DATUMS = TILE_DIM * TILE_DIM
FACES_PER_TILE = 4  # llk2::faces_per_full_tile: a 32x32 tile is 2 x 2 faces of 16 x 16

# Output tiles that fit one Dest section (the tests use half sync): 8 with a 16-bit Dest, 4 with a 32-bit one.
DEST_HALF_TILES = {DestAccumulation.No: 8, DestAccumulation.Yes: 4}


def unwrap_case(case):
    """parametrize hands a lone parameter over wrapped in a one-element tuple: unwrap it."""
    if isinstance(case, tuple) and len(case) == 1:
        (case,) = case
    return case


def tilize_operands(src_A, dims_A, src_B, dims_B, input_format):
    """The two operands tilized and flat, as the unpackers read them. Bfp8_b stimuli are already tiles."""
    if input_format == DataFormat.Bfp8_b:
        return src_A.flatten(), src_B.flatten()
    return (
        tilize_block(src_A, dimensions=dims_A, stimuli_format=input_format).flatten(),
        tilize_block(src_B, dimensions=dims_B, stimuli_format=input_format).flatten(),
    )


def blocks_in_pop_order(tiles, tiles_y, tiles_x, rows, cols, order):
    """The [tiles_y, tiles_x] tile grid cut into [rows, cols] blocks, taken in the given (block row, block col) order."""
    grid = tiles.view(tiles_y, tiles_x, TILE_DATUMS)
    return torch.cat(
        [
            grid[r * rows : (r + 1) * rows, c * cols : (c + 1) * cols].flatten()
            for r, c in order
        ]
    )
