# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Union


@dataclass
class BlockData:
    """Per-block metadata used by fused unpack/math/pack loops.

    Coordinates are in tile units. The block describes a rectangular region of
    tiles within the output tile grid, including tail blocks at the edges.
    """

    block_x: Union[int, str]  # Block origin tile x in the output grid.
    block_y: Union[int, str]  # Block origin tile y in the output grid.
    block_tiles_x: Union[int, str]  # Block width in tiles (x dimension).
    block_tiles_y: Union[int, str]  # Block height in tiles (y dimension).
    tile_count_x: Union[int, str]  # Total tile count along x for the output.
    tile_count_y: Union[int, str]  # Total tile count along y for the output.
    full_x_limit: Union[int, str]  # Exclusive x limit for full blocks region.
    full_y_limit: Union[int, str]  # Exclusive y limit for full blocks region.
    tile_id_global: Union[int, str]  # Global tile id in L1 (row-major).
    tile_id_block: Union[int, str]  # Tile id within the current block.
