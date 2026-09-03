# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Union

Index = Union[int, str]


@dataclass
class BlockData:
    """Per-block metadata used by fused unpack/math/pack loops.

    Coordinates are in tile units. The block describes a rectangular region of
    tiles within the output tile grid, including tail blocks at the edges.

    The ``tile_id_*`` fields carry the current invocation's indices to the unit
    being emitted. Codegen fills them with C++ expressions in the enclosing loop
    variables; they are rewritten before every call.
    """

    block_x: Index
    block_y: Index
    block_tiles_x: Index
    block_tiles_y: Index
    tile_count_x: Index
    tile_count_y: Index
    full_x_limit: Index
    full_y_limit: Index
    tile_id_global: Index
    tile_id_block: Index
    tile_id_src_a: Optional[Index] = None
    tile_id_src_b: Optional[Index] = None
