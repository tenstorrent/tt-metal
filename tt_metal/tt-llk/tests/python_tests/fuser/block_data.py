# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Union

Index = Union[int, str]


@dataclass
class BlockData:
    """Per-block metadata used by fused unpack/math/pack loops.

    Coordinates are in tile units. block_origin_* is the block's top-left tile in
    the output grid; block_cols/block_rows are its size in tiles (tail blocks
    included). The remaining fields are per-call indices: each unit's *_call fills
    the ones it needs (as C++ loop-var expressions) before every emit, so they
    stay None until set and are only meaningful for the emitting unit.
    """

    block_origin_x: Index
    block_origin_y: Index
    block_cols: Index
    block_rows: Index
    tile_id_dest: Optional[Index] = None
    tile_id_src_a: Optional[Index] = None
    tile_id_src_b: Optional[Index] = None
    tile_id_out: Optional[Index] = None
    dest_src0: Optional[Index] = None
    dest_src1: Optional[Index] = None
