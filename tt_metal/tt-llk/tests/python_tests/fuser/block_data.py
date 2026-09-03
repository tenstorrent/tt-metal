# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union

Index = Optional[Union[int, str]]


class InvocationGranularity(Enum):
    NONE = auto()
    TILE = auto()
    ROW = auto()
    BLOCK = auto()


@dataclass(frozen=True)
class KernelInvocation:
    in0: Index = None
    in1: Index = None
    src0: Index = None
    src1: Index = None
    dest: Index = None
    out: Index = None


@dataclass
class BlockData:
    """Per-block metadata used by fused unpack/math/pack loops.

    Coordinates are in tile units. The block describes a rectangular region of
    tiles within the output tile grid, including tail blocks at the edges.
    """

    block_x: Index  # Block origin tile x in the output grid.
    block_y: Index  # Block origin tile y in the output grid.
    block_tiles_x: Index  # Block width in tiles (x dimension).
    block_tiles_y: Index  # Block height in tiles (y dimension).
    tile_count_x: Index  # Total tile count along x for the output.
    tile_count_y: Index  # Total tile count along y for the output.
    full_x_limit: Index  # Exclusive x limit for full blocks region.
    full_y_limit: Index  # Exclusive y limit for full blocks region.
    tile_id_global: Index  # Global tile id in L1 (row-major).
    tile_id_block: Index  # Tile id within the current block.
    tile_id_src_a: Index = None  # First Dest source tile id for binary SFPU.
    tile_id_src_b: Index = None  # Second source tile id for unpack and binary SFPU.
    codegen: bool = False  # Whether indexes contain generated C++ expressions.
    loop_x: bool = False  # Whether generated code loops over block columns.
    loop_y: bool = False  # Whether generated code loops over block rows.

    @property
    def tile_y(self) -> Index:
        """Current invocation's tile-row offset within the block."""
        if self.tile_id_block is None or self.block_tiles_x is None:
            return None
        if isinstance(self.tile_id_block, str) or isinstance(self.block_tiles_x, str):
            return f"({self.tile_id_block}) / {self.block_tiles_x}"
        return self.tile_id_block // self.block_tiles_x
