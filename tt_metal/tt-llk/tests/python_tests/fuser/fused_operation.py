# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple

from helpers.llk_params import (
    DestSync,
    GoldenType,
    ReduceDimension,
    StochasticRounding,
    Tilize,
)
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.tile_shape import TileShape, construct_tile_shape

from .fused_math import ComputePipeline


@dataclass
class FusedOperation:
    math: ComputePipeline
    max_output_dimensions: Tuple[int, int]
    tile_shape: TileShape = construct_tile_shape(
        (DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM)
    )
    stage_id: int = 0
    num_stages: int = 1
    throttle: int = 0
    stochastic_rnd: StochasticRounding = StochasticRounding.No
    tiny_tiles: bool = False
    dest_sync: DestSync = DestSync.Half
    block_size: Tuple[int, int] = (32, 32)
    reduce_dim: Optional[ReduceDimension] = None
    bh_tilize: Tilize = Tilize.No

    def __post_init__(self):
        self.block_tiles_x = self.block_size[1] // self.tile_shape.total_col_dim()
        self.block_tiles_y = self.block_size[0] // self.tile_shape.total_row_dim()

    def unpack(self, config) -> str:
        return self.math.unpack_body(self, config)

    def do_math(self, config) -> str:
        return self.math.math_body(self, config)

    def pack(self, config) -> str:
        return self.math.pack_body(self, config)

    def golden(self, config):
        # calculate l1 golden
        self.math.golden(self, config, GoldenType.L1_GOLDEN)

        # calculate master golden
        self.math.golden(self, config, GoldenType.MASTER_GOLDEN)

    def __str__(self):
        return (
            f"\n{'=' * 60}\n"
            f"Operation {self.stage_id}\n"
            f"{'=' * 60}\n"
            f"  {self.math}\n"
            f"  Block Size: {self.block_size}\n"
            f"  Dest Sync: {self.dest_sync}\n"
        )
