# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Tuple

import torch
from helpers.llk_params import (
    DestSync,
    GoldenType,
    StochasticRounding,
    Tilize,
)

from .fused_math import ComputePipeline
from .fused_operand import Operand


@dataclass
class FusedOperation:
    math: ComputePipeline
    output: Operand
    max_output_dimensions: Tuple[int, int]
    stage_id: int = 0
    num_stages: int = 1
    throttle: int = 0
    stochastic_rnd: StochasticRounding = StochasticRounding.No
    tiny_tiles: bool = False
    dest_sync: DestSync = DestSync.Half
    block_size: Tuple[int, int] = (32, 32)
    bh_tilize: Tilize = Tilize.No

    def __post_init__(self):
        num_rows = self.output.tile_shape.total_row_dim()
        num_cols = self.output.tile_shape.total_col_dim()

        self.block_tiles_x = self.block_size[1] // num_cols
        self.block_tiles_y = self.block_size[0] // num_rows

    def unpack(self, config) -> str:
        return self.math.unpack_body(self, config)

    def do_math(self, config) -> str:
        return self.math.math_body(self, config)

    def pack(self, config) -> str:
        return self.math.pack_body(self, config)

    def golden(self, config) -> torch.Tensor:
        # calculate l1 golden
        l1_golden_tensor = self.math.golden(
            self, config, golden_type=GoldenType.L1_GOLDEN
        )
        l1_golden_tensor = self.math.packer().golden(l1_golden_tensor, self, config)

        self.output.l1_golden = l1_golden_tensor.flatten()

        # calculate master golden
        master_golden_tensor = self.math.golden(
            self, config, golden_type=GoldenType.MASTER_GOLDEN
        )
        master_golden_tensor = self.math.packer().golden(
            master_golden_tensor, self, config
        )

        self.output._master_golden = master_golden_tensor.flatten()

        return master_golden_tensor

    def __str__(self):
        return (
            f"\n{'=' * 60}\n"
            f"Operation {self.stage_id}\n"
            f"{'=' * 60}\n"
            f"  {self.math}\n"
            f"  Output: {self.output}\n"
            f"  Block Size: {self.block_size}\n"
            f"  Dest Sync: {self.dest_sync}\n"
        )
