# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Tuple

import torch
from helpers.llk_params import (
    DestSync,
    StochasticRounding,
    Tilize,
)

from .fused_math import ComputePipeline
from .fused_operand import Operand


@dataclass
class FusedOperation:
    math: ComputePipeline
    output: Operand
    stage_id: int = 0
    num_stages: int = 1
    unpack_to_dest: bool = False
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

        dimensions = [0, 0]
        for op in self.math.operations:
            if op.src_a is not None:
                dimensions[0] = max(dimensions[0], op.src_a.dimensions[0])
                dimensions[1] = max(dimensions[1], op.src_a.dimensions[1])
            if op.src_b is not None:
                dimensions[0] = max(dimensions[0], op.src_b.dimensions[0])
                dimensions[1] = max(dimensions[1], op.src_b.dimensions[1])
        self.max_output_dimensions = dimensions

    def unpack(self, config) -> str:
        return self.math.unpack_body(self, config)

    def do_math(self, config) -> str:
        return self.math.math_body(self, config)

    def pack(self, config) -> str:
        return self.math.pack_body(self, config)

    def golden(self, config) -> torch.Tensor:
        # calculate l1 golden
        src_a_dims = self.math.operations[0].src_a.dimensions
        src_b_dims = self.math.operations[0].src_b.dimensions

        tensor_a = self.math.operations[0].src_a.raw_data.view(src_a_dims)
        tensor_b = self.math.operations[0].src_b.raw_data.view(src_b_dims)

        l1_golden_tensor = self.math.golden(tensor_a, tensor_b, self, config)
        l1_golden_tensor = self.math.packer().golden(l1_golden_tensor, self, config)

        self.output.l1_golden = l1_golden_tensor.flatten()

        # calculate master golden
        golden_tensor_a = self.math.operations[0].src_a.master_golden.view(src_a_dims)
        golden_tensor_b = self.math.operations[0].src_b.master_golden.view(src_b_dims)

        master_golden_tensor = self.math.golden(
            golden_tensor_a, golden_tensor_b, self, config
        )
        master_golden_tensor = self.math.packer().golden(
            master_golden_tensor, self, config
        )

        self.output._master_golden = master_golden_tensor.flatten()

        return master_golden_tensor

    def __str__(self):
        return (
            f"\n{'='*60}\n"
            f"Operation {self.stage_id}\n"
            f"{'='*60}\n"
            f"  {self.math}\n"
            f"  Output: {self.output}\n"
            f"  Block Size: {self.block_size}\n"
            f"  Dest Sync: {self.dest_sync}\n"
        )
