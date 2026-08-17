# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.base_fpu import Fpu
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from fuser.tile_loop import LoopTileByTile, TileLoop
from helpers.llk_params import ReduceDimension


class ReduceBlockMaxFpu(Fpu):
    loop: TileLoop = LoopTileByTile()
    reduce_dim: ReduceDimension = ReduceDimension.Row

    per_block_init = True

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        dest_acc = config.dest_acc.cpp_enum_value
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        return f"_llk_math_reduce_block_max_row_init_<{ct_dim}, {dest_acc}>({tensor_shape});\n"

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        dest_acc = config.dest_acc.cpp_enum_value
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        tile_x_in_block = f"(({block.tile_id_block}) % {block.block_tiles_x})"
        tile_y_in_block = f"(({block.tile_id_block}) / {block.block_tiles_x})"
        dest_expr = f"(({tile_y_in_block}) * {block.block_tiles_x})"
        return (
            f"if (({tile_x_in_block}) % {ct_dim} == 0 ) {{\n"
            f"    _llk_math_reduce_block_max_row_<{ct_dim}, {dest_acc}>({dest_expr}, {tensor_shape});\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_math_reduce_block_max_row_uninit_();\n"

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.reduce_golden(
            tensor_a, tensor_b, tensor_dst, config, operation, compute_unit, True
        )

    def get_headers(self) -> List[str]:
        return ["experimental/llk_math_reduce_custom.h"]
