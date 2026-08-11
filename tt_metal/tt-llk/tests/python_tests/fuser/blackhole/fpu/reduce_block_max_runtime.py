# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from fuser.tile_loop import LoopBlockRow, TileLoop

from .reduce_block_max import ReduceBlockMaxFpu


class ReduceBlockMaxRuntimeFpu(ReduceBlockMaxFpu):
    loop: TileLoop = LoopBlockRow()

    def get_headers(self) -> List[str]:
        return [
            "experimental/llk_math_reduce_custom.h",
            "experimental/llk_math_reduce_runtime_custom.h",
        ]

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
        return f"_llk_math_reduce_block_max_row_init_runtime_<{dest_acc}>({ct_dim}, {tensor_shape});\n"

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        dest_acc = config.dest_acc.cpp_enum_value
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        return f"_llk_math_reduce_block_max_row_runtime_<{dest_acc}>({block.tile_id_block}, {tensor_shape});\n"

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_math_reduce_block_max_row_uninit_runtime_();\n"
