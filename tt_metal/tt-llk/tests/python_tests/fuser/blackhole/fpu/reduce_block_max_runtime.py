# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.block_data import BlockData
from fuser.compute_node import ComputeNode
from fuser.fused_loop import FusedLoop, LoopBlockRow
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig

from .reduce_block_max import ReduceBlockMaxFpu


class ReduceBlockMaxRuntimeFpu(ReduceBlockMaxFpu):
    loop: FusedLoop = LoopBlockRow()

    def get_headers(self) -> List[str]:
        return [
            "experimental/llk_math_reduce_custom.h",
            "experimental/llk_math_reduce_runtime_custom.h",
        ]

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        dest_acc = config.dest_acc.cpp_enum_value
        return f"_llk_math_reduce_block_max_row_init_runtime_<{dest_acc}>({ct_dim});\n"

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        dest_acc = config.dest_acc.cpp_enum_value
        return f"_llk_math_reduce_block_max_row_runtime_<{dest_acc}>({block.tile_id_block});\n"

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        return "_llk_math_reduce_block_max_row_uninit_runtime_();\n"
