# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.base_fpu import Fpu
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.fpu.reduce_block_max_row import reduce_block_max_row_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from helpers.llk_params import ReduceDimension


class ReduceBlockMaxRuntimeFpu(Fpu):
    granularity = InvocationGranularity.ROW
    per_block_init = True
    reduce_dim = ReduceDimension.Row
    golden_fn = staticmethod(reduce_block_max_row_golden)

    def get_headers(self) -> List[str]:
        return ["experimental/llk_math_reduce_runtime_custom.h"]

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        dest_acc = config.dest_acc.cpp_enum_value
        return f"_llk_math_reduce_block_max_row_init_runtime_<{dest_acc}>({block.block_cols}, {tensor_shape});\n"

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        dest_acc = config.dest_acc.cpp_enum_value
        return f"_llk_math_reduce_block_max_row_runtime_<{dest_acc}>({block.tile_id_dest}, {tensor_shape});\n"

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_math_reduce_block_max_row_uninit_runtime_();\n"
