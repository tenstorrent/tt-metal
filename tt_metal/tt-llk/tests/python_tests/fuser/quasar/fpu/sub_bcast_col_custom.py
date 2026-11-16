# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.fpu.sub_bcast_col_custom import sub_bcast_col_custom_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from helpers.llk_params import MathOperation

from .eltwise import EltwiseFpu


class SubBcastColCustomFpu(EltwiseFpu):
    granularity = InvocationGranularity.ROW
    per_block_init = True

    def __init__(self):
        super().__init__(MathOperation.Elwsub)
        self.golden_fn = sub_bcast_col_custom_golden

    def get_headers(self) -> List[str]:
        return ["experimental/llk_math_eltwise_binary_custom.h"]

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        return (
            "_llk_math_eltwise_binary_init_custom_<ckernel::EltwiseBinaryType::ELWSUB, ckernel::BroadcastType::COL>"
            f"({tensor_shape});\n"
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        return f"_llk_math_sub_bcast_cols_reuse_custom_({block.block_cols}, {tensor_shape}, {block.tile_id_dest});\n"
