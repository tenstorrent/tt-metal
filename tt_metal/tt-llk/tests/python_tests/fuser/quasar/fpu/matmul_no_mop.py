# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation

from .matmul import MatmulFpu


class MatmulNoMopFpu(MatmulFpu):
    def get_headers(self) -> List[str]:
        return ["experimental/llk_math_matmul_custom_no_mop.h"]

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value
        return f"_llk_math_matmul_init_no_mop_<{math_fidelity}>({block.block_cols}, {block.block_rows});\n"

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value
        kt_dim = (
            compute_unit.src_a.dimensions[1]
            // compute_unit.src_a.tile_shape.total_col_dim()
        )
        return (
            f"for (std::uint32_t kt = 0; kt < {kt_dim}; kt++)\n"
            f"{{\n"
            f"    _llk_math_matmul_block_no_mop_<{math_fidelity}>({block.block_cols}, {block.block_rows});\n"
            f"}}\n"
        )
