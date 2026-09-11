# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.base_fpu import Fpu
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.fpu.matmul import matmul_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation


class MatmulFpu(Fpu):
    granularity = InvocationGranularity.BLOCK
    per_block_init = True
    golden_fn = staticmethod(matmul_golden)

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_matmul.h",
        ]

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value
        rt_dim = block.block_rows
        ct_dim = block.block_cols

        return (
            f"// Operation {stage}: Matmul FPU\n"
            f"_llk_math_matmul_init_<{math_fidelity}>({ct_dim}, {rt_dim});\n"
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        rt_dim = block.block_rows
        ct_dim = block.block_cols
        num_cols = compute_unit.src_a.tile_shape.total_col_dim()
        kt_dim = compute_unit.src_a.dimensions[1] // num_cols

        return (
            f"for (std::uint32_t kt = 0; kt < {kt_dim}; kt++)\n"
            f"{{\n"
            f"    _llk_math_matmul_block_({ct_dim}, {rt_dim});\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
