# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.block_data import BlockData
from fuser.fused_loop import FusedLoop, LoopBlock
from fuser.fused_math import ComputeNode
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig

from .matmul import MatmulFpu


class MatmulNoMopFpu(MatmulFpu):
    loop: FusedLoop = LoopBlock()
    per_block_init = True

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_matmul.h",
            "experimental/llk_math_matmul_custom_no_mop.h",
        ]

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value
        transpose = "true" if compute_unit.unpack_transpose_faces.value else "false"
        rt_dim = block.block_tiles_y
        ct_dim = block.block_tiles_x

        tile_r_dim_a = compute_unit.src_a.tile_shape.total_row_dim()
        tile_c_dim_a = compute_unit.src_a.tile_shape.total_col_dim()
        tile_r_dim_b = compute_unit.src_b.tile_shape.total_row_dim()
        tile_c_dim_b = compute_unit.src_b.tile_shape.total_col_dim()
        partial_face = compute_unit.src_a.partial_face.cpp_enum_value

        return (
            f"// Operation {stage}: MatmulNoMop FPU\n"
            f"_llk_math_matmul_init_no_mop_<{math_fidelity}>(\n"
            f"    {tile_r_dim_a}, {tile_c_dim_a}, {tile_r_dim_b}, {tile_c_dim_b}, {partial_face}, {transpose}, {ct_dim}, {rt_dim}\n"
            f");\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value
        rt_dim = block.block_tiles_y
        ct_dim = block.block_tiles_x
        num_cols = compute_unit.src_a.tile_shape.total_col_dim()
        kt_dim = compute_unit.src_a.dimensions[1] // num_cols

        return (
            f"for (std::uint32_t kt = 0; kt < {kt_dim}; kt++)\n"
            f"{{\n"
            f"    _llk_math_matmul_no_mop_<{math_fidelity}>({block.tile_id_block}, {ct_dim}, {rt_dim});\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        return "_llk_math_matmul_uninit_no_mop_();\n"
