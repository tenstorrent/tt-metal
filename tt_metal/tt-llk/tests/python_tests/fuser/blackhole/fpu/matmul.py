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
from fuser.tile_loop import LoopBlock, TileLoop


class MatmulFpu(Fpu):
    loop: TileLoop = LoopBlock()
    per_block_init = True

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_matmul.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.matmul_golden(
            tensor_a, tensor_b, tensor_dst, config, operation, compute_unit
        )

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value
        transpose = compute_unit.transpose_within_face.cpp_enum_value
        rt_dim = block.block_tiles_y
        ct_dim = block.block_tiles_x

        partial_face = compute_unit.src_a.partial_face.cpp_enum_value
        tile_r_dim_a = compute_unit.src_a.tile_shape.total_row_dim()
        tile_c_dim_a = compute_unit.src_a.tile_shape.total_col_dim()
        tile_r_dim_b = compute_unit.src_b.tile_shape.total_row_dim()
        tile_c_dim_b = compute_unit.src_b.tile_shape.total_col_dim()

        return (
            f"// Operation {stage}: Matmul FPU\n"
            f"_llk_math_matmul_init_<{math_fidelity}>(\n"
            f"    {tile_r_dim_a}, {tile_c_dim_a}, {tile_r_dim_b}, {tile_c_dim_b}, {partial_face}, {transpose}, {ct_dim}, {rt_dim}\n"
            f");\n"
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        rt_dim = block.block_tiles_y
        ct_dim = block.block_tiles_x
        num_cols = compute_unit.src_a.tile_shape.total_col_dim()
        kt_dim = compute_unit.src_a.dimensions[1] // num_cols
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value

        return (
            f"for (std::uint32_t kt = 0; kt < {kt_dim}; kt++)\n"
            f"{{\n"
            f"    _llk_math_matmul_<{math_fidelity}>({block.tile_id_block}, {ct_dim}, {rt_dim});\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_math_matmul_uninit_();\n"
