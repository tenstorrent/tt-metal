# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fused_fpu import Fpu
from fuser.fused_loop import FusedLoop, LoopBlock
from fuser.fused_math import ComputeNode
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig
from helpers.golden_generators import MatmulGolden, get_golden_generator


class MatmulFpu(Fpu):
    loop: FusedLoop = LoopBlock()

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
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = operation.output.data_format
        math_fidelity = compute_unit.math_fidelity

        generate_golden = get_golden_generator(MatmulGolden)
        golden = generate_golden(
            tensor_a,
            tensor_b,
            output_format,
            math_fidelity,
            input_A_dimensions=operation.src_a.dimensions,
            input_B_dimensions=operation.src_b.dimensions,
            tilize=False,
            input_A_format=operation.src_a.data_format,
            input_B_format=operation.src_b.data_format,
        )

        return (tensor_a, tensor_b, golden)

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

        return (
            f"// Operation {stage}: Matmul FPU\n"
            f"_llk_math_matmul_init_<{math_fidelity}>(\n"
            f"    TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, {transpose}, {ct_dim}, {rt_dim}\n"
            f");\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        rt_dim = block.block_tiles_y
        ct_dim = block.block_tiles_x
        kt_dim = operation.kt_dim
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value

        return (
            f"for (std::uint32_t kt = 0; kt < {kt_dim}; kt++)\n"
            f"{{\n"
            f"    _llk_math_matmul_<{math_fidelity}>({block.tile_id_block}, {ct_dim}, {rt_dim});\n"
            f"}}\n"
        )
