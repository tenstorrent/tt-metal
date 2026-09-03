# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fused_fpu import Fpu
from fuser.fused_loop import FusedLoop, LoopBlock
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig
from helpers.golden_generators import MatmulGolden, get_golden_generator


class MatmulFpu(Fpu):
    loop: FusedLoop = LoopBlock()
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
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = config.sentinel.golden_math_format
        math_fidelity = compute_unit.math_fidelity

        generate_golden = get_golden_generator(MatmulGolden)
        golden = generate_golden(
            tensor_a,
            tensor_b,
            output_format,
            math_fidelity,
            input_A_dimensions=compute_unit.src_a.dimensions,
            input_B_dimensions=compute_unit.src_b.dimensions,
            tilize=False,
            input_A_format=compute_unit.src_a.data_format,
            input_B_format=compute_unit.src_b.data_format,
        )

        return (tensor_a, tensor_b, golden)

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        math_fidelity = compute_unit.math_fidelity.cpp_enum_value
        rt_dim = block.block_tiles_y
        ct_dim = block.block_tiles_x

        return (
            f"// Operation {stage}: Matmul FPU\n"
            f"_llk_math_matmul_init_<{math_fidelity}>({ct_dim}, {rt_dim});\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        rt_dim = block.block_tiles_y
        ct_dim = block.block_tiles_x
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
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
