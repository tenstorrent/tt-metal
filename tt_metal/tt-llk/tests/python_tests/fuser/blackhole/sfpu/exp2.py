# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from fuser.block_data import BlockData
from fuser.compute_node import ComputeNode
from fuser.fused_operation import FusedOperation
from fuser.fused_sfpu import Sfpu
from fuser.fuser_config import GlobalConfig
from helpers.golden_generators import (
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    MathOperation,
)


class Exp2Sfpu(Sfpu):
    def __init__(
        self,
        approx_mode: ApproximationMode = ApproximationMode.No,
        iterations: int = 8,
        dest_idx: int = 0,
        fill_const_value=5,
    ):
        self.iterations = iterations
        self.approx_mode = approx_mode
        self.dest_idx = dest_idx
        self.fill_const_value = fill_const_value

    def get_headers(self) -> List[str]:
        return [
            "ckernel_defs.h",
            "ckernel_sfpu.h",
            "llk_math_common.h",
            "llk_math_eltwise_unary_sfpu.h",
            "sfpu_operations.h",
        ]

    def golden(
        self,
        tensor: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        batch_dims: tuple,
        batch_tile_cnt: int,
    ) -> torch.Tensor:
        format_input = operation.output.data_format
        format_output = operation.output.data_format
        dest_acc = config.dest_acc

        # For exp2, we use the standard golden generator but with exp2 operation
        # The actual exp2 operation would need to be added to MathOperation enum
        # For now, we'll compute it manually in the test
        return torch.exp2(tensor)

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.cpp_enum_value
        approx_mode = self.approx_mode.cpp_enum_value
        # Note: We would need to add EXP2 to MathOperation enum for this to work
        # For now, we'll use a placeholder that would need to be updated
        op = f"SfpuType::EXP2"  # This would need to be added to the enum

        return (
            f"    // Operation {stage}: Unary EXP2 SFPU\n"
            f"    test_utils::call_unary_sfpu_operation_init<{op}, {approx_mode}, {dest_acc}, {self.iterations}>();\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.cpp_enum_value
        approx_mode = self.approx_mode.cpp_enum_value
        op = f"SfpuType::EXP2"  # This would need to be added to the enum

        return (
            f"    test_utils::call_unary_sfpu_operation<"
            f"dest_sync{stage}, {dest_acc}, "
            f"{op}, {approx_mode}, {dest_acc}, {self.iterations}"
            f">({self.dest_idx}, {config.sentinel.math_format}, {self.fill_const_value});\n"
        )

    def __str__(self) -> str:
        return f"Exp2Sfpu()"