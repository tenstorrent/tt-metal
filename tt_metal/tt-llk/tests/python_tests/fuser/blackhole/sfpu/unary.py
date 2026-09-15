# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from fuser.base_sfpu import Sfpu
from fuser.block_data import BlockData
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from fuser.sfpu_node import SfpuNode
from helpers.llk_params import (
    ApproximationMode,
    MathOperation,
)


class UnarySfpu(Sfpu):
    def __init__(
        self,
        operation: MathOperation,
        approx_mode: ApproximationMode = ApproximationMode.No,
        iterations: int = 8,
        dest_idx: int = 0,
        fill_const_value=5,
    ):
        if not operation in MathOperation.get_sfpu_unary_operations():
            raise ValueError(
                f"Operation {operation} is not a valid SFPU unary operation."
            )
        self.iterations = iterations
        self.approx_mode = approx_mode
        self.operation = operation
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
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: SfpuNode,
        batch_dims: tuple,
        batch_tile_cnt: int,
    ) -> torch.Tensor:
        return self.unary_sfpu_golden(
            tensor, config, operation, compute_unit, batch_dims
        )

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: SfpuNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.cpp_enum_value
        approx_mode = self.approx_mode.cpp_enum_value
        op = f"SfpuType::{self.operation.cpp_enum_value}"

        return (
            f"    // Operation {stage}: Unary {self.operation.cpp_enum_value} SFPU\n"
            f"    test_utils::call_unary_sfpu_operation_init<{op}, {approx_mode}, {dest_acc}, {self.iterations}>();\n"
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: SfpuNode,
        block: BlockData,
    ) -> str:
        dest_sync = operation.dest_sync.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        approx_mode = self.approx_mode.cpp_enum_value
        op = f"SfpuType::{self.operation.cpp_enum_value}"

        return (
            f"    test_utils::call_unary_sfpu_operation<"
            f"{dest_sync}, {dest_acc}, "
            f"{op}, {approx_mode}, {dest_acc}, {self.iterations}"
            f">({self.dest_idx}, {config.sentinel.math_format}, {self.fill_const_value});\n"
        )

    def __str__(self) -> str:
        return f"UnarySfpu({self.operation})"
