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
        fill_const_value=5,
    ):
        if not operation in MathOperation.get_sfpu_unary_operations():
            raise ValueError(
                f"Operation {operation} is not a valid SFPU unary operation."
            )
        self.iterations = iterations
        self.approx_mode = approx_mode
        self.operation = operation
        self.fill_const_value = fill_const_value

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_eltwise_unary_sfpu.h",
            "sfpu_operations_quasar.h",
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
        op = f"SfpuType::{self.operation.cpp_enum_value}"
        en_32bit_dest = config.dest_acc.cpp_enum_value
        approx_mode = self.approx_mode.cpp_enum_value
        return (
            f"// Operation {stage}: Unary {self.operation.cpp_enum_value} SFPU\n"
            f"_llk_math_eltwise_sfpu_init_();\n"
            f"test_utils::init_unary_sfpu_operation_quasar<{op}, {en_32bit_dest}, {approx_mode}>();\n"
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: SfpuNode,
        block: BlockData,
    ) -> str:
        op = f"SfpuType::{self.operation.cpp_enum_value}"
        dest_sync = operation.dest_sync.cpp_enum_value
        en_32bit_dest = config.dest_acc.cpp_enum_value
        sfpu_format = config.sentinel._math_format.cpp_enum_value
        approx_mode = self.approx_mode.cpp_enum_value
        quasar_iterations = self.iterations // 4
        return (
            f"test_utils::call_unary_sfpu_operation_quasar<"
            f"{op}, {dest_sync}, {en_32bit_dest}, {approx_mode}, {quasar_iterations}"
            f">({block.tile_id_block}, {sfpu_format});\n"
        )

    def __str__(self) -> str:
        return f"UnarySfpu({self.operation})"
