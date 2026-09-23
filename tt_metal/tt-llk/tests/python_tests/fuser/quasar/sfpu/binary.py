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
    DstRoundingMode,
    MathOperation,
)


class BinarySfpu(Sfpu):
    def __init__(
        self,
        operation: MathOperation,
        approx_mode: ApproximationMode = ApproximationMode.No,
        iterations: int = 8,
        dst_index_in0: int = 0,
        dst_index_in1: int = 1,
        dst_index_out: int = 0,
        dst_rounding_mode: DstRoundingMode = DstRoundingMode.Default,
    ):
        if not operation in MathOperation.get_sfpu_binary_operations():
            raise ValueError(
                f"Operation {operation} is not a valid SFPU binary operation."
            )
        self.operation = operation
        self.approx_mode = approx_mode
        self.iterations = iterations
        self.dst_index_in0 = dst_index_in0
        self.dst_index_in1 = dst_index_in1
        self.dst_index_out = dst_index_out
        self.dst_rounding_mode = dst_rounding_mode

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_eltwise_binary_sfpu.h",
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
        return self.binary_sfpu_golden(
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
        op = f"ckernel::BinaryOp::{self.operation.cpp_enum_value}"
        en_32bit_dest = config.dest_acc.cpp_enum_value
        approx_mode = self.approx_mode.cpp_enum_value

        return (
            f"    // Operation {stage}: Binary {self.operation.cpp_enum_value} SFPU\n"
            f"    _llk_math_eltwise_sfpu_init_();\n"
            f"    test_utils::init_binary_sfpu_operation_quasar<"
            f"{op}, {en_32bit_dest}, false, {approx_mode}>();\n"
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: SfpuNode,
        block: BlockData,
    ) -> str:
        op = f"ckernel::BinaryOp::{self.operation.cpp_enum_value}"
        dest_sync = operation.dest_sync.cpp_enum_value
        en_32bit_dest = config.dest_acc.cpp_enum_value
        approx_mode = self.approx_mode.cpp_enum_value
        quasar_iterations = self.iterations // 4
        src1 = self.dst_index_in0
        src2 = self.dst_index_in1
        dst = self.dst_index_out
        data_format = config.sentinel._math_format.cpp_enum_value
        dst_rounding_mode = self.dst_rounding_mode.cpp_enum_value

        return (
            f"test_utils::call_binary_sfpu_operation_quasar<"
            f"{op}, {dest_sync}, {en_32bit_dest}, {dst_rounding_mode}, "
            f"{quasar_iterations}, false, {approx_mode}"
            f">({src1} /* src0_tile */, {src2} /* src1_tile */, {dst} /* dst_tile */, {data_format});\n"
        )

    def __str__(self) -> str:
        return f"BinarySfpu({self.operation})"
