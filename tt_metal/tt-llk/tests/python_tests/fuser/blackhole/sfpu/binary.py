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
    BinarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
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

    def get_headers(self) -> List[str]:
        return [
            "ckernel_defs.h",
            "ckernel_sfpu.h",
            "ckernel_sfpu_binary.h",
            "llk_math_common.h",
            "llk_math_eltwise_binary_sfpu.h",
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
        math_format = operation.output.data_format

        generate_binary_golden = get_golden_generator(BinarySFPUGolden)
        golden_tensor = generate_binary_golden(
            self.operation,
            tensor,
            self.dst_index_in0,
            self.dst_index_in1,
            self.dst_index_out,
            self.iterations,
            batch_dims,
            math_format,
            skip_tilize=True,
        )

        return golden_tensor

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id

        return (
            f"    // Operation {stage}: Binary {self.operation.cpp_enum_value} SFPU\n"
            f"    _llk_math_eltwise_binary_sfpu_init_<SfpuType::add1>();\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        op = f"ckernel::BinaryOp::{self.operation.cpp_enum_value}"
        approx_mode = self.approx_mode.cpp_enum_value
        iterations = self.iterations
        src1 = self.dst_index_in0
        src2 = self.dst_index_in1
        dst = self.dst_index_out

        if self.operation == MathOperation.SfpuAddTopRow:
            format = "0"
        else:
            format = f"math_format{stage}"

        return (
            f"    _llk_math_eltwise_binary_sfpu_start_<dest_sync{stage}>(0);\n"
            f"    test_utils::call_binary_sfpu_operation<{approx_mode}, {op}, {iterations}, {format}>({src1}, {src2}, {dst});\n"
            f"    _llk_math_eltwise_binary_sfpu_done_();\n"
        )

    def __str__(self) -> str:
        return f"BinarySfpu({self.operation})"
