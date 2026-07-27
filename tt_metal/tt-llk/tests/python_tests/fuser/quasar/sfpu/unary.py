# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from fuser.block_data import BlockData
from fuser.fused_operation import FusedOperation
from fuser.fused_sfpu import Sfpu
from fuser.fuser_config import GlobalConfig
from fuser.sfpu_node import SfpuNode
from helpers.golden_generators import (
    UnarySFPUGolden,
    get_golden_generator,
)
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
            "llk_math_common.h",
            "llk_math_eltwise_unary_sfpu.h",
            "sfpu_operations_quasar.h",
        ]

    def golden(
        self,
        tensor: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: SfpuNode,
        batch_dims: tuple,
        batch_tile_cnt: int,
    ) -> torch.Tensor:
        format_input = config.sentinel.golden_math_format
        format_output = config.sentinel.golden_math_format
        dest_acc = config.dest_acc

        generate_sfpu_golden = get_golden_generator(UnarySFPUGolden)

        return generate_sfpu_golden(
            self.operation,
            tensor,
            format_output,
            dest_acc,
            format_input,
            batch_dims,
            self.iterations,
            self.dest_idx,
            self.fill_const_value,
            skip_tilize=True,
        )

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: SfpuNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        op = f"SfpuType::{self.operation.cpp_enum_value}"

        return (
            f"    // Operation {stage}: Unary {self.operation.cpp_enum_value} SFPU\n"
            f"    _llk_math_eltwise_sfpu_init_();\n"
            f"    test_utils::init_unary_sfpu_operation_quasar<{op}>();\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: SfpuNode,
        block: BlockData,
    ) -> str:
        op = f"SfpuType::{self.operation.cpp_enum_value}"
        en_32bit_dest = config.dest_acc.cpp_enum_value
        sfpu_format = config.sentinel._math_format.cpp_enum_value
        return (
            f"    test_utils::call_unary_sfpu_operation_quasar<"
            f"{op}, {en_32bit_dest}, {self.iterations}"
            f">({self.dest_idx}, {sfpu_format});\n"
        )

    def __str__(self) -> str:
        return f"UnarySfpu({self.operation})"
