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
            "llk_math_common.h",
            "llk_math_eltwise_binary_sfpu.h",
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
        math_format = config.sentinel.golden_math_format

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
        compute_unit: SfpuNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        op = f"ckernel::BinaryOp::{self.operation.cpp_enum_value}"

        return (
            f"    // Operation {stage}: Binary {self.operation.cpp_enum_value} SFPU\n"
            f"    _llk_math_eltwise_sfpu_init_();\n"
            f"    test_utils::init_binary_sfpu_operation_quasar<{op}>();\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: SfpuNode,
        block: BlockData,
    ) -> str:
        op = f"ckernel::BinaryOp::{self.operation.cpp_enum_value}"
        en_32bit_dest = config.dest_acc.cpp_enum_value
        src1 = self.dst_index_in0
        src2 = self.dst_index_in1
        dst = self.dst_index_out

        return (
            f"    test_utils::call_binary_sfpu_operation_quasar<"
            f"{op}, {en_32bit_dest}, {self.iterations}"
            f">({self.dst_index_in0} /* base_dst_index */, "
            f"{src1} /* src0_tile */, {src2} /* src1_tile */, {dst} /* dst_tile */, "
            f"{config.sentinel.math_format});\n"
        )

    def __str__(self) -> str:
        return f"BinarySfpu({self.operation})"
