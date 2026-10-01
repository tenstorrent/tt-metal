# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.base_sfpu import Sfpu
from fuser.golden.sfpu.ternary import ternary_golden
from fuser.indexing import InvocationGranularity
from helpers.llk_params import ApproximationMode, MathOperation


class TernarySfpu(Sfpu):
    granularity = InvocationGranularity.TILE
    input_count = 3
    golden_fn = staticmethod(ternary_golden)

    def __init__(
        self,
        operation: MathOperation,
        approx_mode: ApproximationMode = ApproximationMode.No,
        iterations: int = 8,
    ):
        self.operation = operation
        self.approx_mode = approx_mode
        self.iterations = iterations

    def get_headers(self):
        return ["llk_math_common.h", "sfpu_operations.h"]

    def init(self, operation, config, compute_unit, block):
        op = f"SfpuType::{self.operation.cpp_enum_value}"
        return (
            "test_utils::call_ternary_sfpu_operation_init<"
            f"{op}, {self.approx_mode.cpp_enum_value}, "
            f"{config.dest_acc.cpp_enum_value}>();\n"
        )

    def calculate(self, operation, config, compute_unit, block):
        op = f"SfpuType::{self.operation.cpp_enum_value}"
        vector_mode = (
            "ckernel::VectorMode::R"
            if operation.tile_shape.tile_dims in ((16, 32), (32, 16))
            else "ckernel::VectorMode::RC"
        )
        data_format = config.sentinel._sfpu_format.cpp_enum_value
        return (
            "test_utils::call_ternary_sfpu_operation<"
            f"{operation.dest_sync.cpp_enum_value}, {config.dest_acc.cpp_enum_value}, "
            f"{op}, {self.approx_mode.cpp_enum_value}, "
            f"{config.dest_acc.cpp_enum_value}, {data_format}, {self.iterations}>("
            f"{block.dest_src0}, {block.dest_src1}, {block.dest_src2}, "
            f"{block.tile_id_dest}, 0, {vector_mode});\n"
        )

    def __str__(self):
        return f"TernarySfpu({self.operation})"
