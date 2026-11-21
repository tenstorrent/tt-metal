# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.base_sfpu import Sfpu
from fuser.golden.sfpu.ternary import ternary_golden
from fuser.indexing import InvocationGranularity
from helpers.format_config import DataFormat
from helpers.llk_params import ApproximationMode, MathOperation


class TernarySfpu(Sfpu):
    granularity = InvocationGranularity.TILE
    input_count = 3
    golden_fn = staticmethod(ternary_golden)
    # _calculate_where_ static_asserts on these Dst formats.
    _WHERE_FORMATS = {
        DataFormat.Float32,
        DataFormat.Float16_b,
        DataFormat.Int32,
        DataFormat.UInt32,
    }

    def __init__(
        self,
        operation: MathOperation,
        approx_mode: ApproximationMode = ApproximationMode.No,
        iterations: int = 8,
    ):
        if operation not in MathOperation.get_sfpu_ternary_operations():
            raise ValueError(
                f"Operation {operation} is not a valid SFPU ternary operation."
            )
        self.operation = operation
        self.approx_mode = approx_mode
        self.iterations = iterations

    def supports_dest_format(self, dest_format) -> bool:
        return (
            self.operation != MathOperation.SfpuWhere
            or dest_format in self._WHERE_FORMATS
        )

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
        vector_mode = self._vector_mode(operation)
        data_format = compute_unit.dest_format.cpp_enum_value
        return (
            "test_utils::call_ternary_sfpu_operation<"
            f"{operation.dest_sync.cpp_enum_value}, {config.dest_acc.cpp_enum_value}, "
            f"{op}, {self.approx_mode.cpp_enum_value}, "
            f"{config.dest_acc.cpp_enum_value}, {data_format}, {self.iterations}>("
            f"{block.dest_src0} /*dst_index_in0*/, "
            f"{block.dest_src1} /*dst_index_in1*/, "
            f"{block.dest_src2} /*dst_index_in2*/, "
            f"{block.tile_id_dest} /*dst_index_out*/, 0 /*value*/, {vector_mode});\n"
        )

    def __str__(self):
        return f"TernarySfpu({self.operation})"
