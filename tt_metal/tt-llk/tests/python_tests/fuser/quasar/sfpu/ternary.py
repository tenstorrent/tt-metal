# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.blackhole.sfpu.ternary import TernarySfpu as BaseTernarySfpu


class TernarySfpu(BaseTernarySfpu):
    def get_headers(self):
        return [
            "llk_math_common.h",
            "sfpu_operations_quasar.h",
        ]

    def init(self, operation, config, compute_unit, block):
        op = f"SfpuType::{self.operation.cpp_enum_value}"
        return (
            f"test_utils::init_ternary_sfpu_operation_quasar<{op}, "
            f"{config.dest_acc.cpp_enum_value}, {self.approx_mode.cpp_enum_value}>();\n"
        )

    def calculate(self, operation, config, compute_unit, block):
        op = f"SfpuType::{self.operation.cpp_enum_value}"
        return (
            f"test_utils::call_ternary_sfpu_operation_quasar<{op}, "
            f"{operation.dest_sync.cpp_enum_value}, {config.dest_acc.cpp_enum_value}, "
            f"{self.approx_mode.cpp_enum_value}, {self.iterations}>("
            f"{block.dest_src0}, {block.dest_src1}, {block.dest_src2}, "
            f"{block.tile_id_dest});\n"
        )
