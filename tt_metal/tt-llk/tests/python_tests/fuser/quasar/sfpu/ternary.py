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
            f"{config.dest_acc.cpp_enum_value} /*is_fp32_dest_acc_en*/, "
            f"{self.approx_mode.cpp_enum_value} /*APPROX*/>();\n"
        )

    def calculate(self, operation, config, compute_unit, block):
        op = f"SfpuType::{self.operation.cpp_enum_value}"
        tile_elements = operation.tile_shape.total_tile_size()
        dst_tile_shape = f"ckernel::trisc::DstTileShape::Tile32x{tile_elements // 32}"
        vector_mode = self._vector_mode(operation)
        return (
            f"test_utils::call_ternary_sfpu_operation_quasar<{op}, "
            f"{operation.dest_sync.cpp_enum_value}, "
            f"{config.dest_acc.cpp_enum_value} /*is_fp32_dest_acc_en*/, "
            f"{self.approx_mode.cpp_enum_value} /*APPROX*/, "
            f"{self.iterations} /*ITERATIONS*/, "
            f"{dst_tile_shape} /*TILE_SHAPE*/>("
            f"{block.dest_src0} /*src0_tile*/, "
            f"{block.dest_src1} /*src1_tile*/, "
            f"{block.dest_src2} /*src2_tile*/, "
            f"{block.tile_id_dest} /*dst_tile*/, "
            f"{vector_mode} /*vector_mode*/);\n"
        )
