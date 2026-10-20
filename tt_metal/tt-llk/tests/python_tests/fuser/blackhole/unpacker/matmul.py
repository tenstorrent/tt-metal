# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.unpack.matmul import unpack_matmul_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation


class MatmulUnpacker(Unpacker):
    granularity = InvocationGranularity.BLOCK
    per_block_init = True
    reverse_operands = True

    golden_fn = staticmethod(unpack_matmul_golden)

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_AB_matmul.h",
            "llk_unpack_common.h",
        ]

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        num_cols = compute_unit.src_a.tile_shape.total_col_dim()
        kt_dim = compute_unit.src_a.dimensions[1] // num_cols
        rt_dim = block.block_rows
        ct_dim = block.block_cols
        return f"_perf_unpack_matmul_mock(1, {rt_dim}, {kt_dim}, {ct_dim});\n"

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        num_cols = compute_unit.src_a.tile_shape.total_col_dim()
        kt_dim = compute_unit.src_a.dimensions[1] // num_cols
        rt_dim = block.block_rows
        ct_dim = block.block_cols
        return f"_perf_math_matmul_mock(1, {rt_dim}, {kt_dim}, {ct_dim});\n"

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        in0_tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        in1_tensor_shape = compute_unit.src_b.tile_shape.cpp_value
        rt_dim = block.block_rows
        ct_dim = block.block_cols
        num_cols = compute_unit.src_a.tile_shape.total_col_dim()
        kt_dim = compute_unit.src_a.dimensions[1] // num_cols
        transpose = compute_unit.transpose_within_face.cpp_enum_value

        return (
            f"_llk_unpack_AB_matmul_init_<>({transpose}, {ct_dim}, {rt_dim}, {kt_dim}, "
            f"{in0_tensor_shape}, {in1_tensor_shape});\n"
        )

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        rt_dim = block.block_rows
        ct_dim = block.block_cols
        num_cols = compute_unit.src_a.tile_shape.total_col_dim()
        kt_dim = compute_unit.src_a.dimensions[1] // num_cols
        unpack_tile_size_a = compute_unit.src_a.tile_size
        unpack_tile_size_b = compute_unit.src_b.tile_size
        full_ct_dim = (
            compute_unit.src_b.dimensions[1]
            // compute_unit.src_b.tile_shape.total_col_dim()
        )
        in0_tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        in1_tensor_shape = compute_unit.src_b.tile_shape.cpp_value
        buffer_a = compute_unit.src_a.cpp_name
        buffer_b = compute_unit.src_b.cpp_name

        return (
            f"    {{\n"
            f"        for (std::uint32_t kt = 0; kt < {kt_dim}; ++kt) {{\n"
            f"            std::uint32_t srca_tile_idx = ({block.tile_id_src_a}) + kt;\n"
            f"            std::uint32_t srcb_tile_idx = ({block.tile_id_src_b}) + kt * {full_ct_dim};\n"
            f"            _llk_unpack_AB_matmul_<>(\n"
            f"                L1_ADDRESS({buffer_a}[0]), L1_ADDRESS({buffer_b}[0]),\n"
            f"                srca_tile_idx, srcb_tile_idx,\n"
            f"                {unpack_tile_size_a}, {unpack_tile_size_b},\n"
            f"                {in0_tensor_shape}, {in1_tensor_shape},\n"
            f"                {ct_dim}, {rt_dim}, {kt_dim}\n"
            f"            );\n"
            f"        }}\n"
            f"    }}\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return f"_llk_unpack_AB_matmul_uninit_();\n"
