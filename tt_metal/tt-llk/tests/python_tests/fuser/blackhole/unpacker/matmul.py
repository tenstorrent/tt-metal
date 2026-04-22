# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.compute_node import ComputeNode
from fuser.fused_loop import FusedLoop, LoopBlock
from fuser.fused_operation import FusedOperation
from fuser.fused_unpacker import Unpacker
from fuser.fuser_config import GlobalConfig
from helpers.golden_generators import TransposeGolden, get_golden_generator
from helpers.llk_params import Transpose


class MatmulUnpacker(Unpacker):
    loop: FusedLoop = LoopBlock()

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_AB_matmul.h",
            "llk_unpack_common.h",
        ]

    def perf_set_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        num_cols = compute_unit.src_a.tile_shape.total_col_dim()
        kt_dim = compute_unit.src_a.dimensions[1] // num_cols
        rt_dim = block.block_tiles_y
        ct_dim = block.block_tiles_x
        return f"_perf_unpack_matmul_mock(1, {rt_dim}, {kt_dim}, {ct_dim});\n"

    def perf_clear_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        num_cols = compute_unit.src_a.tile_shape.total_col_dim()
        kt_dim = compute_unit.src_a.dimensions[1] // num_cols
        rt_dim = block.block_tiles_y
        ct_dim = block.block_tiles_x
        return f"_perf_math_matmul_mock(1, {rt_dim}, {kt_dim}, {ct_dim});\n"

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_matrix = get_golden_generator(TransposeGolden)

        if compute_unit.unpack_transpose_faces == Transpose.Yes:
            tensor_b = t_matrix.transpose_faces_multi_tile(
                tensor_b,
                compute_unit.src_b.data_format,
                compute_unit.src_b.tile_count,
                tilize=True,
                input_dimensions=compute_unit.src_b.dimensions,
            )

        if compute_unit.unpack_transpose_within_face == Transpose.Yes:
            tensor_b = t_matrix.transpose_within_faces_multi_tile(
                tensor_b,
                compute_unit.src_b.data_format,
                compute_unit.src_b.tile_count,
                untilize=True,
                input_dimensions=compute_unit.src_b.dimensions,
            )

        return tensor_a, tensor_b

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        face_r_dim = compute_unit.src_a.tile_shape.face_r_dim
        rt_dim = block.block_tiles_y
        ct_dim = block.block_tiles_x
        num_cols = compute_unit.src_a.tile_shape.total_col_dim()
        kt_dim = compute_unit.src_a.dimensions[1] // num_cols
        transpose_faces = compute_unit.unpack_transpose_faces.cpp_enum_value

        return f"_llk_unpack_AB_matmul_init_<>({transpose_faces}, {ct_dim}, {rt_dim}, {kt_dim}, {face_r_dim}, {face_r_dim});\n"

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        rt_dim = block.block_tiles_y
        ct_dim = block.block_tiles_x
        num_cols = compute_unit.src_a.tile_shape.total_col_dim()
        kt_dim = compute_unit.src_a.dimensions[1] // num_cols
        unpack_tile_size_a = compute_unit.src_a.tile_size
        unpack_tile_size_b = compute_unit.src_b.tile_size
        full_ct_dim = compute_unit.src_b.dimensions[1] // 32
        output_ct_dim = compute_unit.src_a.tile_count_x
        src_a_partial_face = compute_unit.src_a.partial_face.cpp_enum_value
        src_b_partial_face = compute_unit.src_b.partial_face.cpp_enum_value
        buffer_a = compute_unit.src_a.cpp_name
        buffer_b = compute_unit.src_b.cpp_name

        return (
            f"    {{\n"
            f"        std::uint32_t row = ({block.tile_id_global}) / {output_ct_dim};\n"
            f"        std::uint32_t col = ({block.tile_id_global}) % {output_ct_dim};\n"
            f"        for (std::uint32_t kt = 0; kt < {kt_dim}; ++kt) {{\n"
            f"            std::uint32_t srca_tile_idx = row * {kt_dim} + kt;\n"
            f"            std::uint32_t srcb_tile_idx = kt * {full_ct_dim} + col;\n"
            f"            _llk_unpack_AB_matmul_<>(\n"
            f"                L1_ADDRESS({buffer_a}[0]), L1_ADDRESS({buffer_b}[0]),\n"
            f"                srca_tile_idx, srcb_tile_idx,\n"
            f"                {unpack_tile_size_a}, {unpack_tile_size_b},\n"
            f"                {src_a_partial_face}, {src_b_partial_face},\n"
            f"                {ct_dim}, {rt_dim}, {kt_dim}\n"
            f"            );\n"
            f"        }}\n"
            f"    }}\n"
        )
