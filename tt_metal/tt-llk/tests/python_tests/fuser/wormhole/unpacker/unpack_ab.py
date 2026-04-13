# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fused_loop import FusedLoop, LoopTileByTile
from fuser.fused_math import ComputeNode
from fuser.fused_operation import FusedOperation
from fuser.fused_unpacker import Unpacker
from fuser.fuser_config import GlobalConfig
from helpers.golden_generators import (
    BroadcastGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import BroadcastType, Transpose
from helpers.tilize_untilize import tilize_block, untilize_block


class UnpackerAB(Unpacker):
    loop: FusedLoop = LoopTileByTile()

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_AB.h",
            "llk_unpack_common.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_matrix = get_golden_generator(TransposeGolden)
        if compute_unit.broadcast_type != BroadcastType.None_:
            tilized_b = tilize_block(
                tensor_b, operation.src_b.dimensions, operation.src_b.data_format
            )
            broadcast_golden = get_golden_generator(BroadcastGolden)
            broadcast_result = broadcast_golden(
                compute_unit.broadcast_type,
                tilized_b,
                operation.src_b.data_format,
                operation.num_faces,
                operation.src_b.tile_count,
                operation.face_r_dim,
            )
            tensor_b = untilize_block(
                broadcast_result,
                operation.src_b.data_format,
                operation.src_b.dimensions,
            )

        if compute_unit.unpack_transpose_faces == Transpose.Yes:
            tensor_a = t_matrix.transpose_faces_multi_tile(
                tensor_a,
                operation.src_a.data_format,
                operation.src_a.tile_count,
                tilize=True,
                untilize=True,
                input_dimensions=operation.src_a.dimensions,
            )

        if compute_unit.unpack_transpose_within_face == Transpose.Yes:
            tensor_a = t_matrix.transpose_within_faces_multi_tile(
                tensor_a,
                operation.src_a.data_format,
                operation.src_a.tile_count,
                tilize=True,
                untilize=True,
                input_dimensions=operation.src_a.dimensions,
            )

        return tensor_a.flatten(), tensor_b.flatten()

    def perf_set_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        num_faces = operation.num_faces
        if compute_unit.broadcast_type == BroadcastType.Scalar:
            return (
                f"_perf_unpack_loop_set_valid<false, true>(1);\n"
                f"_perf_unpack_loop_set_valid<true, false>({num_faces});\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Column:
            return (
                f"_perf_unpack_loop_set_valid<false, true>(2);\n"
                f"_perf_unpack_loop_set_valid<true, false>({num_faces});\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Row:
            return f"_perf_unpack_loop_set_valid<true, true>({num_faces});\n"
        else:
            return f"_perf_unpack_loop_set_valid<true, true>({num_faces});\n"

    def perf_clear_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        num_faces = operation.num_faces
        if compute_unit.broadcast_type == BroadcastType.Scalar:
            return (
                f"_perf_math_loop_clear_valid<false, true>(1);\n"
                f"_perf_math_loop_clear_valid<true, false>({num_faces});\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Column:
            return (
                f"_perf_math_loop_clear_valid<false, true>(2);\n"
                f"_perf_math_loop_clear_valid<true, false>({num_faces});\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Row:
            return f"_perf_math_loop_clear_valid<true, true>({num_faces});\n"
        else:
            return f"_perf_math_loop_clear_valid<true, true>({num_faces});\n"

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        tile_shape = operation.src_a.tile_shape
        transpose_value = "1" if compute_unit.unpack_transpose_faces.value else "0"

        shape_var = f"tensor_shape_stage_{operation.stage_id}"
        return (
            f"const ckernel::TensorShape {shape_var} = "
            f"{{{tile_shape.face_r_dim}, {tile_shape.face_c_dim}, {tile_shape.num_faces_r_dim}, {tile_shape.num_faces_c_dim}}};\n"
            f"_llk_unpack_AB_init_<{broadcast_type}>({shape_var}, {transpose_value});\n"
        )

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        broadcast_type = f"BroadcastType::{compute_unit.broadcast_type.value}"
        return f"_llk_unpack_AB_<{broadcast_type}>(L1_ADDRESS(buffer_A{stage}[{block.tile_id_global}]), L1_ADDRESS(buffer_B{stage}[{block.tile_id_global}]));\n"
