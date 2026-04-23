# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.compute_node import ComputeNode
from fuser.fused_loop import FusedLoop, LoopTileByTile
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


class UnpackerA(Unpacker):
    loop: FusedLoop = LoopTileByTile()

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_A.h",
            "llk_unpack_common.h",
            "llk_unpack_tilize.h",
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
            tensor_b = tensor_a
            tensor_a = None
            tensor_b = tilize_block(
                tensor_b, operation.src_a.dimensions, operation.src_a.data_format
            )
            broadcast_golden = get_golden_generator(BroadcastGolden)
            tensor_b = broadcast_golden(
                compute_unit.broadcast_type,
                tensor_b,
                operation.src_a.data_format,
                operation.num_faces,
                operation.src_a.tile_count,
                operation.face_r_dim,
            )
            tensor_b = untilize_block(
                tensor_b,
                operation.src_a.data_format,
                operation.src_a.dimensions,
            )
        else:
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
            tensor_b = None

        return tensor_a, tensor_b

    def perf_set_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        if compute_unit.broadcast_type == BroadcastType.Scalar:
            return "_perf_unpack_loop_set_valid<false, true>(1);\n"
        elif compute_unit.broadcast_type == BroadcastType.Column:
            return (
                "_perf_unpack_loop_set_valid<false, true>(2);\n"
                "_perf_unpack_loop_set_valid<true, false>(1);\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Row:
            return "_perf_unpack_loop_set_valid<false, true>(4);\n"
        else:
            num_faces = operation.num_faces
            return f"_perf_unpack_loop_set_valid<true, true>({num_faces});\n"

    def perf_clear_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        if compute_unit.broadcast_type == BroadcastType.Scalar:
            return "_perf_math_loop_clear_valid<false, true>(1);\n"
        elif compute_unit.broadcast_type == BroadcastType.Column:
            return (
                "_perf_math_loop_clear_valid<false, true>(2);\n"
                "_perf_math_loop_clear_valid<true, false>(1);\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Row:
            return "_perf_math_loop_clear_valid<false, true>(4);\n"
        else:
            num_faces = operation.num_faces
            return f"_perf_math_loop_clear_valid<true, true>({num_faces});\n"

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        unpack_to_dest = "true" if operation.unpack_to_dest else "false"
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        face_r_dim = operation.face_r_dim
        num_faces = operation.num_faces
        transpose_faces = compute_unit.unpack_transpose_faces.cpp_enum_value
        transpose_within_face = compute_unit.unpack_transpose_within_face.cpp_enum_value

        return (
            f"    _llk_unpack_A_init_<{broadcast_type}, false, {reuse_dest}, {unpack_to_dest}>(\n"
            f"        {transpose_faces}, {transpose_within_face}, {face_r_dim}, {num_faces}, unpack_a_src_format{stage}, unpack_a_dst_format{stage}\n"
            f"    );\n"
        )

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        unpack_to_dest = "true" if operation.unpack_to_dest else "false"
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value

        return (
            f"_llk_unpack_A_<{broadcast_type}, false, {reuse_dest}, {unpack_to_dest}>(\n"
            f"    L1_ADDRESS(buffer_A{stage}[{block.tile_id_global}]), unpack_a_src_format{stage}, unpack_a_dst_format{stage}\n"
            f");\n"
        )
