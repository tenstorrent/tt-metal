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
from helpers.llk_params import BroadcastType, EltwiseBinaryReuseDestType, Transpose
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
                tensor_b, compute_unit.src_a.dimensions, compute_unit.src_a.data_format
            )
            broadcast_golden = get_golden_generator(BroadcastGolden)
            tensor_b = broadcast_golden(
                compute_unit.broadcast_type,
                tensor_b,
                compute_unit.src_a.data_format,
                compute_unit.src_a.tile_shape.total_num_faces(),
                compute_unit.src_a.tile_count,
                compute_unit.src_a.tile_shape.face_r_dim,
            )
            tensor_b = untilize_block(
                tensor_b,
                compute_unit.src_a.data_format,
                compute_unit.src_a.dimensions,
            )
        else:
            if compute_unit.unpack_transpose_faces == Transpose.Yes:
                tensor_a = t_matrix.transpose_faces_multi_tile(
                    tensor_a,
                    compute_unit.src_a.data_format,
                    compute_unit.src_a.tile_count,
                    tilize=True,
                    untilize=True,
                    input_dimensions=compute_unit.src_a.dimensions,
                )

            if compute_unit.unpack_transpose_within_face == Transpose.Yes:
                tensor_a = t_matrix.transpose_within_faces_multi_tile(
                    tensor_a,
                    compute_unit.src_a.data_format,
                    compute_unit.src_a.tile_count,
                    tilize=True,
                    untilize=True,
                    input_dimensions=compute_unit.src_a.dimensions,
                )
            tensor_b = None

        if compute_unit.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
            tensor_b = tensor_a
            tensor_a = None

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
            num_faces = compute_unit.src_a.tile_shape.total_num_faces()
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
            num_faces = compute_unit.src_a.tile_shape.total_num_faces()
            return f"_perf_math_loop_clear_valid<true, true>({num_faces});\n"

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        unpack_to_dest = compute_unit.unpack_to_dest.cpp_enum_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        face_r_dim = compute_unit.src_a.tile_shape.face_r_dim
        num_faces = compute_unit.src_a.tile_shape.total_num_faces()
        transpose_faces = compute_unit.unpack_transpose_faces.cpp_enum_value
        transpose_within_face = compute_unit.unpack_transpose_within_face.cpp_enum_value
        acc_to_dest = compute_unit.acc_to_dest.cpp_enum_value

        return (
            f"    _llk_unpack_A_init_<{broadcast_type}, {acc_to_dest}, {reuse_dest}, {unpack_to_dest}>(\n"
            f"        {transpose_faces}, {transpose_within_face}, {face_r_dim}, {num_faces}, {config.sentinel.unpack_a_src_format}, {config.sentinel.unpack_a_dst_format}\n"
            f"    );\n"
        )

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        unpack_to_dest = compute_unit.unpack_to_dest.cpp_enum_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        acc_to_dest = compute_unit.acc_to_dest.cpp_enum_value
        buffer_a = compute_unit.src_a.cpp_name

        return (
            f"_llk_unpack_A_<{broadcast_type}, {acc_to_dest}, {reuse_dest}, {unpack_to_dest}>(\n"
            f"    L1_ADDRESS({buffer_a}[{block.tile_id_global}]), {config.sentinel.unpack_a_src_format}, {config.sentinel.unpack_a_dst_format}\n"
            f");\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        face_r_dim = compute_unit.src_a.tile_shape.face_r_dim

        return f"_llk_unpack_A_uninit_<{broadcast_type}>({face_r_dim});\n"
