# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fused_loop import FusedLoop, LoopTileByTile
from fuser.fused_operation import FusedOperation
from fuser.fused_unpacker import Unpacker
from fuser.fuser_config import GlobalConfig
from helpers.tilize_untilize import tilize_block


class UnpackerTilizeA(Unpacker):
    loop: FusedLoop = LoopTileByTile()

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_tilize.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tilized_a = tilize_block(
            tensor_a,
            compute_unit.src_a.dimensions,
            compute_unit.src_a.data_format,
            compute_unit.src_a.tile_shape.total_num_faces(),
            tile_dimensions=(
                compute_unit.src_a.tile_shape.total_row_dim(),
                compute_unit.src_a.tile_shape.total_col_dim(),
            ),
        )

        return tilized_a, None

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        full_ct_dim = compute_unit.src_a.tile_count_x
        buf_desc_id = compute_unit.src_a.buf_desc_id
        face_r_dim = compute_unit.src_a.tile_shape.face_r_dim
        face_c_dim = compute_unit.src_a.tile_shape.face_c_dim
        num_faces_r = compute_unit.src_a.tile_shape.num_faces_r_dim
        num_faces_c = compute_unit.src_a.tile_shape.num_faces_c_dim
        en_32bit_dest = "true" if config.dest_acc.value else "false"
        block_ct_dim = 1

        return (
            f"_llk_unpack_tilize_init_<p_unpacr::UNP_A, {en_32bit_dest}>"
            f"({buf_desc_id}, {full_ct_dim}, {block_ct_dim}, "
            f"ckernel::TensorShape{{{face_r_dim}, {face_c_dim}, {num_faces_r}, {num_faces_c}}});\n"
        )

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        full_ct_dim = compute_unit.src_a.tile_count_x
        face_r_dim = compute_unit.src_a.tile_shape.face_r_dim
        num_faces_r = compute_unit.src_a.tile_shape.num_faces_r_dim
        num_faces_c = compute_unit.src_a.tile_shape.num_faces_c_dim
        y_stride = full_ct_dim * num_faces_r * face_r_dim

        return (
            f"{{\n"
            f"std::uint32_t row = ({block.tile_id_global}) / {full_ct_dim};\n"
            f"std::uint32_t col = ({block.tile_id_global}) % {full_ct_dim};\n"
            f"_llk_unpack_tilize_<p_unpacr::UNP_A>(row * {y_stride} + col * {num_faces_c});\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
