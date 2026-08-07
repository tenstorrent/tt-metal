# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fused_loop import FusedLoop, LoopBlockRow
from fuser.fused_operation import FusedOperation
from fuser.fused_unpacker import Unpacker
from fuser.fuser_config import GlobalConfig
from helpers.tilize_untilize import tilize_block


class UnpackerTilizeA(Unpacker):
    loop: FusedLoop = LoopBlockRow()
    per_block_init = True

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
            tile_dimensions=[
                compute_unit.src_a.tile_shape.total_row_dim(),
                compute_unit.src_a.tile_shape.total_col_dim(),
            ],
            face_r_dim=compute_unit.src_a.tile_shape.face_r_dim,
        )

        return tilized_a, None

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        buf_desc_id = compute_unit.src_a.buf_desc_id
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        en_32bit_dest = config.dest_acc.cpp_enum_value
        full_ct_dim = compute_unit.src_a.tile_count_x
        block_ct_dim = block.block_tiles_x

        return (
            f"_llk_unpack_tilize_init_<p_unpacr::UNP_A, {en_32bit_dest}>"
            f"({buf_desc_id}, {full_ct_dim}, {block_ct_dim}, {tensor_shape});\n"
        )

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        num_faces_r_dim = compute_unit.src_a.tile_shape.num_faces_r_dim
        face_r_dim = compute_unit.src_a.tile_shape.face_r_dim

        return (
            f"_llk_unpack_tilize_<p_unpacr::UNP_A>"
            f"({block.tile_id_global} * {num_faces_r_dim} * {face_r_dim});\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
