# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from fuser.tile_loop import LoopBlockRow, TileLoop


class UnpackerTilizeA(Unpacker):
    loop: TileLoop = LoopBlockRow()
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
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.tilize_golden(tensor_a, config, operation, compute_unit),
            None,
        )

    def init(
        self,
        operation: L1Operation,
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
        operation: L1Operation,
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
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
