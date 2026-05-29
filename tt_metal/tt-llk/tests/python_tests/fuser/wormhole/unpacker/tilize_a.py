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
from helpers.tilize_untilize import tilize_block


class UnpackerTilizeA(Unpacker):
    loop: FusedLoop = LoopTileByTile()

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_tilize.h",
        ]

    def perf_set_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        valid_cnt = 4
        return f"_perf_unpack_loop_set_valid<true, true>({valid_cnt});\n"

    def perf_clear_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        valid_cnt = 4
        return f"_perf_math_loop_clear_valid<true, true>({valid_cnt});\n"

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tilized_a = tilize_block(
            tensor_a,
            compute_unit.src_a.dimensions,
            compute_unit.src_a.data_format,
            compute_unit.src_a.tile_shape.total_num_faces(),
        )

        return tilized_a, None

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        face_r_dim = compute_unit.src_a.tile_shape.face_r_dim
        block_ct_dim = compute_unit.src_a.tile_count_x

        return f"_llk_unpack_tilize_init_({config.sentinel.unpack_a_src_format}, {config.sentinel.unpack_a_dst_format}, {block_ct_dim}, {face_r_dim}, false);\n"

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        face_r_dim = compute_unit.src_a.tile_shape.face_r_dim
        num_faces = compute_unit.src_a.tile_shape.total_num_faces()
        block_ct_dim = compute_unit.src_a.tile_count_x
        buffer_a = compute_unit.src_a.cpp_name

        return (
            f"{{\n"
            f"std::uint32_t row = ({block.tile_id_global}) / {block_ct_dim};\n"
            f"std::uint32_t col = ({block.tile_id_global}) % {block_ct_dim};\n"
            f"_llk_unpack_tilize_(L1_ADDRESS({buffer_a}[row * {block_ct_dim}]), col, {config.sentinel.unpack_a_src_format}, {config.sentinel.unpack_a_dst_format}, {block_ct_dim}, {face_r_dim}, {num_faces}, false);\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        face_r_dim = compute_unit.src_a.tile_shape.face_r_dim

        return f"_llk_unpack_tilize_uninit_({config.sentinel.unpack_a_dst_format}, {face_r_dim});\n\n"
