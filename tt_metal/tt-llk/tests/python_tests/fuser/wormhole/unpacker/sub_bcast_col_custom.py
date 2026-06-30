# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fused_loop import FusedLoop, LoopBlockRow
from fuser.fused_math import ComputeNode
from fuser.fused_operation import FusedOperation
from fuser.fused_unpacker import Unpacker
from fuser.fuser_config import GlobalConfig
from helpers.golden_generators import BroadcastGolden, get_golden_generator
from helpers.llk_params import BroadcastType
from helpers.tilize_untilize import tilize_block, untilize_block


class SubBcastColCustomUnpacker(Unpacker):
    loop: FusedLoop = LoopBlockRow()
    per_block_init = True

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "experimental/llk_unpack_AB_sub_bcast_col_custom.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tile_count_x = compute_unit.src_b.tile_count_x
        tile_count_y = compute_unit.src_b.tile_count_y
        block_tiles_x = operation.block_tiles_x
        num_faces = compute_unit.src_a.tile_shape.total_num_faces()
        face_r_dim = compute_unit.src_a.tile_shape.face_r_dim

        src_b_tile_dims = (
            compute_unit.src_b.tile_shape.total_row_dim(),
            compute_unit.src_b.tile_shape.total_col_dim(),
        )
        src_b_num_faces = compute_unit.src_b.tile_shape.total_num_faces()
        tilized_b = tilize_block(
            tensor_b,
            compute_unit.src_b.dimensions,
            compute_unit.src_b.data_format,
            num_faces=src_b_num_faces,
            tile_dimensions=src_b_tile_dims,
        )
        broadcast_golden = get_golden_generator(BroadcastGolden)

        for ty in range(tile_count_y):
            for bx in range(0, tile_count_x, block_tiles_x):
                first_tile_idx = ty * tile_count_x + bx
                ct_dim = min(block_tiles_x, tile_count_x - bx)
                broadcast_tile = broadcast_golden(
                    BroadcastType.Column,
                    tilized_b[first_tile_idx],
                    compute_unit.src_b.data_format,
                    num_faces,
                    1,
                    face_r_dim,
                )
                for tx_offset in range(ct_dim):
                    tilized_b[first_tile_idx + tx_offset] = broadcast_tile

        tensor_b = untilize_block(
            tilized_b,
            compute_unit.src_b.data_format,
            compute_unit.src_b.dimensions,
            tile_dimensions=src_b_tile_dims,
            num_faces=src_b_num_faces,
        )

        return tensor_a.flatten(), tensor_b.flatten()

    def perf_set_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        return (
            f"_perf_unpack_loop_set_valid<false, true>(1);\n"
            f"_perf_unpack_loop_set_valid<true, false>({ct_dim});\n"
        )

    def perf_clear_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        return (
            f"_perf_math_loop_clear_valid<true, false>({ct_dim});\n"
            f"_perf_math_loop_clear_valid<false, true>(1);\n"
        )

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        return "_llk_unpack_AB_sub_bcast_col_init_custom_();\n"

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        buffer_a = compute_unit.src_a.cpp_name
        buffer_b = compute_unit.src_b.cpp_name
        return (
            f"_llk_unpack_AB_sub_bcast_col_custom_("
            f"L1_ADDRESS({buffer_a}[{block.tile_id_global}]), "
            f"L1_ADDRESS({buffer_b}[{block.tile_id_global}]), "
            f"{ct_dim});\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        return "_llk_unpack_AB_sub_bcast_col_uninit_custom_();\n"
