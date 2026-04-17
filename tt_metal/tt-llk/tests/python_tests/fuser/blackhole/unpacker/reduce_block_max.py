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


class ReduceBlockMaxUnpacker(Unpacker):
    loop: FusedLoop = LoopTileByTile()

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        dest_acc = config.dest_acc.cpp_enum_value
        return f"_llk_unpack_AB_reduce_block_max_row_init_<{ct_dim}, {dest_acc}>();\n"

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        ct_dim = block.block_tiles_x
        tile_x_abs = f"(({block.tile_id_global}) % {block.tile_count_x})"
        tile_x_in_block = f"({tile_x_abs} - {block.block_x})"
        return (
            f"if (({tile_x_in_block}) % {ct_dim} == 0 ) {{\n"
            f"_llk_unpack_AB_reduce_block_max_row_(L1_ADDRESS(buffer_A{stage}[{block.tile_id_global}]), L1_ADDRESS(buffer_B{stage}[{block.tile_id_global}]));\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        face_r_dim = operation.face_r_dim
        return f"_llk_unpack_AB_reduce_block_max_row_uninit_({face_r_dim}, {face_r_dim});\n"

    def perf_set_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        tile_x_abs = f"(({block.tile_id_global}) % {block.tile_count_x})"
        tile_x_in_block = f"({tile_x_abs} - {block.block_x})"
        return (
            f"if (({tile_x_in_block}) % {ct_dim} == 0) {{\n"
            f"    _perf_unpack_loop_set_valid<false, true>(1);\n"
            f"    _perf_unpack_loop_set_valid<true, false>({ct_dim});\n"
            f"}}\n"
        )

    def perf_clear_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        tile_x_in_block = f"(({block.tile_id_block}) % {block.block_tiles_x})"
        return (
            f"if (({tile_x_in_block}) % {ct_dim} == 0) {{\n"
            f"    _perf_math_loop_clear_valid<true, false>({ct_dim});\n"
            f"    _perf_math_loop_clear_valid<false, true>(1);\n"
            f"}}\n"
        )

    def get_headers(self) -> List[str]:
        return ["experimental/llk_unpack_AB_reduce_custom.h"]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor_a, tensor_b
