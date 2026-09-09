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
from fuser.tile_loop import LoopTileByTile, TileLoop


class ReduceBlockMaxUnpacker(Unpacker):
    loop: TileLoop = LoopTileByTile()

    per_block_init = True

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        dest_acc = config.dest_acc.cpp_enum_value
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        return f"_llk_unpack_AB_reduce_block_max_row_init_<{ct_dim}, {dest_acc}, /*respect_trigger=*/false>({tensor_shape});\n"

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        tile_x_abs = f"(({block.tile_id_global}) % {block.tile_count_x})"
        tile_x_in_block = f"({tile_x_abs} - {block.block_x})"
        buffer_a = compute_unit.src_a.cpp_name
        buffer_b = compute_unit.src_b.cpp_name
        return (
            f"if (({tile_x_in_block}) % {ct_dim} == 0 ) {{\n"
            f"_llk_unpack_AB_reduce_block_max_row_(L1_ADDRESS({buffer_a}[{block.tile_id_global}]), L1_ADDRESS({buffer_b}[{block.tile_id_global}]));\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return f"_llk_unpack_AB_reduce_block_max_row_uninit_();\n"

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
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
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
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
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor_a, tensor_b
