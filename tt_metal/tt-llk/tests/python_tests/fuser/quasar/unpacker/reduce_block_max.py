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


class ReduceBlockMaxUnpacker(Unpacker):
    loop: FusedLoop = LoopTileByTile()

    per_block_init = True

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        dest_acc = config.dest_acc.cpp_enum_value
        buf_desc_id_a = compute_unit.src_a.buf_desc_id
        buf_desc_id_b = compute_unit.src_b.buf_desc_id
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        return (
            f"_llk_unpack_reduce_block_max_row_init_"
            f"<{ct_dim}, {block.tile_count_x}, {block.tile_count_y}, {dest_acc}>"
            f"({buf_desc_id_a}, {buf_desc_id_b}, {tensor_shape});\n"
        )

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        tile_x_abs = f"(({block.tile_id_global}) % {block.tile_count_x})"
        tile_x_in_block = f"({tile_x_abs} - {block.block_x})"
        dest_acc = config.dest_acc.cpp_enum_value
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        tiny_condition = f"({block.tile_id_global} == 0)"
        block_condition = f"(({tile_x_in_block}) % {ct_dim} == 0)"
        return (
            f"if ((({tensor_shape}).num_faces_r_dim == 1 && {tiny_condition}) || "
            f"(({tensor_shape}).num_faces_r_dim != 1 && {block_condition})) {{\n"
            f"_llk_unpack_reduce_block_max_row_"
            f"<{ct_dim}, {block.tile_count_x}, {block.tile_count_y}, {dest_acc}>"
            f"({block.tile_id_global}, {block.tile_id_global}, "
            f"{compute_unit.src_a.buf_desc_id}, {compute_unit.src_b.buf_desc_id}, "
            f"{tensor_shape});\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_unpack_reduce_block_max_row_uninit_();\n"

    def get_headers(self) -> List[str]:
        return ["experimental/llk_unpack_reduce_custom.h"]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor_a, tensor_b
