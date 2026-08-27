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


class UnpackReduceTilize(Unpacker):
    loop: TileLoop = LoopTileByTile()

    def __init__(self, reduce_dim, reduce_pool):
        self.reduce_dim = reduce_dim
        self.reduce_pool = reduce_pool

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_reduce_col_tilizeA_strided.h",
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
            tensor_b,
        )

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        num_faces = compute_unit.src_a.tile_shape.total_num_faces()
        return (
            f"_perf_unpack_loop_set_valid<false, true>(1);\n"
            f"_perf_unpack_loop_set_valid<true, false>({num_faces});\n"
        )

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        num_faces = compute_unit.src_a.tile_shape.total_num_faces()
        return (
            f"_perf_math_loop_clear_valid<true, false>({num_faces});\n"
            f"_perf_math_loop_clear_valid<false, true>(1);\n"
        )

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        buf_desc_id_a = compute_unit.src_a.buf_desc_id
        buf_desc_id_b = compute_unit.src_b.buf_desc_id
        desc_a = compute_unit.src_a.cpp_desc_name
        full_ct_dim = compute_unit.src_a.tile_count_x
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        reduce_pool = self.reduce_pool.cpp_enum_value

        return (
            f"{desc_a}.f.y_dim = 1;\n"
            f"{desc_a}.f.z_dim = 1;\n"
            f"ckernel::trisc::_configure_buf_desc_table_({buf_desc_id_a}, {desc_a});\n"
            f"_llk_unpack_reduce_col_tilizeA_strided_init_<{reduce_pool}>"
            f"({buf_desc_id_a}, {buf_desc_id_b}, {full_ct_dim}, {tensor_shape});\n"
        )

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        row_stride = (
            compute_unit.src_a.tile_count_x
            * compute_unit.src_a.tile_shape.total_row_dim()
        )
        l1_row_idx = (
            f"{row_stride} * ({block.block_y} + tile_y) + ({block.block_x} + tile_x)"
        )

        return (
            f"_llk_unpack_reduce_col_tilizeA_strided_"
            f"({tensor_shape}, {l1_row_idx}, {block.tile_id_global});\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
