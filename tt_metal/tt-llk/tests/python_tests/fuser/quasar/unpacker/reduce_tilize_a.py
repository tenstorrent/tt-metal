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


class UnpackReduceTilize(Unpacker):
    loop: FusedLoop = LoopTileByTile()

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_reduce_col_tilizeA_strided.h",
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

        return tilized_a, tensor_b

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        buf_desc_id_a = compute_unit.src_a.buf_desc_id
        buf_desc_id_b = compute_unit.src_b.buf_desc_id
        desc_a = compute_unit.src_a.cpp_desc_name
        full_ct_dim = compute_unit.src_a.tile_count_x
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value

        return (
            f"{desc_a}.buf_desc.f.y_dim = 1;\n"
            f"{desc_a}.buf_desc.f.z_dim = 1;\n"
            f"ckernel::trisc::_configure_buf_desc_table_({buf_desc_id_a}, {desc_a}.buf_desc);\n"
            f"_llk_unpack_reduce_col_tilizeA_strided_init_"
            f"({buf_desc_id_a}, {buf_desc_id_b}, {full_ct_dim}, {tensor_shape});\n"
        )

    def unpack(
        self,
        operation: FusedOperation,
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
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
