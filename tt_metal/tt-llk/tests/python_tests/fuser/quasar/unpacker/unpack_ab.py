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
from helpers.golden_generators import BroadcastGolden, get_golden_generator
from helpers.llk_params import BroadcastType
from helpers.tilize_untilize import tilize_block, untilize_block


class UnpackerAB(Unpacker):
    loop: FusedLoop = LoopTileByTile()

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_binary_operands.h",
            "llk_unpack_binary_broadcast_operands.h",
            "llk_unpack_common.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if compute_unit.broadcast_type != BroadcastType.None_:
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
            broadcast_result = broadcast_golden(
                compute_unit.broadcast_type,
                tilized_b,
                compute_unit.src_b.data_format,
                compute_unit.src_a.tile_shape.total_num_faces(),
                compute_unit.src_b.tile_count,
                compute_unit.src_a.tile_shape.face_r_dim,
            )
            tensor_b = untilize_block(
                broadcast_result,
                compute_unit.src_b.data_format,
                compute_unit.src_b.dimensions,
                tile_dimensions=src_b_tile_dims,
                num_faces=src_b_num_faces,
            )

        return tensor_a.flatten(), tensor_b.flatten()

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        buf_desc_id_a = compute_unit.src_a.buf_desc_id
        buf_desc_id_b = compute_unit.src_b.buf_desc_id

        if compute_unit.broadcast_type != BroadcastType.None_:
            broadcast_type = compute_unit.broadcast_type.cpp_enum_value
            return (
                f"_llk_unpack_binary_broadcast_operands_init_<{broadcast_type}>"
                f"({buf_desc_id_a}, {buf_desc_id_b}, 1);\n"
            )

        return (
            f"_llk_unpack_binary_operands_init_({buf_desc_id_a}, {buf_desc_id_b}, 1);\n"
        )

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.broadcast_type != BroadcastType.None_:
            return f"_llk_unpack_binary_broadcast_operands_({block.tile_id_global}, {block.tile_id_global});\n"

        return f"_llk_unpack_binary_operands_({block.tile_id_global}, {block.tile_id_global});\n"

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
