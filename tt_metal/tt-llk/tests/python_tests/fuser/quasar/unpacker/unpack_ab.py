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
from helpers.llk_params import BroadcastType


class UnpackerAB(Unpacker):
    loop: TileLoop = LoopTileByTile()

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
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor_b = self.broadcast_golden(tensor_b, config, operation, compute_unit)

        return tensor_a.flatten(), tensor_b.flatten()

    def init(
        self,
        operation: L1Operation,
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
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.broadcast_type != BroadcastType.None_:
            tile_id_b = (
                block.tile_id_global
                if compute_unit.broadcast_tile is None
                else compute_unit.broadcast_tile
            )
            return f"_llk_unpack_binary_broadcast_operands_({block.tile_id_global}, {tile_id_b});\n"

        return f"_llk_unpack_binary_operands_({block.tile_id_global}, {block.tile_id_global});\n"

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
