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


class UnaryBroadcastUnpacker(Unpacker):
    loop: TileLoop = LoopTileByTile()

    def _srcb_dvalids_per_tile(self, compute_unit: FpuNode) -> int:
        if compute_unit.broadcast_type == BroadcastType.Scalar:
            return 1
        return compute_unit.src_a.tile_shape.total_num_faces()

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        count = self._srcb_dvalids_per_tile(compute_unit)
        return f"_perf_unpack_loop_set_valid<false, true>({count});\n"

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        count = self._srcb_dvalids_per_tile(compute_unit)
        return f"_perf_math_loop_clear_valid<false, true>({count});\n"

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_unary_broadcast_operands.h",
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
        buf_desc_id = compute_unit.src_b.buf_desc_id
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        return (
            f"_llk_unpack_unary_broadcast_operands_init_<p_unpacr::UNP_B, {broadcast_type}, false>"
            f"({buf_desc_id}, 1);\n"
        )

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return (
            f"_llk_unpack_unary_broadcast_operands_<p_unpacr::UNP_B, false>"
            f"({block.tile_id_global});\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
