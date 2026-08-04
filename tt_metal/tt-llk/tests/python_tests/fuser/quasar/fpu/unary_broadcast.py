# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fused_fpu import Fpu
from fuser.fused_loop import FusedLoop, LoopTileByTile
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig
from helpers.golden_generators import BroadcastGolden, get_golden_generator
from helpers.tilize_untilize import tilize_block, untilize_block


class UnaryBroadcastFpu(Fpu):
    loop: FusedLoop = LoopTileByTile()

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_unary_broadcast.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tile_dims = (
            operation.tile_shape.total_row_dim(),
            operation.tile_shape.total_col_dim(),
        )
        num_faces = operation.tile_shape.total_num_faces()
        tilized_b = tilize_block(
            tensor_b,
            compute_unit.src_b.dimensions,
            compute_unit.src_b.data_format,
            num_faces=num_faces,
            tile_dimensions=tile_dims,
        )
        broadcast_golden = get_golden_generator(BroadcastGolden)
        broadcast_result = broadcast_golden(
            compute_unit.broadcast_type,
            tilized_b,
            compute_unit.src_b.data_format,
            num_faces,
            compute_unit.src_b.tile_count,
            operation.tile_shape.face_r_dim,
        )
        golden_tensor = untilize_block(
            broadcast_result,
            compute_unit.src_b.data_format,
            compute_unit.src_b.dimensions,
            tile_dimensions=tile_dims,
            num_faces=num_faces,
        )
        return tensor_a, tensor_b, golden_tensor

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        tensor_shape = operation.tile_shape.cpp_value
        return (
            f"// Operation {stage}: Unary Broadcast FPU\n"
            f"_llk_math_eltwise_unary_broadcast_init_<{broadcast_type}, false>"
            f"({tensor_shape});\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return f"_llk_math_eltwise_unary_broadcast_({block.tile_id_block});\n"

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
