# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.base_fpu import Fpu
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from fuser.quasar.unpacker.unpack_a import (
    _uses_upk_to_dest_semaphores,
    upk_to_dest_math_ack,
)
from fuser.tile_loop import LoopBlockRow, TileLoop


class DatacopyFpu(Fpu):
    loop: TileLoop = LoopBlockRow()
    per_block_init = True

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_eltwise_unary_datacopy.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.datacopy_golden(
            tensor_a, tensor_b, tensor_dst, config, operation, compute_unit
        )

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.unpack_to_dest.value:
            return ""

        stage = operation.stage_id
        data_copy_type = compute_unit.data_copy_type.cpp_enum_value
        num_faces = operation.tile_shape.total_num_faces()
        face_r_dim = operation.tile_shape.face_r_dim
        num_rows_per_matrix = face_r_dim * num_faces
        en_32bit_dest = config.dest_acc.cpp_enum_value

        return (
            f"// Operation {stage}: Datacopy FPU\n"
            f"_llk_math_eltwise_unary_datacopy_init_<{data_copy_type}, {en_32bit_dest}>"
            f"({num_rows_per_matrix}, {block.block_tiles_x});\n"
        )

    def calculate(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.unpack_to_dest.value:
            if not _uses_upk_to_dest_semaphores(config):
                return ""
            return upk_to_dest_math_ack()

        return f"_llk_math_eltwise_unary_datacopy_({block.tile_id_block});\n"

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
