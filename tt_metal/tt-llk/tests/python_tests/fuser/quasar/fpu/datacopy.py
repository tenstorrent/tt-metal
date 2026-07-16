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
from helpers.golden_generators import DataCopyGolden, get_golden_generator


class DatacopyFpu(Fpu):
    loop: FusedLoop = LoopTileByTile()

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
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_tensor = tensor_a

        golden_generator = get_golden_generator(DataCopyGolden)
        golden_tensor = golden_generator(
            source_tensor,
            config.sentinel.golden_math_format,
            num_faces=operation.tile_shape.total_num_faces(),
            input_dimensions=compute_unit.src_a.dimensions,
            face_r_dim=operation.tile_shape.face_r_dim,
            tile_shape=operation.tile_shape,
        )

        return (tensor_a, tensor_b, golden_tensor)

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        data_copy_type = compute_unit.data_copy_type.cpp_enum_value
        num_faces = operation.tile_shape.total_num_faces()
        face_r_dim = operation.tile_shape.face_r_dim
        num_rows_per_matrix = face_r_dim * num_faces
        en_32bit_dest = config.dest_acc.cpp_enum_value

        return (
            f"// Operation {stage}: Datacopy FPU\n"
            f"_llk_math_eltwise_unary_datacopy_init_<{data_copy_type}, {en_32bit_dest}>"
            f"({num_rows_per_matrix}, 1);\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return f"_llk_math_eltwise_unary_datacopy_({block.tile_id_block});\n"

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
