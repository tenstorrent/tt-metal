# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fused_fpu import Fpu
from fuser.fused_loop import FusedLoop, LoopTileByTile
from fuser.fused_math import ComputeNode
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import BroadcastType


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
        compute_unit: ComputeNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if compute_unit.broadcast_type != BroadcastType.None_:
            source_tensor = tensor_b
        else:
            source_tensor = tensor_a

        golden_generator = get_golden_generator(DataCopyGolden)
        golden_tensor = golden_generator(
            source_tensor,
            operation.output.data_format,
            num_faces=operation.num_faces,
            input_dimensions=operation.src_a.dimensions,
            face_r_dim=operation.face_r_dim,
        )

        return (tensor_a, tensor_b, golden_tensor)

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.cpp_enum_value
        tilize_en = operation.bh_tilize.cpp_enum_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        data_copy_type = compute_unit.data_copy_type.cpp_enum_value
        num_faces = operation.num_faces
        is_int_fpu_en = dest_acc

        return (
            f"    // Operation {stage}: Datacopy FPU\n"
            f"    _llk_math_eltwise_unary_datacopy_init_<{data_copy_type}, {dest_acc}, {broadcast_type}, {tilize_en}, {is_int_fpu_en}>(\n"
            f"        {num_faces}, math_format{stage}\n"
            f"    );\n"
        )

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.cpp_enum_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        unpack_to_dest = "true" if operation.unpack_to_dest else "false"
        data_copy_type = f"DataCopyType::{compute_unit.data_copy_type.name}"
        num_faces = operation.num_faces

        return (
            f"    _llk_math_eltwise_unary_datacopy_<{data_copy_type}, dest_sync{stage}, {dest_acc}, {broadcast_type}, {unpack_to_dest}>(\n"
            f"        {block.tile_id_block}, math_format{stage}, math_format{stage}, {num_faces}\n"
            f"    );\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        return f"_llk_math_eltwise_unary_datacopy_uninit_<{broadcast_type}, false>();\n"
