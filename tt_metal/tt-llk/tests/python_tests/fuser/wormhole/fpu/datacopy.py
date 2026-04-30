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
from helpers.llk_params import BroadcastType, DataFormat


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
            num_faces=operation.output.tile_shape.total_num_faces(),
            input_dimensions=compute_unit.src_a.dimensions,
            face_r_dim=operation.output.tile_shape.face_r_dim,
        )

        return (tensor_a, tensor_b, golden_tensor)

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        dest_acc = config.dest_acc.cpp_enum_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        data_copy_type = compute_unit.data_copy_type.cpp_enum_value
        num_faces = operation.output.tile_shape.total_num_faces()
        _int_fpu_formats = {DataFormat.Int8, DataFormat.UInt8, DataFormat.Int32}
        is_int_fpu_en = (
            "true"
            if (
                (
                    compute_unit.src_a is not None
                    and compute_unit.src_a.data_format in _int_fpu_formats
                )
                or (
                    compute_unit.src_b is not None
                    and compute_unit.src_b.data_format in _int_fpu_formats
                )
            )
            else "false"
        )

        return (
            f"_llk_math_eltwise_unary_datacopy_init_<{data_copy_type}, {dest_acc}, {broadcast_type}, {is_int_fpu_en}>(\n"
            f"    {num_faces}, {config.sentinel.math_format}\n"
            f");\n"
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
        unpack_to_dest = compute_unit.unpack_to_dest.cpp_enum_value
        data_copy_type = f"DataCopyType::{compute_unit.data_copy_type.name}"

        code = (
            f"    _llk_math_eltwise_unary_datacopy_<{data_copy_type}, dest_sync{stage}, {dest_acc}, {broadcast_type}, {unpack_to_dest}>(\n"
            f"        {block.tile_id_block}, {config.sentinel.math_format}, {config.sentinel.math_format}\n"
            f"    );\n"
        )

        return code

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        unpack_to_dest = compute_unit.unpack_to_dest.cpp_enum_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        return f"_llk_math_eltwise_unary_datacopy_uninit_<{broadcast_type}, {unpack_to_dest}>();\n"
