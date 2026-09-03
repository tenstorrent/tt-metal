# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.base_fpu import Fpu
from fuser.block_data import BlockData, InvocationGranularity
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from helpers.llk_params import DataFormat


class DatacopyFpu(Fpu):
    granularity = InvocationGranularity.TILE

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
        dest_acc = config.dest_acc.cpp_enum_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        data_copy_type = compute_unit.data_copy_type.cpp_enum_value
        num_faces = operation.tile_shape.total_num_faces()
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
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        dest_sync = operation.dest_sync.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        unpack_to_dest = compute_unit.unpack_to_dest.cpp_enum_value
        data_copy_type = f"DataCopyType::{compute_unit.data_copy_type.name}"

        code = (
            f"    _llk_math_eltwise_unary_datacopy_<{data_copy_type}, {dest_sync}, {dest_acc}, {broadcast_type}, {unpack_to_dest}>(\n"
            f"        {block.tile_id_block}, {config.sentinel.math_format}, {config.sentinel.math_format}\n"
            f"    );\n"
        )

        return code

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        unpack_to_dest = compute_unit.unpack_to_dest.cpp_enum_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        return f"_llk_math_eltwise_unary_datacopy_uninit_<{broadcast_type}, {unpack_to_dest}>();\n"
