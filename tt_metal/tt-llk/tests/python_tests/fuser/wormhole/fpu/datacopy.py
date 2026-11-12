# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.base_fpu import Fpu
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.fpu.datacopy import datacopy_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from helpers.tile_shape import cpp_tensor_shape


class DatacopyFpu(Fpu):
    granularity = InvocationGranularity.TILE
    golden_fn = staticmethod(datacopy_golden)

    def get_headers(self) -> List[str]:
        return [
            "llk_math_common.h",
            "llk_math_eltwise_unary_datacopy.h",
        ]

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
        tensor_shape = cpp_tensor_shape(operation.tile_shape)
        is_int_fpu_en = compute_unit.int_fpu_en

        return (
            f"_llk_math_eltwise_unary_datacopy_init_<{data_copy_type}, {dest_acc}, {broadcast_type}, {is_int_fpu_en}>(\n"
            f"    {tensor_shape}, {compute_unit.unpack_A_dst.cpp_underlying_value}\n"
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
        tensor_shape = cpp_tensor_shape(operation.tile_shape)
        src_format = compute_unit.unpack_A_src.cpp_underlying_value
        dst_format = compute_unit.unpack_A_dst.cpp_underlying_value

        code = (
            f"    _llk_math_eltwise_unary_datacopy_<{data_copy_type}, {dest_sync}, {dest_acc}, {broadcast_type}, {unpack_to_dest}>(\n"
            f"        {block.tile_id_dest}, {src_format}, {dst_format}, {tensor_shape}\n"
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
