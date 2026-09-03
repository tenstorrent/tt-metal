# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData, InvocationGranularity
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from helpers.llk_params import BroadcastType


class UnpackerA(Unpacker):
    granularity = InvocationGranularity.TILE

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_A.h",
            "llk_unpack_common.h",
            "llk_unpack_tilize.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if compute_unit.broadcast_type != BroadcastType.None_:
            tensor_b = self.broadcast_golden(
                tensor_a, config, operation, compute_unit, operand=compute_unit.src_a
            )
            tensor_a = None
        else:
            tensor_a = self.transpose_golden(tensor_a, config, operation, compute_unit)
            tensor_b = None

        tensor_a, tensor_b = self.reuse_dest_golden(
            tensor_a, tensor_b, config, operation, compute_unit
        )

        return tensor_a, tensor_b

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.broadcast_type == BroadcastType.Scalar:
            return "_perf_unpack_loop_set_valid<true, true>(1);\n"
        elif compute_unit.broadcast_type == BroadcastType.Column:
            return (
                "_perf_unpack_loop_set_valid<false, true>(2);\n"
                "_perf_unpack_loop_set_valid<true, false>(1);\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Row:
            return "_perf_unpack_loop_set_valid<false, true>(4);\n"
        else:
            num_faces = compute_unit.src_a.tile_shape.total_num_faces()
            return f"_perf_unpack_loop_set_valid<true, true>({num_faces});\n"

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.broadcast_type == BroadcastType.Scalar:
            return "_perf_math_loop_clear_valid<true, true>(1);\n"
        elif compute_unit.broadcast_type == BroadcastType.Column:
            return (
                "_perf_math_loop_clear_valid<false, true>(2);\n"
                "_perf_math_loop_clear_valid<true, false>(1);\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Row:
            return "_perf_math_loop_clear_valid<false, true>(4);\n"
        else:
            num_faces = compute_unit.src_a.tile_shape.total_num_faces()
            return f"_perf_math_loop_clear_valid<true, true>({num_faces});\n"

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        unpack_to_dest = compute_unit.unpack_to_dest.cpp_enum_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        transpose_faces = compute_unit.transpose_faces.cpp_enum_value
        transpose_within_face = compute_unit.transpose_within_face.cpp_enum_value
        acc_to_dest = compute_unit.acc_to_dest.cpp_enum_value

        return (
            f"_llk_unpack_A_init_<{broadcast_type}, {acc_to_dest}, {reuse_dest}, {unpack_to_dest}>(\n"
            f"    {transpose_faces}, {transpose_within_face}, {tensor_shape}, {config.sentinel.unpack_a_src_format}, {config.sentinel.unpack_a_dst_format}\n"
            f");\n"
        )

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        unpack_to_dest = compute_unit.unpack_to_dest.cpp_enum_value
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        acc_to_dest = compute_unit.acc_to_dest.cpp_enum_value
        buffer_a = compute_unit.src_a.cpp_name

        return (
            f"_llk_unpack_A_<{broadcast_type}, {acc_to_dest}, {reuse_dest}, {unpack_to_dest}>(\n"
            f"    L1_ADDRESS({buffer_a}[{block.tile_id_global}]), {config.sentinel.unpack_a_src_format}, {config.sentinel.unpack_a_dst_format}\n"
            f");\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        return f"_llk_unpack_A_uninit_<{broadcast_type}>();\n"
