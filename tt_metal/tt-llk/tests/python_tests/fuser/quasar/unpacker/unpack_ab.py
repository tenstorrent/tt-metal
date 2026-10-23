# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.unpack.unpack_ab import unpack_ab_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from fuser.operand import BfdResource
from helpers.llk_params import BroadcastType


class UnpackerAB(Unpacker):
    granularity = InvocationGranularity.TILE
    golden_fn = staticmethod(unpack_ab_golden)

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_binary_operands.h",
            "llk_unpack_binary_broadcast_operands.h",
            "llk_unpack_common.h",
        ]

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.broadcast_type == BroadcastType.None_:
            return "_perf_unpack_loop_set_valid<true, true>(1);\n"
        srcb_dvalids = (
            1
            if compute_unit.broadcast_type == BroadcastType.Scalar
            else compute_unit.src_a.tile_shape.total_num_faces()
        )
        return (
            f"_perf_unpack_loop_set_valid<true, false>(1);\n"
            f"_perf_unpack_loop_set_valid<false, true>({srcb_dvalids});\n"
        )

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.broadcast_type == BroadcastType.None_:
            return "_perf_math_loop_clear_valid<true, true>(1);\n"
        srcb_only_clears = (
            0
            if compute_unit.broadcast_type == BroadcastType.Scalar
            else compute_unit.src_a.tile_shape.total_num_faces() - 1
        )
        code = ""
        if srcb_only_clears > 0:
            code += f"_perf_math_loop_clear_valid<false, true>({srcb_only_clears});\n"
        code += "_perf_math_loop_clear_valid<true, true>(1);\n"
        return code

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        bfd_program = compute_unit.src_a.bfd_alloc_and_program(
            BfdResource.UNP0, result_name="bfd_a"
        ) + compute_unit.src_b.bfd_alloc_and_program(
            BfdResource.UNP1, result_name="bfd_b"
        )

        if compute_unit.broadcast_type != BroadcastType.None_:
            broadcast_type = compute_unit.broadcast_type.cpp_enum_value
            tensor_shape = compute_unit.src_a.tile_shape.cpp_value
            return (
                "{\n"
                + bfd_program
                + f"_llk_unpack_binary_broadcast_operands_init_<{broadcast_type}>"
                f"(bfd_a, bfd_b, {tensor_shape}, 1);\n}}\n"
            )

        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        return (
            "{\n"
            + bfd_program
            + f"_llk_unpack_binary_operands_init_(bfd_a, bfd_b, {tensor_shape}, 1);\n}}\n"
        )

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        if compute_unit.broadcast_type != BroadcastType.None_:
            tile_id_b = block.tile_id_src_b
            tensor_shape = compute_unit.src_a.tile_shape.cpp_value
            return f"_llk_unpack_binary_broadcast_operands_({block.tile_id_src_a}, {tile_id_b}, {tensor_shape});\n"

        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        return f"_llk_unpack_binary_operands_({block.tile_id_src_a}, {block.tile_id_src_b}, {tensor_shape});\n"

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
