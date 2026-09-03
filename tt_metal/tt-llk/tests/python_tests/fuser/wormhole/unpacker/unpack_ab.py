# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from helpers.llk_params import (
    BroadcastType,
    UnpackToDest,
)


class UnpackerAB(Unpacker):
    granularity = InvocationGranularity.TILE

    per_call_golden = True

    def supports_per_call(self, node) -> bool:
        return super().supports_per_call(node) and (
            node.unpack_to_dest == UnpackToDest.No
        )

    def golden_call(
        self,
        call,
        inputs,
        srcs,
        compute_unit: FpuNode,
        operation: L1Operation,
        config: GlobalConfig,
    ) -> None:
        tile_b = inputs.tile_b(call.in1)
        if tile_b is not None:
            tile_b = self.broadcast_tile_golden(
                tile_b, operation, compute_unit, compute_unit.src_b
            )
        tile_a = inputs.tile_a(call.in0)
        if tile_a is not None:
            tile_a = self.transpose_tile_golden(tile_a, config, operation, compute_unit)
        srcs.push(tile_a, tile_b)

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_AB.h",
            "llk_unpack_common.h",
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
        tensor_a = self.transpose_golden(tensor_a, config, operation, compute_unit)

        return tensor_a.flatten(), tensor_b.flatten()

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        num_faces = compute_unit.src_a.tile_shape.total_num_faces()
        if compute_unit.broadcast_type == BroadcastType.Scalar:
            return (
                f"_perf_unpack_loop_set_valid<false, true>(1);\n"
                f"_perf_unpack_loop_set_valid<true, false>({num_faces});\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Column:
            return (
                f"_perf_unpack_loop_set_valid<false, true>(2);\n"
                f"_perf_unpack_loop_set_valid<true, false>({num_faces});\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Row:
            return f"_perf_unpack_loop_set_valid<true, true>({num_faces});\n"
        else:
            return f"_perf_unpack_loop_set_valid<true, true>({num_faces});\n"

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        num_faces = compute_unit.src_a.tile_shape.total_num_faces()
        if compute_unit.broadcast_type == BroadcastType.Scalar:
            return (
                f"_perf_math_loop_clear_valid<false, true>(1);\n"
                f"_perf_math_loop_clear_valid<true, false>({num_faces});\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Column:
            return (
                f"_perf_math_loop_clear_valid<false, true>(2);\n"
                f"_perf_math_loop_clear_valid<true, false>({num_faces});\n"
            )
        elif compute_unit.broadcast_type == BroadcastType.Row:
            return f"_perf_math_loop_clear_valid<true, true>({num_faces});\n"
        else:
            return f"_perf_math_loop_clear_valid<true, true>({num_faces});\n"

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        broadcast_type = compute_unit.broadcast_type.cpp_enum_value
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        if compute_unit.transpose_faces.value:
            transpose_value = (
                "ckernel::Transpose::Both"
                if compute_unit.transpose_within_face.value
                else "ckernel::Transpose::InterFace"
            )
        else:
            transpose_value = (
                "ckernel::Transpose::IntraFace"
                if compute_unit.transpose_within_face.value
                else "ckernel::Transpose::None"
            )

        return f"_llk_unpack_AB_init_<{broadcast_type}>({tensor_shape}, {transpose_value});\n"

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        broadcast_type = f"BroadcastType::{compute_unit.broadcast_type.value}"
        buffer_a = compute_unit.src_a.cpp_name
        buffer_b = compute_unit.src_b.cpp_name
        tile_id_b = (
            block.tile_id_global
            if compute_unit.broadcast_tile is None
            else compute_unit.broadcast_tile
        )
        return f"_llk_unpack_AB_<{broadcast_type}>(L1_ADDRESS({buffer_a}[{block.tile_id_global}]), L1_ADDRESS({buffer_b}[{tile_id_b}]));\n"

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return f"_llk_unpack_AB_uninit_();\n"
