# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fused_loop import FusedLoop, LoopTileByTile
from fuser.fused_math import ComputeNode
from fuser.fused_operation import FusedOperation
from fuser.fused_unpacker import Unpacker
from fuser.fuser_config import GlobalConfig


class ReduceUnpacker(Unpacker):
    loop: FusedLoop = LoopTileByTile()

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_AB.h",
            "llk_unpack_AB_reduce.h",
            "llk_unpack_common.h",
            "llk_unpack_tilize.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor_a, tensor_b

    def perf_set_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        num_faces = operation.num_faces
        face_r_dim = operation.face_r_dim
        return (
            f"_perf_unpack_loop_set_valid<false, true>(1);\n"
            f"_perf_unpack_loop_set_valid<true, false>({face_r_dim * num_faces});\n"
        )

    def perf_clear_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        num_faces = operation.num_faces
        face_r_dim = operation.face_r_dim
        return (
            f"_perf_math_loop_clear_valid<false, true>(1);\n"
            f"_perf_math_loop_clear_valid<true, false>({face_r_dim * num_faces});\n"
        )

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        reduce_dim = compute_unit.reduce_dim.cpp_enum_value
        pool_type = compute_unit.reduce_pool.cpp_enum_value

        tile_shape = operation.src_a.tile_shape
        tensor_shape_instantiation: str = (
            f"ckernel::TensorShape{{{tile_shape.face_r_dim}, {tile_shape.face_c_dim}, {tile_shape.num_faces_r_dim}, {tile_shape.num_faces_c_dim}}}"
        )

        return (
            f"_llk_unpack_AB_reduce_init_<{pool_type}, {reduce_dim}>(\n"
            f"{tensor_shape_instantiation});\n"
        )

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id

        reduce_dim = compute_unit.reduce_dim.cpp_enum_value
        pool_type = compute_unit.reduce_pool.cpp_enum_value
        return f"_llk_unpack_AB_reduce_<{pool_type}, {reduce_dim}>(L1_ADDRESS(buffer_A{stage}[{block.tile_id_global}]), L1_ADDRESS(buffer_B{stage}[{block.tile_id_global}]));\n"
