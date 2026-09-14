# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from fuser.tile_loop import LoopTileByTile, TileLoop
from helpers.llk_params import ReduceDimension, ReducePool


class ReduceUnpacker(Unpacker):
    loop: TileLoop = LoopTileByTile()

    def __init__(self, reduce_dim, reduce_pool):
        self.reduce_dim = reduce_dim
        self.reduce_pool = reduce_pool

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
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor_a, tensor_b

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        num_faces = compute_unit.src_a.tile_shape.total_num_faces()
        is_full_tile = (
            self.reduce_dim == ReduceDimension.Row
            and self.reduce_pool != ReducePool.Max
            and num_faces == 4
        )
        iterations = 1 if is_full_tile else num_faces
        return f"_perf_unpack_loop_set_valid<true, true>({iterations});\n"

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        num_faces = compute_unit.src_a.tile_shape.total_num_faces()
        is_full_tile = (
            self.reduce_dim == ReduceDimension.Row
            and self.reduce_pool != ReducePool.Max
            and num_faces == 4
        )
        iterations = 1 if is_full_tile else num_faces
        return f"_perf_math_loop_clear_valid<true, true>({iterations});\n"

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        reduce_dim = self.reduce_dim.cpp_enum_value
        pool_type = self.reduce_pool.cpp_enum_value

        return (
            f"_llk_unpack_AB_reduce_init_<{pool_type}, {reduce_dim}>(\n"
            f"{compute_unit.src_a.tile_shape.cpp_value});\n"
        )

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        buffer_a = compute_unit.src_a.cpp_name
        buffer_b = compute_unit.src_b.cpp_name
        reduce_dim = self.reduce_dim.cpp_enum_value
        pool_type = self.reduce_pool.cpp_enum_value
        return f"_llk_unpack_AB_reduce_<{pool_type}, {reduce_dim}>(L1_ADDRESS({buffer_a}[{block.tile_id_global}]), L1_ADDRESS({buffer_b}[{block.tile_id_global}]));\n"
