# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.unpack.unpack import unpack_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from helpers.llk_params import ReduceDimension, ReducePool


class ReduceUnpacker(Unpacker):
    granularity = InvocationGranularity.TILE
    golden_fn = staticmethod(unpack_golden)

    def __init__(self, reduce_dim, reduce_pool):
        self.reduce_dim = reduce_dim
        self.reduce_pool = reduce_pool
        self.reverse_operands = (
            reduce_dim == ReduceDimension.Row and reduce_pool != ReducePool.Max
        )

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_AB.h",
            "llk_unpack_AB_reduce.h",
            "llk_unpack_common.h",
            "llk_unpack_tilize.h",
        ]

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_perf_unpack_loop_set_valid<true, true>(1);\n"

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_perf_math_loop_clear_valid<true, true>(1);\n"

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
        return f"_llk_unpack_AB_reduce_<{pool_type}, {reduce_dim}>(L1_ADDRESS({buffer_a}[{block.tile_id_src_a}]), L1_ADDRESS({buffer_b}[{block.tile_id_src_b}]));\n"
