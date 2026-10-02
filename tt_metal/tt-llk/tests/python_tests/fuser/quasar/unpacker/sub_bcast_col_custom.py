# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.unpack.sub_bcast_col_custom import unpack_sub_bcast_col_custom_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from fuser.operand import BfdResource, bfd_current


class SubBcastColCustomUnpacker(Unpacker):
    granularity = InvocationGranularity.ROW
    per_block_init = True
    golden_fn = staticmethod(unpack_sub_bcast_col_custom_golden)

    def get_headers(self) -> List[str]:
        return ["experimental/llk_unpack_AB_sub_bcast_col_custom.h"]

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return (
            "_perf_unpack_loop_set_valid<false, true>(1);\n"
            f"_perf_unpack_loop_set_valid<true, false>({block.block_cols});\n"
        )

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return (
            f"_perf_math_loop_clear_valid<true, false>({block.block_cols});\n"
            "_perf_math_loop_clear_valid<false, true>(1);\n"
        )

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        bfd_program = compute_unit.src_a.bfd_alloc_and_program(
            BfdResource.UNP0
        ) + compute_unit.src_b.bfd_alloc_and_program(BfdResource.UNP1)
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        return (
            bfd_program
            + f"_llk_unpack_AB_sub_bcast_col_init_custom_({tensor_shape});\n"
        )

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        id_a = bfd_current(BfdResource.UNP0)
        id_b = bfd_current(BfdResource.UNP1)
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        return (
            f"_llk_unpack_AB_sub_bcast_col_custom_({id_a}, {id_b}, "
            f"{block.tile_id_src_a}, {block.tile_id_src_b}, {block.block_cols}, {tensor_shape});\n"
        )
