# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.golden.unpack.reduce_block_max import unpack_reduce_block_max_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from fuser.operand import BfdResource, bfd_current


class ReduceBlockMaxRuntimeUnpacker(Unpacker):
    granularity = InvocationGranularity.ROW
    per_block_init = True
    golden_fn = staticmethod(unpack_reduce_block_max_golden)

    def get_headers(self) -> List[str]:
        return ["experimental/llk_unpack_AB_reduce_runtime_custom.h"]

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        num_faces = block.block_cols * compute_unit.src_a.tile_shape.total_num_faces()
        return (
            "_perf_unpack_loop_set_valid<false, true>(1);\n"
            f"_perf_unpack_loop_set_valid<true, false>({num_faces});\n"
        )

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        num_faces = block.block_cols * compute_unit.src_a.tile_shape.total_num_faces()
        return (
            f"_perf_math_loop_clear_valid<true, false>({num_faces});\n"
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
        id_a = bfd_current(BfdResource.UNP0)
        id_b = bfd_current(BfdResource.UNP1)
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        dest_acc = config.dest_acc.cpp_enum_value
        return (
            bfd_program
            + f"_llk_unpack_AB_reduce_block_max_row_init_runtime_<{dest_acc}>"
            f"({block.block_cols}, false, {id_a}, {id_b}, {tensor_shape});\n"
        )

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        id_b = bfd_current(BfdResource.UNP1)
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        return (
            f"_llk_unpack_AB_reduce_block_max_row_runtime_({block.block_cols}, "
            f"{block.tile_id_src_a}, {block.tile_id_src_b}, {id_b}, {tensor_shape});\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_unpack_AB_reduce_block_max_row_uninit_runtime_();\n"
