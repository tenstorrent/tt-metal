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


class SubBcastColCustomUnpacker(Unpacker):
    granularity = InvocationGranularity.ROW
    per_block_init = True

    golden_fn = staticmethod(unpack_sub_bcast_col_custom_golden)

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "experimental/llk_unpack_AB_sub_bcast_col_custom.h",
        ]

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_cols
        return (
            f"_perf_unpack_loop_set_valid<false, true>(1);\n"
            f"_perf_unpack_loop_set_valid<true, false>({ct_dim});\n"
        )

    def perf_clear_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_cols
        return (
            f"_perf_math_loop_clear_valid<true, false>({ct_dim});\n"
            f"_perf_math_loop_clear_valid<false, true>(1);\n"
        )

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_unpack_AB_sub_bcast_col_init_custom_();\n"

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_cols
        buffer_a = compute_unit.src_a.cpp_name
        buffer_b = compute_unit.src_b.cpp_name
        return (
            f"_llk_unpack_AB_sub_bcast_col_custom_("
            f"L1_ADDRESS({buffer_a}[{block.tile_id_src_a}]), "
            f"L1_ADDRESS({buffer_b}[{block.tile_id_src_b}]), "
            f"{ct_dim});\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_unpack_AB_sub_bcast_col_uninit_custom_();\n"
