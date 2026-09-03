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


class SubBcastColCustomUnpacker(Unpacker):
    granularity = InvocationGranularity.ROW
    per_block_init = True

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "experimental/llk_unpack_AB_sub_bcast_col_custom.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor_b = self.broadcast_golden(
            tensor_b, config, operation, compute_unit, per_block=True
        )
        return tensor_a.flatten(), tensor_b.flatten()

    def perf_set_valid(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
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
        ct_dim = block.block_tiles_x
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
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        return f"_llk_unpack_AB_sub_bcast_col_init_custom_({tensor_shape});\n"

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        buffer_a = compute_unit.src_a.cpp_name
        buffer_b = compute_unit.src_b.cpp_name
        return (
            f"_llk_unpack_AB_sub_bcast_col_custom_("
            f"L1_ADDRESS({buffer_a}[{block.tile_id_global}]), "
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
