# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fused_loop import FusedLoop, LoopBlockRow
from fuser.fused_math import ComputeNode
from fuser.fused_operation import FusedOperation
from fuser.fused_unpacker import Unpacker
from fuser.fuser_config import GlobalConfig


class ReduceBlockMaxRuntimeUnpacker(Unpacker):
    loop: FusedLoop = LoopBlockRow()
    per_block_init = True

    def get_headers(self) -> List[str]:
        return ["experimental/llk_unpack_AB_reduce_custom_runtime.h"]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor_a, tensor_b

    def perf_set_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        return (
            f"_perf_unpack_loop_set_valid<false, true>(1);\n"
            f"_perf_unpack_loop_set_valid<true, false>({ct_dim});\n"
        )

    def perf_clear_valid(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        return (
            f"_perf_math_loop_clear_valid<true, false>({ct_dim});\n"
            f"_perf_math_loop_clear_valid<false, true>(1);\n"
        )

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        dest_acc = config.dest_acc.cpp_enum_value
        return f"_llk_unpack_AB_reduce_block_max_row_init_runtime_<{dest_acc}>({ct_dim});\n"

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        buffer_a = compute_unit.src_a.cpp_name
        buffer_b = compute_unit.src_b.cpp_name
        return f"_llk_unpack_AB_reduce_block_max_row_runtime_(L1_ADDRESS({buffer_a}[{block.tile_id_global}]), L1_ADDRESS({buffer_b}[{block.tile_id_global}]));\n"

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        return f"_llk_unpack_AB_reduce_block_max_row_uninit_runtime_();\n"
