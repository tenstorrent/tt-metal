# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.compute_node import ComputeNode
from fuser.fused_fpu import Fpu
from fuser.fused_loop import FusedLoop, LoopTileByTile
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig
from helpers.golden_generators import ReduceBlockMaxRowGolden, get_golden_generator


class ReduceBlockMaxFpu(Fpu):
    loop: FusedLoop = LoopTileByTile()

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        dest_acc = config.dest_acc.cpp_enum_value
        return f"_llk_math_reduce_block_max_row_init_<{ct_dim}, {dest_acc}>();\n"

    def calculate(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        ct_dim = block.block_tiles_x
        dest_acc = config.dest_acc.cpp_enum_value
        tile_x_in_block = f"(({block.tile_id_block}) % {block.block_tiles_x})"
        tile_y_in_block = f"(({block.tile_id_block}) / {block.block_tiles_x})"
        dest_expr = f"(({tile_y_in_block}) * {block.block_tiles_x})"
        return (
            f"if (({tile_x_in_block}) % {ct_dim} == 0 ) {{\n"
            f"    _llk_math_reduce_block_max_row_<{ct_dim}, {dest_acc}>({dest_expr});\n"
            f"}}\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        return "_llk_math_reduce_block_max_row_uninit_();\n"

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        tensor_dst: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_format = operation.output.data_format

        golden_tensor = torch.zeros_like(tensor_dst)
        src_a_reduced_tensor = torch.zeros_like(tensor_a)
        dest_golden_tensor = torch.zeros_like(tensor_dst)

        tile_count_x = operation.output.tile_count_x
        tile_count_y = operation.output.tile_count_y
        block_tiles_x = operation.block_tiles_x
        block_tiles_y = operation.block_tiles_y

        full_blocks_x = tile_count_x // block_tiles_x
        full_blocks_y = tile_count_y // block_tiles_y
        remaining_tiles_x = tile_count_x % block_tiles_x
        remaining_tiles_y = tile_count_y % block_tiles_y

        full_x_limit = full_blocks_x * block_tiles_x
        full_y_limit = full_blocks_y * block_tiles_y

        generate_golden = get_golden_generator(ReduceBlockMaxRowGolden)

        def process_block(block_x, block_y, block_tiles_x_eff, block_tiles_y_eff):
            src_start_row = block_y * 32
            src_end_row = (block_y + block_tiles_y_eff) * 32
            start_col = block_x * 32
            end_col = (block_x + block_tiles_x_eff) * 32
            dst_start_row = block_y * 32
            dst_end_row = (block_y + block_tiles_y_eff) * 32
            block_dims = [block_tiles_y_eff * 32, block_tiles_x_eff * 32]

            src_a_reduced_tensor[dst_start_row:dst_end_row, start_col:end_col] = (
                generate_golden(
                    tensor_a[src_start_row:src_end_row, start_col:end_col].clone(),
                    block_tiles_x_eff,
                    output_format,
                    block_dims,
                )
            )

            dest_golden_tensor[dst_start_row:dst_end_row, start_col:end_col] = (
                generate_golden(
                    tensor_dst[src_start_row:src_end_row, start_col:end_col].clone(),
                    block_tiles_x_eff,
                    output_format,
                    block_dims,
                )
            )

        if full_blocks_x > 0 and full_blocks_y > 0:
            for block_x in range(0, full_x_limit, block_tiles_x):
                for block_y in range(0, full_y_limit, block_tiles_y):
                    process_block(block_x, block_y, block_tiles_x, block_tiles_y)

        if remaining_tiles_y > 0 and full_blocks_x > 0:
            for block_x in range(0, full_x_limit, block_tiles_x):
                process_block(block_x, full_y_limit, block_tiles_x, remaining_tiles_y)

        if remaining_tiles_x > 0 and full_blocks_y > 0:
            for block_y in range(0, full_y_limit, block_tiles_y):
                process_block(full_x_limit, block_y, remaining_tiles_x, block_tiles_y)

        if remaining_tiles_x > 0 and remaining_tiles_y > 0:
            process_block(
                full_x_limit, full_y_limit, remaining_tiles_x, remaining_tiles_y
            )

        golden_tensor = golden_tensor.flatten()
        src_a_reduced_tensor = src_a_reduced_tensor.flatten()
        dest_golden_tensor = dest_golden_tensor.flatten()

        for i in range(golden_tensor.numel()):
            golden_tensor[i] = max(src_a_reduced_tensor[i], dest_golden_tensor[i])

        return (tensor_a, tensor_b, golden_tensor)

    def get_headers(self) -> List[str]:
        return ["experimental/llk_math_reduce_custom.h"]
