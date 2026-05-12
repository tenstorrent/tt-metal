# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig
    from .block_data import BlockData
    from .compute_node import ComputeNode

from helpers.golden_generators import PackGolden
from helpers.tilize_untilize import tilize_block, untilize_block

from .fused_loop import FusedLoop


class Packer:
    loop: FusedLoop = FusedLoop()

    @staticmethod
    def _l1_acc_golden(
        tensor: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        output_dims = operation.output.dimensions
        output_format = operation.output.data_format
        tile_size = operation.output.tile_shape.total_tile_size()
        tile_count_x = operation.output.tile_count_x
        tile_count_y = operation.output.tile_count_y
        block_tiles_x = operation.block_tiles_x
        block_tiles_y = operation.block_tiles_y

        tensor = tilize_block(tensor, output_dims, output_format).flatten()
        tile_grid = tensor.view(tile_count_y, tile_count_x, tile_size)

        accumulated = torch.zeros(
            block_tiles_y, block_tiles_x, tile_size, dtype=tensor.dtype
        )
        for by in range(0, tile_count_y, block_tiles_y):
            for bx in range(0, tile_count_x, block_tiles_x):
                bty = min(block_tiles_y, tile_count_y - by)
                btx = min(block_tiles_x, tile_count_x - bx)
                accumulated[:bty, :btx] += tile_grid[by : by + bty, bx : bx + btx]

        result_grid = torch.zeros(
            tile_count_y, tile_count_x, tile_size, dtype=tensor.dtype
        )
        result_grid[:block_tiles_y, :block_tiles_x] = accumulated
        return untilize_block(result_grid.flatten(), output_format, output_dims)

    @staticmethod
    def _relu_golden(
        tensor: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        intermediate_format = config.sentinel.golden_format.pack_src
        relu_config = PackGolden.generate_relu_config(
            operation.pack_relu, operation.relu_threshold, intermediate_format
        )
        return PackGolden.apply_relu(tensor, relu_config, intermediate_format)

    def get_headers(self) -> List[str]:
        return []

    def golden(
        self,
        tensor: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        return tensor

    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        block: "BlockData",
    ) -> str:
        return ""

    def pack(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        block: "BlockData",
    ) -> str:
        return ""

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        block: "BlockData",
    ) -> str:
        return ""
