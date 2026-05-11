# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from fuser.block_data import BlockData
from fuser.compute_node import ComputeNode
from fuser.fused_loop import FusedLoop
from fuser.fused_operation import FusedOperation
from fuser.fused_packer import Packer as BasePacker
from fuser.fuser_config import GlobalConfig
from helpers.golden_generators import PackGolden
from helpers.llk_params import L1Accumulation, PackerReluType
from helpers.tilize_untilize import tilize_block, untilize_block


class Packer(BasePacker):
    loop: FusedLoop = FusedLoop()

    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_common.h",
            "perf.h",
        ]

    def golden(
        self,
        tensor: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
    ) -> torch.Tensor:
        if operation.pack_relu != PackerReluType.NoRelu:
            intermediate_format = config.sentinel.golden_format.pack_src
            relu_config = PackGolden.generate_relu_config(
                operation.pack_relu, operation.relu_threshold, intermediate_format
            )
            tensor = PackGolden.apply_relu(tensor, relu_config, intermediate_format)

        if operation.pack_l1_acc == L1Accumulation.Yes:
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
            tensor = untilize_block(result_grid.flatten(), output_format, output_dims)

        return tensor

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        dest_acc = config.dest_acc.cpp_enum_value
        bh_tilize = operation.bh_tilize.cpp_enum_value
        face_r_dim = operation.output.tile_shape.face_r_dim
        num_faces = operation.output.tile_shape.total_num_faces()
        dest_sync = f"DstSync::Sync{operation.dest_sync.name}"
        return (
            f"    _llk_pack_init_<false, false, {bh_tilize}>(\n"
            f"        {config.sentinel.pack_src_format}, {face_r_dim}, TILE_C_DIM, {num_faces}, 1\n"
            f"    );\n"
            f"    _llk_pack_dest_init_<{dest_sync}, {dest_acc}>();\n"
        )

    def pack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        dest_acc = config.dest_acc.cpp_enum_value
        dest_sync = f"DstSync::Sync{operation.dest_sync.name}"
        buffer = operation.output.cpp_name
        return f"_llk_pack_<{dest_sync}, {dest_acc}, false>({block.tile_id_block}, L1_ADDRESS({buffer}[{block.tile_id_global}]));\n"
