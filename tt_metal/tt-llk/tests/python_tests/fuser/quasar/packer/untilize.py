# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from fuser.block_data import BlockData
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from fuser.pack_node import PackNode
from fuser.tile_loop import LoopBlockRow, TileLoop
from helpers.llk_params import PackerReluType

from .packer import Packer


class PackUntilize(Packer):
    loop: TileLoop = LoopBlockRow()
    per_block_init = True

    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_untilize.h",
        ]

    def golden(
        self,
        tensor: torch.Tensor,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
    ) -> torch.Tensor:
        if pack_node.pack_relu != PackerReluType.NoRelu:
            tensor = self.relu_golden(tensor, config, operation, pack_node)

        return self.untilize_golden(tensor, config, operation, pack_node)

    def init(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        full_ct_dim = pack_node.output.tile_count_x
        block_ct_dim = block.block_tiles_x
        buf_desc_id = pack_node.output.buf_desc_id
        tensor_shape = pack_node.output.tile_shape.cpp_value

        return f"_llk_pack_untilize_init_<{full_ct_dim}, {block_ct_dim}>({buf_desc_id}, {tensor_shape});\n"

    def pack(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        tile_shape = pack_node.output.tile_shape
        y_stride = (
            pack_node.output.tile_count_x
            * tile_shape.num_faces_r_dim
            * tile_shape.face_r_dim
        )
        l1_row_idx = f"{y_stride} * ({block.block_y} + tile_y) + {block.block_x}"

        return f"_llk_pack_untilize_({block.tile_id_block}, {l1_row_idx});\n"
