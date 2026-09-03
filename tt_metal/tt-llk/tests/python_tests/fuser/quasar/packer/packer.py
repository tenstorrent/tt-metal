# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from fuser.base_packer import Packer as BasePacker
from fuser.block_data import BlockData
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from fuser.pack_node import PackNode
from helpers.llk_params import PackerReluType


class Packer(BasePacker):

    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_common.h",
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

        return tensor

    def init(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        buf_desc_id = pack_node.output.buf_desc_id
        tensor_shape = pack_node.output.tile_shape.cpp_value
        return f"_llk_pack_init_({buf_desc_id}, {tensor_shape}, 1);\n"

    def pack(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        return f"_llk_pack_({block.tile_id_block}, {block.tile_id_global}, ckernel::DEFAULT_TENSOR_SHAPE);\n"
