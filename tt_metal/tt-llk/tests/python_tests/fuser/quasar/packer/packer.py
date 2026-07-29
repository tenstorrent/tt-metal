# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from fuser.block_data import BlockData
from fuser.fused_loop import FusedLoop
from fuser.fused_operation import FusedOperation
from fuser.fused_packer import Packer as BasePacker
from fuser.fuser_config import GlobalConfig
from fuser.pack_node import PackNode
from helpers.llk_params import L1Accumulation, PackerReluType


class Packer(BasePacker):
    loop: FusedLoop = FusedLoop()

    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_common.h",
        ]

    def golden(
        self,
        tensor: torch.Tensor,
        pack_node: PackNode,
        operation: FusedOperation,
        config: GlobalConfig,
    ) -> torch.Tensor:
        if pack_node.pack_relu != PackerReluType.NoRelu:
            tensor = self._relu_golden(tensor, pack_node, config)

        if pack_node.pack_l1_accumulation == L1Accumulation.Yes:
            tensor = self._l1_acc_golden(tensor, pack_node, operation, config)

        return tensor

    def init(
        self,
        pack_node: PackNode,
        operation: FusedOperation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        buf_desc_id = pack_node.output.buf_desc_id
        tensor_shape = pack_node.output.tile_shape.cpp_value
        en_32bit_dest = config.dest_acc.cpp_enum_value
        return (
            f"_llk_pack_init_<{en_32bit_dest}>({buf_desc_id}, " f"{tensor_shape}, 1);\n"
        )

    def pack(
        self,
        pack_node: PackNode,
        operation: FusedOperation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        return f"_llk_pack_({block.tile_id_block}, {block.tile_id_global}, ckernel::DEFAULT_TENSOR_SHAPE);\n"
