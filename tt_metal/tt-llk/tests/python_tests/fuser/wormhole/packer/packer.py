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
from fuser.tile_loop import TileLoop
from helpers.llk_params import L1Accumulation, PackerReluType


class Packer(BasePacker):
    loop: TileLoop = TileLoop()

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

        if pack_node.pack_l1_accumulation == L1Accumulation.Yes:
            tensor = self.l1_acc_golden(tensor, config, operation, pack_node)

        return tensor

    def init(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        face_r_dim = pack_node.output.tile_shape.face_r_dim
        num_faces = pack_node.output.tile_shape.total_num_faces()
        return (
            f"    _llk_pack_init_<PackMode::Default, false /* zero_output */>(\n"
            f"        {config.sentinel.pack_dst_format}, {face_r_dim}, {num_faces}\n"
            f"    );\n"
        )

    def pack(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        dest_acc = config.dest_acc.cpp_enum_value
        dest_sync = f"DstSync::Sync{operation.dest_sync.name}"
        buffer = pack_node.output.cpp_name
        return f"_llk_pack_<{dest_sync}, {dest_acc}, ckernel::PackMode::Default>({block.tile_id_block}, L1_ADDRESS({buffer}[{block.tile_id_global}]));\n"
