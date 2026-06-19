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
            "perf.h",
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
        bh_pack_mode = operation.bh_tilize.pack_mode_value
        face_r_dim = pack_node.output.tile_shape.face_r_dim
        num_faces = pack_node.output.tile_shape.total_num_faces()
        return (
            f"    _llk_pack_init_<{bh_pack_mode}, false /* zero_output */, false /* skip_addrmod_config */>(\n"
            f"        {config.sentinel.pack_src_format}, {face_r_dim}, TILE_C_DIM, {num_faces}, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */\n"
            f"    );\n"
        )

    def pack(
        self,
        pack_node: PackNode,
        operation: FusedOperation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        dest_acc = config.dest_acc.cpp_enum_value
        dest_sync = f"DstSync::Sync{operation.dest_sync.name}"
        buffer = pack_node.output.cpp_name
        return f"_llk_pack_<{dest_sync}, {dest_acc}, ckernel::PackMode::Default>({block.tile_id_block}, L1_ADDRESS({buffer}[{block.tile_id_global}]));\n"
