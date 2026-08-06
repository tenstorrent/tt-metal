# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from fuser.block_data import BlockData
from fuser.fused_loop import FusedLoop, LoopBlockRow
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig
from fuser.pack_node import PackNode
from helpers.llk_params import PackerReluType

from .common import untilize_l1_address
from .packer import Packer


class PackUntilize(Packer):
    loop: FusedLoop = LoopBlockRow()
    per_block_init = True
    pack_mode = "PackMode::Untilize"

    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_common.h",
            "llk_pack_untilize.h",
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

        return self._untilize_golden(tensor, pack_node)

    def init(
        self,
        pack_node: PackNode,
        operation: FusedOperation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        block_ct_dim = block.block_tiles_x
        full_ct_dim = pack_node.output.tile_count_x
        face_r_dim = pack_node.output.tile_shape.face_r_dim
        num_faces = pack_node.output.tile_shape.total_num_faces()

        return (
            f"_llk_pack_untilize_init_<{block_ct_dim}, {full_ct_dim}>(\n"
            f"    {config.sentinel.pack_dst_format}, {face_r_dim}, {num_faces}\n"
            f");\n"
        )

    def pack(
        self,
        pack_node: PackNode,
        operation: FusedOperation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        block_ct_dim = block.block_tiles_x
        full_ct_dim = pack_node.output.tile_count_x
        face_r_dim = pack_node.output.tile_shape.face_r_dim

        return (
            f"_llk_pack_untilize_<{block_ct_dim}, {full_ct_dim}>(\n"
            f"    {untilize_l1_address(pack_node.output, block)},\n"
            f"    {config.sentinel.pack_dst_format}, {face_r_dim}, {block.tile_id_block}\n"
            f");\n"
        )

    def uninit(
        self,
        pack_node: PackNode,
        operation: FusedOperation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        return "_llk_pack_untilize_uninit_();\n"
