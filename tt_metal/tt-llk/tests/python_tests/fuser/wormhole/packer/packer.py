# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from fuser.block_data import BlockData
from fuser.fused_loop import FusedLoop
from fuser.fused_math import ComputeNode
from fuser.fused_operation import FusedOperation
from fuser.fused_packer import Packer as BasePacker
from fuser.fuser_config import GlobalConfig


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
        operation: FusedOperation,
        config: GlobalConfig,
    ) -> torch.Tensor:
        return tensor

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.cpp_enum_value
        face_r_dim = operation.face_r_dim
        num_faces = operation.num_faces
        dest_sync = f"DstSync::Sync{operation.dest_sync.name}"
        return (
            f"    _llk_pack_init_<false, false>(\n"
            f"        pack_dst_format{stage}, {face_r_dim}, {num_faces}\n"
            f"    );\n"
            f"    _llk_pack_dest_init_<{dest_sync}, {dest_acc}, false>();\n"
        )

    def pack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: ComputeNode,
        block: BlockData,
    ) -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.cpp_enum_value
        dest_sync = f"DstSync::Sync{operation.dest_sync.name}"
        return f"_llk_pack_<{dest_sync}, {dest_acc}, false>({block.tile_id_block}, L1_ADDRESS(buffer_Res{stage}[{block.tile_id_global}]));\n"
