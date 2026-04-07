# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

import torch

from .chip_architecture import ChipArchitecture

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig
    from .block_data import BlockData
    from .fused_math import ComputeNode

from .fused_loop import FusedLoop


class Packer:
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
        stage = operation.stage_id
        dest_acc = config.dest_acc.cpp_enum_value
        bh_tilize = operation.bh_tilize.cpp_enum_value
        face_r_dim = operation.face_r_dim
        num_faces = operation.num_faces
        dest_sync = f"DstSync::Sync{operation.dest_sync.name}"
        if config.architecture == ChipArchitecture.BLACKHOLE:
            code = (
                f"    _llk_pack_init_<false, false, {bh_tilize}>(\n"
                f"        pack_dst_format{stage}, pack_dst_format{stage}, {face_r_dim}, TILE_C_DIM, {num_faces}, false, false\n"
                f"    );\n"
                f"    _llk_pack_dest_init_<{dest_sync}, {dest_acc}>();\n"
            )
        elif config.architecture == ChipArchitecture.WORMHOLE:
            code = (
                f"    _llk_pack_init_<false, false>(\n"
                f"        pack_dst_format{stage}, {face_r_dim}, {num_faces}\n"
                f"    );\n"
                f"    _llk_pack_dest_init_<{dest_sync}, {dest_acc}, false>();\n"
            )
        else:
            raise ValueError("Unsupported architecture for packer")

        return code

    def pack(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        block: "BlockData",
    ) -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.cpp_enum_value
        dest_sync = f"DstSync::Sync{operation.dest_sync.name}"
        return f"_llk_pack_<{dest_sync}, {dest_acc}, false>({block.tile_id_block}, L1_ADDRESS(buffer_Res{stage}[{block.tile_id_global}]));\n"

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        block: "BlockData",
    ) -> str:
        return ""
