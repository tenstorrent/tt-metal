# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.base_packer import Packer as BasePacker
from fuser.block_data import BlockData
from fuser.fuser_config import GlobalConfig
from fuser.golden.pack.pack import pack_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from fuser.pack_node import PackNode


class Packer(BasePacker):
    granularity = InvocationGranularity.TILE
    golden_fn = staticmethod(pack_golden)

    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_common.h",
        ]

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
        return f"_llk_pack_<{dest_sync}, {dest_acc}, ckernel::PackMode::Default>({block.tile_id_dest}, L1_ADDRESS({buffer}[{block.tile_id_out}]));\n"
