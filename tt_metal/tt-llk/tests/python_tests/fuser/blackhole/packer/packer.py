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
from helpers.tile_shape import cpp_tensor_shape


class Packer(BasePacker):
    granularity = InvocationGranularity.TILE
    golden_fn = staticmethod(pack_golden)

    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_common.h",
            "perf.h",
        ]

    def init(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        bh_pack_mode = operation.bh_tilize.pack_mode_value
        return (
            f"    _llk_pack_init_<{bh_pack_mode}, false /* zero_output */, false /* skip_addrmod_config */>(\n"
            f"        {config.sentinel.pack_src_format}, {cpp_tensor_shape(pack_node.output.tile_shape)}, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */\n"
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
