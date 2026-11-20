# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.block_data import BlockData
from fuser.fuser_config import GlobalConfig
from fuser.golden.pack.untilize import untilize_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from fuser.pack_node import PackNode
from helpers.tile_shape import cpp_tensor_shape

from .common import untilize_l1_address
from .packer import Packer


class PackUntilize(Packer):
    granularity = InvocationGranularity.ROW

    golden_fn = staticmethod(untilize_golden)

    per_block_init = True
    pack_mode = "PackMode::Untilize"
    requires_dest_remap = True

    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_common.h",
            "llk_pack_untilize.h",
        ]

    def init(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        block_ct_dim = block.block_cols
        full_ct_dim = pack_node.output.tile_count_x

        return (
            f"_llk_pack_untilize_init_<{block_ct_dim}, {full_ct_dim}>(\n"
            f"    {config.sentinel.pack_src_format}, {config.sentinel.pack_dst_format}, {cpp_tensor_shape(pack_node.output.tile_shape)}\n"
            f");\n"
        )

    def pack(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        block_ct_dim = block.block_cols
        full_ct_dim = pack_node.output.tile_count_x

        return (
            f"_llk_pack_untilize_<{block_ct_dim}, {full_ct_dim}>(\n"
            f"    {untilize_l1_address(pack_node.output, block)},\n"
            f"    {cpp_tensor_shape(pack_node.output.tile_shape)}, {block.tile_id_dest}\n"
            f");\n"
        )

    def uninit(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        return f"_llk_pack_untilize_uninit_({config.sentinel.pack_src_format});\n"
