# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.block_data import BlockData
from fuser.fuser_config import GlobalConfig
from fuser.golden.pack.untilize import untilize_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from fuser.operand import BfdResource, bfd_current
from fuser.pack_node import PackNode

from .packer import Packer


class PackUntilize(Packer):
    granularity = InvocationGranularity.ROW

    golden_fn = staticmethod(untilize_golden)

    per_block_init = True

    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_untilize.h",
        ]

    def init(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        full_ct_dim = pack_node.output.tile_count_x
        block_ct_dim = block.block_cols
        tensor_shape = pack_node.output.tile_shape.cpp_value

        return (
            pack_node.output.bfd_alloc_and_program(BfdResource.PACK0)
            + f"_llk_pack_untilize_init_<{full_ct_dim}, {block_ct_dim}>"
            f"({bfd_current(BfdResource.PACK0)}, {tensor_shape});\n"
        )

    def pack(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        full_ct_dim = pack_node.output.tile_count_x
        tensor_shape = pack_node.output.tile_shape.cpp_value
        row_stride = full_ct_dim * pack_node.output.tile_shape.total_row_dim()
        tile_row = f"({block.tile_id_out}) / {full_ct_dim}"
        tile_col = f"({block.tile_id_out}) % {full_ct_dim}"
        l1_row_idx = f"{row_stride} * ({tile_row}) + ({tile_col})"

        return (
            f"_llk_pack_untilize_set_dst_offset_({tensor_shape}, {l1_row_idx});\n"
            f"_llk_pack_untilize_({block.tile_id_dest}, 0);\n"
        )
