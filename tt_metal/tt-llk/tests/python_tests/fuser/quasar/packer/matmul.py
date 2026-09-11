# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from fuser.block_data import BlockData
from fuser.fuser_config import GlobalConfig
from fuser.golden.pack.matmul import pack_matmul_golden
from fuser.indexing import InvocationGranularity
from fuser.l1_operation import L1Operation
from fuser.operand import BfdResource, bfd_current
from fuser.pack_node import PackNode

from .packer import Packer


class MatmulPacker(Packer):
    granularity = InvocationGranularity.BLOCK
    per_block_init = True
    golden_fn = staticmethod(pack_matmul_golden)

    def get_headers(self) -> List[str]:
        return [
            "llk_pack.h",
            "llk_pack_matmul.h",
        ]

    def init(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        subblock_r_dim = block.block_rows
        subblock_c_dim = block.block_cols
        tile_count_x = (
            operation.max_output_dimensions[1] // operation.tile_shape.total_col_dim()
        )
        num_subblocks_c_dim = tile_count_x // subblock_c_dim
        return (
            pack_node.output.bfd_alloc_and_program(BfdResource.PACK0)
            + f"_llk_pack_matmul_init_({bfd_current(BfdResource.PACK0)}, "
            f"{subblock_r_dim}, {subblock_c_dim}, {num_subblocks_c_dim});\n"
        )

    def pack(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        return f"_llk_pack_matmul_({block.tile_id_dest}, {block.tile_id_out});\n"
