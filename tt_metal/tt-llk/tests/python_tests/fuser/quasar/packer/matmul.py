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
        tile_count_x = pack_node.output.tile_count_x
        row_wise = tile_count_x % subblock_c_dim != 0
        if row_wise:
            subblock_r_dim = 1
        num_subblocks_c_dim = 1 if row_wise else tile_count_x // subblock_c_dim
        tensor_shape = pack_node.output.tile_shape.cpp_value
        return (
            pack_node.output.bfd_alloc_and_program(BfdResource.PACK0)
            + f"_llk_pack_matmul_init_({bfd_current(BfdResource.PACK0)}, "
            f"{subblock_r_dim}, {subblock_c_dim}, {num_subblocks_c_dim}, {tensor_shape});\n"
        )

    def pack(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        tile_count_x = pack_node.output.tile_count_x
        tensor_shape = pack_node.output.tile_shape.cpp_value
        if tile_count_x % block.block_cols == 0:
            return f"_llk_pack_matmul_({block.tile_id_dest}, {block.tile_id_out}, {tensor_shape});\n"
        return (
            f"for (std::uint32_t pack_row = 0; pack_row < {block.block_rows}; pack_row++) {{\n"
            f"    _llk_pack_matmul_(({block.tile_id_dest}) + pack_row * {block.block_cols}, "
            f"({block.tile_id_out}) + pack_row * {tile_count_x}, {tensor_shape});\n"
            f"}}\n"
        )
