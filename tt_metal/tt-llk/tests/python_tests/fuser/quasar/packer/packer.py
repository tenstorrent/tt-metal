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
from fuser.operand import BfdResource
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
        tensor_shape = pack_node.output.tile_shape.cpp_value
        return (
            "{\n"
            + pack_node.output.bfd_alloc_and_program(
                BfdResource.PACK0, result_name="bfd_id"
            )
            + f"_llk_pack_init_(bfd_id, {tensor_shape}, 1);\n"
            + "}\n"
        )

    def pack(
        self,
        pack_node: PackNode,
        operation: L1Operation,
        config: GlobalConfig,
        block: BlockData,
    ) -> str:
        tensor_shape = pack_node.output.tile_shape.cpp_value
        return (
            f"_llk_pack_({block.tile_id_dest}, {block.tile_id_out}, {tensor_shape});\n"
        )
