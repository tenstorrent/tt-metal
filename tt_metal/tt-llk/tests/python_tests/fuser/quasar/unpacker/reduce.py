# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fused_loop import FusedLoop, LoopTileByTile
from fuser.fused_operation import FusedOperation
from fuser.fused_unpacker import Unpacker
from fuser.fuser_config import GlobalConfig


class ReduceUnpacker(Unpacker):
    loop: FusedLoop = LoopTileByTile()

    def __init__(self, reduce_dim, reduce_pool):
        self.reduce_dim = reduce_dim
        self.reduce_pool = reduce_pool

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_reduce.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor_a, tensor_b

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        buf_desc_id_a = compute_unit.src_a.buf_desc_id
        buf_desc_id_b = compute_unit.src_b.buf_desc_id
        reduce_dim = self.reduce_dim.cpp_enum_value
        reduce_pool = self.reduce_pool.cpp_enum_value

        return (
            f"_llk_unpack_reduce_init_<{reduce_pool}, {reduce_dim}>"
            f"({buf_desc_id_a}, {buf_desc_id_b}, "
            f"{compute_unit.src_a.tile_shape.cpp_value}, "
            f"1);\n"
        )

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return (
            f"_llk_unpack_reduce_({block.tile_id_global}, {block.tile_id_global}, "
            f"{compute_unit.src_a.tile_shape.cpp_value});\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
