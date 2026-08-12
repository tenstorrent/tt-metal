# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from fuser.base_unpacker import Unpacker
from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fuser_config import GlobalConfig
from fuser.l1_operation import L1Operation
from fuser.tile_loop import LoopBlockRow, LoopTileByTile, TileLoop
from helpers.llk_params import DestSync, EltwiseBinaryReuseDestType


def _unp_sel(compute_unit: FpuNode) -> str:
    if compute_unit.unpack_to_dest.value:
        return "p_unpacr::UNP_DEST"
    if compute_unit.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
        return "p_unpacr::UNP_B"
    return "p_unpacr::UNP_A"


class UnpackerA(Unpacker):
    loop: TileLoop = LoopBlockRow()
    per_block_init = True

    def __init__(
        self, reuse_dest: EltwiseBinaryReuseDestType = EltwiseBinaryReuseDestType.NONE
    ):
        self.reuse_dest = reuse_dest
        if reuse_dest != EltwiseBinaryReuseDestType.NONE:
            self.loop = LoopTileByTile()

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_unary_operand.h",
            "llk_math_common.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor_a = self.transpose_golden(tensor_a, config, operation, compute_unit)

        tensor_a, tensor_b = self.reuse_dest_golden(
            tensor_a, tensor_b, config, operation, compute_unit
        )

        return tensor_a, tensor_b

    def init(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        buf_desc_id = compute_unit.src_a.buf_desc_id
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        en_32bit_dest = config.dest_acc.cpp_enum_value
        unpack_to_dest = compute_unit.unpack_to_dest.cpp_enum_value
        transpose_en = compute_unit.transpose_faces.cpp_enum_value
        unp_sel = _unp_sel(compute_unit)
        num_tiles = (
            1
            if compute_unit.reuse_dest != EltwiseBinaryReuseDestType.NONE
            else block.block_tiles_x
        )

        code = ""
        if compute_unit.unpack_to_dest.value:
            num_sem = 2 if operation.dest_sync == DestSync.Half else 1
            code += f"_llk_sync_init_(semaphore::UNPACK_MATH, {num_sem}, 0);\n"
        code += (
            f"_llk_unpack_unary_operand_init_<{unp_sel}, {transpose_en}, {en_32bit_dest}, {reuse_dest}, {unpack_to_dest}>"
            f"({buf_desc_id}, {tensor_shape}, {num_tiles});\n"
        )
        return code

    def unpack(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        unp_sel = _unp_sel(compute_unit)
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        unpack_to_dest = compute_unit.unpack_to_dest.cpp_enum_value
        dest_sync = operation.dest_sync.cpp_enum_value

        return (
            f"_llk_unpack_unary_operand_<{unp_sel}, {reuse_dest}, {unpack_to_dest}, {dest_sync}>"
            f"({block.tile_id_global}, {tensor_shape});\n"
        )

    def uninit(
        self,
        operation: L1Operation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
