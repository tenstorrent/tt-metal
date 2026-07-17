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
from helpers.llk_params import EltwiseBinaryReuseDestType


class UnpackerA(Unpacker):
    loop: FusedLoop = LoopTileByTile()

    def get_headers(self) -> List[str]:
        return [
            "llk_unpack_common.h",
            "llk_unpack_unary_operand.h",
        ]

    def golden(
        self,
        tensor_a: torch.Tensor,
        tensor_b: torch.Tensor,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if compute_unit.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
            tensor_b = tensor_a
            tensor_a = None

        return tensor_a, tensor_b

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        buf_desc_id = compute_unit.src_a.buf_desc_id
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value
        en_32bit_dest = config.dest_acc.cpp_enum_value

        return (
            f"_llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false, {en_32bit_dest}, {reuse_dest}>"
            f"({buf_desc_id}, {tensor_shape}, 1);\n"
        )

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        tensor_shape = compute_unit.src_a.tile_shape.cpp_value
        reuse_dest = compute_unit.reuse_dest.cpp_enum_value

        return (
            f"_llk_unpack_unary_operand_<p_unpacr::UNP_A, {reuse_dest}>"
            f"({block.tile_id_global}, {tensor_shape});\n"
        )

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
