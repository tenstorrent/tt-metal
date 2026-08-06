# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from fuser.block_data import BlockData
from fuser.fpu_node import FpuNode
from fuser.fused_loop import FusedLoop, LoopTileByTile
from fuser.fused_operation import FusedOperation
from fuser.fuser_config import GlobalConfig

from .unpack_a import UnpackerA


class TransposeDestUnpacker(UnpackerA):
    loop: FusedLoop = LoopTileByTile()

    def init(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""

    def unpack(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return "_llk_unpack_set_srcB_dummy_valid_();\n"

    def uninit(
        self,
        operation: FusedOperation,
        config: GlobalConfig,
        compute_unit: FpuNode,
        block: BlockData,
    ) -> str:
        return ""
