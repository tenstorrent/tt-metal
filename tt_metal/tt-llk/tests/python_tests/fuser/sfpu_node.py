# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig

from .base_sfpu import Sfpu
from .block_data import BlockData
from .indexing import KernelInvocation


class SfpuNode:
    block_tiles_x = None
    block_tiles_y = None

    def __init__(self, sfpu: Sfpu, loop_spec=None):
        self.sfpu = sfpu
        self.loop_spec = loop_spec

    def sfpu_init(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if config.skip_math_init:
            return ""
        return self.sfpu.init(operation, config, self, block)

    def sfpu_call(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
        call: KernelInvocation,
    ) -> str:
        if config.skip_math_init:
            return ""
        block.dest_src0 = call.src0
        block.dest_src1 = call.src1
        block.tile_id_dest = call.dest
        return self.sfpu.calculate(operation, config, self, block)

    def sfpu_uninit(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if config.skip_math_init:
            return ""
        return self.sfpu.uninit(operation, config, self, block)

    def get_headers(self):
        return self.sfpu.get_headers()

    def __str__(self):
        return f"{self.sfpu}"
