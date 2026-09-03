# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig

from .base_sfpu import Sfpu
from .block_data import BlockData, KernelInvocation


class SfpuNode:
    def __init__(
        self, sfpu: Sfpu, blocks: Optional[List[Tuple[KernelInvocation, ...]]] = None
    ):
        self.sfpu = sfpu
        self.blocks = blocks

    def automatic_call(self, call: KernelInvocation, block: BlockData, unpack=False):
        if self.sfpu.input_count == 2:
            return KernelInvocation(src0=0, src1=0, dest=0)
        return KernelInvocation(dest=0)

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
    ):
        if config.skip_math_init:
            return ""
        block.tile_id_src_a = call.src0
        block.tile_id_src_b = call.src1
        block.tile_id_block = call.dest
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

    def golden_call(
        self,
        call: KernelInvocation,
        tensor_dst: torch.Tensor,
        operation: "L1Operation",
        config: "GlobalConfig",
        master=False,
    ) -> torch.Tensor:
        tile_dims = (
            operation.tile_shape.total_row_dim(),
            operation.tile_shape.total_col_dim(),
        )
        tile_size = operation.tile_shape.total_tile_size()
        if call.src0 is None:
            result = self.sfpu.golden(
                tensor_dst[call.dest].clone(),
                operation,
                config,
                self,
                tile_dims,
                1,
            )
            tensor_dst[call.dest] = result.view(tile_size)
            return tensor_dst

        work = torch.stack(
            (
                tensor_dst[call.src0].clone(),
                tensor_dst[call.src1].clone(),
                tensor_dst[call.dest].clone(),
            )
        )
        result = self.sfpu.golden(
            work.flatten(),
            operation,
            config,
            self,
            (tile_dims[0] * 3, tile_dims[1]),
            3,
        )
        tensor_dst[call.dest] = result.view(3, tile_size)[2]
        return tensor_dst

    def get_headers(self):
        return self.sfpu.get_headers()

    def __str__(self):
        return f"{self.sfpu}"
