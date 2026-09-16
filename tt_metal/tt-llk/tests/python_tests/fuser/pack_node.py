# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig

from helpers.golden_generators import ReduceGolden
from helpers.llk_params import (
    L1Accumulation,
    PackerReluType,
    ReduceDimension,
    ReducePool,
)

from .arch_common import pack_common
from .base_packer import Packer
from .block_data import BlockData
from .operand import Operand


class PackNode:
    """Wraps a packer with its output operand and pack settings.

    Analogous to FpuNode on the math side. Each PackNode represents
    one pack destination within an operation. Multiple PackNodes allow a
    single math result to be packed to different output buffers with
    independent relu or L1 accumulation configs.
    """

    def __init__(
        self,
        packer: Packer,
        output: Operand,
        pack_relu: PackerReluType = PackerReluType.NoRelu,
        relu_threshold: float = 0.0,
        pack_l1_accumulation: L1Accumulation = L1Accumulation.No,
    ):
        self.packer = packer
        self.output = output
        self.pack_relu = pack_relu
        self.relu_threshold = relu_threshold
        self.pack_l1_accumulation = pack_l1_accumulation

    def init(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ) -> str:
        code = self.packer.init(self, operation, config, block)
        code += pack_common.relu_config(config, operation, self)
        code += pack_common.l1_accumulation_config(config, operation, self)
        return code

    def pack_loop(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        block: BlockData,
    ) -> str:
        return self.packer.loop.pack_loop(operation, config, self, block)

    def uninit(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> str:
        return self.packer.uninit(self, operation, config, None)

    def golden(
        self,
        tensor: torch.Tensor,
        operation: "L1Operation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        if operation.reduce_pool == ReducePool.Max:
            # Reapply the edge mask after any fused math/SFPU operations and before pack transforms.
            mask = torch.ones_like(tensor, dtype=torch.bool)
            tile_rows, tile_cols = operation.tile_shape.tile_dims
            if operation.reduce_dim == ReduceDimension.Row:
                mask[:, ::tile_cols] = False
            elif operation.reduce_dim == ReduceDimension.Column:
                mask[::tile_rows, :] = False
            else:
                mask[::tile_rows, ::tile_cols] = False
            tensor = tensor.masked_fill(
                mask,
                ReduceGolden.padding_value(ReducePool.Max, self.output.data_format),
            )
        return self.packer.golden(tensor, self, operation, config)

    def get_headers(self) -> List[str]:
        return self.packer.get_headers()

    def __str__(self):
        return f"PackNode({self.packer}, output={self.output})"
