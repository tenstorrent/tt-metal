# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig

from helpers.golden_generators import PackGolden
from helpers.llk_params import L1Accumulation, PackerReluType

from .block_data import BlockData
from .fused_operand import Operand
from .fused_packer import Packer


class PackNode:
    """Wraps a packer with its output operand and pack settings.

    Analogous to ComputeNode on the math side. Each PackNode represents
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

    def _relu_config(self, config: "GlobalConfig") -> str:
        pack_src_format = config.sentinel._pack_src

        relu_config = PackGolden.generate_relu_config(
            self.pack_relu, self.relu_threshold, pack_src_format
        )
        return f"_llk_pack_relu_config_(ReluConfig::from_packed({relu_config}));\n"

    def _l1_accumulation_config(self) -> str:
        l1_acc = self.pack_l1_accumulation.cpp_enum_value
        return f"_llk_pack_reconfig_l1_acc_({l1_acc});\n"

    def reconfig(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> str:
        return config.sentinel.configure_pack(config, operation, self)

    def configure(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ) -> str:
        code = self.packer.init(self, operation, config, block)
        code += self._relu_config(config)
        code += self._l1_accumulation_config()
        return code

    def pack_loop(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ) -> str:
        return self.packer.loop.pack_loop(operation, config, self, block)

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> str:
        return self.packer.uninit(self, operation, config, None)

    def golden(
        self,
        tensor: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        return self.packer.golden(tensor, self, operation, config)

    def get_headers(self) -> List[str]:
        return self.packer.get_headers()

    def __str__(self):
        return f"PackNode({self.packer}, output={self.output})"
