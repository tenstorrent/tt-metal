# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig
    from .block_data import BlockData
    from .compute_node import ComputeNode

from .fused_loop import FusedLoop


class Packer:
    loop: FusedLoop = FusedLoop()

    def get_headers(self) -> List[str]:
        return []

    def golden(
        self,
        tensor: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        return tensor

    def init(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        block: "BlockData",
    ) -> str:
        return ""

    def pack(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        block: "BlockData",
    ) -> str:
        return ""

    def uninit(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        compute_unit: "ComputeNode",
        block: "BlockData",
    ) -> str:
        return ""
