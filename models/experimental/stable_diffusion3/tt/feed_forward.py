# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .linear import TtLinear, TtLinearParameters
from .substate import substate

if TYPE_CHECKING:
    import torch


@dataclass
class TtFeedForwardParameters:
    in_proj: TtLinearParameters
    out_proj: TtLinearParameters

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtFeedForwardParameters:
        return cls(
            in_proj=TtLinearParameters.from_torch(
                substate(state, "net.0.proj"), dtype=dtype, device=device, shard_dim=-1
            ),
            out_proj=TtLinearParameters.from_torch(substate(state, "net.2"), dtype=dtype, device=device, shard_dim=-2),
        )


class TtFeedForward:
    def __init__(self, parameters: TtFeedForwardParameters) -> None:
        super().__init__()

        self.in_proj = TtLinear(parameters.in_proj)
        self.out_proj = TtLinear(parameters.out_proj)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x2 = self.in_proj(x)
        # Turning on fast_and_approximate_mode leads to big changes in the generated image.
        # The image quality might still be okay.
        x3 = ttnn.gelu(x2, fast_and_approximate_mode=False)
        ttnn.deallocate(x2)

        result = self.out_proj(x3)
        ttnn.deallocate(x3)

        result = ttnn.experimental.all_reduce(
            result,
            math_op=ttnn.ReduceType.Sum,
            num_links=1,
            memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
            topology=ttnn.Topology.Ring,
        )

        return result
