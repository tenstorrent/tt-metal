# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from .utils import from_torch

if TYPE_CHECKING:
    import torch


@dataclass
class TtLinearParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtLinearParameters:
        if "bias" in state:
            orig_bias = state["bias"]
            bias = state["bias"].unsqueeze(0)
        else:
            bias = None

        return cls(
            weight=from_torch(
                state["weight"].transpose(0, 1),
                dtype=dtype,
                mesh_device=device,
            ),
            bias=from_torch(bias, dtype=dtype, mesh_device=device) if bias is not None else None,
        )

    @property
    def in_channels(self) -> int:
        return self.weight.shape[0]

    @property
    def out_channels(self) -> int:
        return self.weight.shape[1]


class TtLinear:
    def __init__(self, parameters: TtLinearParameters) -> None:
        self._in_channels = parameters.in_channels
        self._weight = parameters.weight
        self._bias = parameters.bias

    def __call__(
        self,
        x: ttnn.Tensor,
        *,
        memory_config: ttnn.MemoryConfig | None = None,
        program_config: ttnn.MatmulProgramConfig | None = None,
        core_grid: ttnn.CoreGrid | None = None,
        output_tile: list[int] | None = None,
        dtype: ttnn.DataType | None = None,
        deallocate: bool = False,
    ) -> ttnn.Tensor:
        assert x.shape[-1] == self._in_channels, "input tensor does not have the expected shape"

        weight = self._weight
        bias = self._bias

        output = ttnn.linear(
            x,
            weight,
            bias=bias,
            memory_config=memory_config,
            program_config=program_config,
            core_grid=core_grid,
            output_tile=output_tile,
            dtype=dtype,
        )

        if deallocate:
            ttnn.deallocate(x)

        return output

    @property
    def device(self) -> ttnn.Device:
        return self._weight.device()
