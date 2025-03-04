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
            bias = state["bias"].unsqueeze(0)
        else:
            bias = None

        return cls(
            weight=from_torch(
                state["weight"].transpose(0, 1),
                dtype=dtype,
                mesh_device=device,
                shard_dim=-1,
            ),
            bias=from_torch(
                bias,
                dtype=dtype,
                mesh_device=device,
                shard_dim=-1,
            )
            if bias is not None
            else None,
        )

    @classmethod
    def from_torch_col_parallel(
        cls,
        state: dict[str, torch.Tensor],
        *,
        n_local_heads: int,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> TtLinearParameters:
        if "bias" in state:
            torch_bias = state["bias"].unsqueeze(0)
        else:
            torch_bias = None

        def shuffle_heads(tensor):
            # Given torch tensor with output features in the last dimension,
            # shuffle heads to allow for column parallel computation
            in_dim = tensor.shape[0]
            tensor = tensor.reshape(in_dim, 3, device.get_num_devices(), n_local_heads, -1)  # [ID, 3, ND, NLH, DH]
            tensor = tensor.permute(0, 2, 1, 3, 4)  # [ID, ND, 3, NLH, DH]
            tensor = tensor.reshape(in_dim, -1)  # [ID, ND*3*NLH*DH]
            return tensor

        torch_weight = state["weight"].transpose(0, 1)
        return cls(
            weight=from_torch(
                shuffle_heads(torch_weight),
                dtype=dtype,
                mesh_device=device,
                shard_dim=-1,
            ),
            bias=from_torch(
                shuffle_heads(torch_bias),
                dtype=dtype,
                mesh_device=device,
                shard_dim=-1,
            )
            if torch_bias is not None
            else None,
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
