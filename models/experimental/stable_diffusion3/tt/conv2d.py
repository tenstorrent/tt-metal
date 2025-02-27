# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    import torch


@dataclass
class TtConv2dParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        dtype: ttnn.DataType | None = None,
    ) -> TtConv2dParameters:
        return cls(
            weight=ttnn.from_torch(state["weight"], dtype=dtype),
            bias=ttnn.from_torch(state["bias"].reshape((1, 1, 1, -1)), dtype=dtype) if "bias" in state else None,
        )

    @property
    def in_channels(self) -> int:
        return self.weight.shape[1]

    @property
    def out_channels(self) -> int:
        return self.weight.shape[0]

    @property
    def kernel_size(self) -> tuple[int, int]:
        return self.weight.shape[-2], self.weight.shape[-1]


class TtConv2d:
    def __init__(
        self,
        parameters: TtConv2dParameters,
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
    ) -> None:
        self._stride = stride
        self._padding = padding

        self._in_channels = parameters.in_channels
        self._out_channels = parameters.out_channels
        self._kernel_size = parameters.kernel_size

        self._weight = parameters.weight
        self._bias = parameters.bias

    def call_without_reshape(
        self, x: ttnn.Tensor, *, conv_config: ttnn.Conv2dConfig | None = None
    ) -> tuple[ttnn.Tensor, list[int]]:
        batch_size = x.shape[0]
        device = x.device()
        memory_config_in = ttnn.get_memory_config(x)

        result, [output_height, output_width], [prepared_weight, prepared_bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self._weight,
            bias_tensor=self._bias,
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            device=device,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
            batch_size=batch_size,
            input_height=x.shape[1],
            input_width=x.shape[2],
            return_output_dim=True,
            return_weights_and_bias=True,
            conv_config=conv_config,
        )

        result = ttnn.to_memory_config(result, memory_config=memory_config_in)

        self._weight = prepared_weight
        self._bias = prepared_bias

        shape = [batch_size, output_height, output_width, self._out_channels]
        return result, shape

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        result, shape = self.call_without_reshape(x)
        # TODO: deallocate result
        return result.reshape(shape)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def kernel_size(self) -> tuple[int, int]:
        return self._kernel_size
