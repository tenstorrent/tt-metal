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
    device: ttnn.MeshDevice | ttnn.Device

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        device: ttnn.MeshDevice | ttnn.Device,
        dtype: ttnn.DataType | None = None,
    ) -> TtConv2dParameters:
        return cls(
            weight=ttnn.from_torch(state["weight"], dtype=dtype),
            bias=ttnn.from_torch(state["bias"].reshape((1, 1, 1, -1)), dtype=dtype) if "bias" in state else None,
            device=device,
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
    """
    Limitations of DRAM slicing (slice_count > 1), taken from https://github.com/tenstorrent/tt-metal/pull/19686:
    - Only works with activations of dtype BFloat16.
    - No logic to check if preprocessed weights can be safely reused. This is okay if all slices are
      approximately the same size.
    - Slice output is interleaved. This needs an additional interleaved to sharded op.
    - Doesn't support prepared weights.
    """

    def __init__(
        self,
        parameters: TtConv2dParameters,
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        slice_count: int = 1,
    ) -> None:
        self._stride = stride
        self._padding = padding
        self._slice_count = slice_count

        self._in_channels = parameters.in_channels
        self._out_channels = parameters.out_channels
        self._kernel_size = parameters.kernel_size

        self._weight = parameters.weight
        self._bias = parameters.bias
        self._device = parameters.device

    def call_without_reshape(
        self,
        x: ttnn.Tensor,
        *,
        conv_config: ttnn.Conv2dConfig | None = None,
        memory_config: ttnn.MemoryConfig | None,
    ) -> tuple[ttnn.Tensor, list[int]]:
        batch_size = x.shape[0]

        result, [output_height, output_width], [prepared_weight, prepared_bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self._weight,
            bias_tensor=self._bias,
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            device=self._device,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
            batch_size=batch_size,
            input_height=x.shape[1],
            input_width=x.shape[2],
            return_output_dim=True,
            return_weights_and_bias=True,
            conv_config=conv_config,
            memory_config=memory_config,
            slice_config=(
                ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceWidth, num_slices=self._slice_count)
                if self._slice_count > 1
                else None
            ),
        )

        self._weight = prepared_weight
        self._bias = prepared_bias

        shape = [batch_size, output_height, output_width, self._out_channels]
        return result, shape

    def __call__(
        self,
        x: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig | None,
    ) -> ttnn.Tensor:
        result, shape = self.call_without_reshape(x, memory_config=memory_config)
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
