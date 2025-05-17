# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

if TYPE_CHECKING:
    import torch

SLICE_COUNT_FACTOR = {
    (16, 512, 128 * 128): 1024 // 2,
    (20, 32, 128 * 256): 1024 // 16,
    (32, 32, 64 * 64): 1024 // 128,
    (128, 3, 256 * 256): 1024 * 4,  # hangs with lower slice_count
    (128, 3, 512 * 512): 1024 * 16,  # hangs with lower slice_count
    (128, 3, 1024 * 1024): 1024 * 64,  # hangs with lower slice_count
    (128, 128, 512 * 512): 1024 * 4,
    (128, 128, 1024 * 1024): 1024 * 16,
    (256, 128, 256 * 256): 1024 * 2,
    (256, 128, 512 * 512): 1024 * 8,
    (256, 128, 1024 * 1024): 1024 * 32,
    (256, 256, 512 * 512): 1024 * 8,
    (256, 256, 1024 * 1024): 1024 * 32,
    (512, 256, 256 * 256): 1024 * 4,
    (512, 256, 512 * 512): 1024 * 16,
    (512, 512, 128 * 128): 1024 // 2,
    (512, 512, 256 * 256): 1024 * 2,
    (512, 512, 512 * 512): 1024 * 8,
}


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
    ) -> None:
        self._stride = stride
        self._padding = padding

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
        memory_config: ttnn.MemoryConfig | None = None,
    ) -> tuple[ttnn.Tensor, list[int]]:
        (batch_size, height, width, _) = x.shape

        k = SLICE_COUNT_FACTOR.get((self._in_channels, self._out_channels, height * width), 1)
        slice_count = -(-k * batch_size // 1024)

        if slice_count > 1:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        x, [output_height, output_width], [prepared_weight, prepared_bias] = ttnn.conv2d(
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
            input_height=height,
            input_width=width,
            return_output_dim=True,
            return_weights_and_bias=True,
            conv_config=conv_config,
            memory_config=memory_config,
            slice_config=(
                ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceWidth, num_slices=slice_count)
                if slice_count > 1
                else None
            ),
        )

        if slice_count > 1:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        self._weight = prepared_weight
        self._bias = prepared_bias

        shape = [batch_size, output_height, output_width, self._out_channels]
        return x, shape

    def __call__(
        self,
        x: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig | None = None,
    ) -> ttnn.Tensor:
        x, shape = self.call_without_reshape(x, memory_config=memory_config)
        return x.reshape(shape)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def kernel_size(self) -> tuple[int, int]:
        return self._kernel_size
