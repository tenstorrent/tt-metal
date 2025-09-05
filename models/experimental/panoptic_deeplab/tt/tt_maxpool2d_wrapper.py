# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import ttnn
from loguru import logger


class SliceMode(Enum):
    """Enumeration for different slicing modes"""

    NONE = "none"
    CHANNEL = "channel"


@dataclass
class SliceConfig:
    """Configuration for manual slicing control"""

    mode: SliceMode = SliceMode.NONE
    num_slices: int = 1

    def __post_init__(self):
        if self.mode != SliceMode.NONE and self.num_slices <= 1:
            raise ValueError(f"{self.mode.value} slicing requires num_slices > 1")


@dataclass
class TtMaxPool2dParameters:
    device: ttnn.MeshDevice
    kernel_size: tuple[int, int]
    stride: tuple[int, int] = None
    padding: tuple[int, int] = (0, 0)
    dilation: tuple[int, int] = (1, 1)
    ceil_mode: bool = False
    slice_config: SliceConfig = None

    def __post_init__(self):
        if self.stride is None:
            self.stride = self.kernel_size
        if self.slice_config is None:
            self.slice_config = SliceConfig()

    @classmethod
    def create(
        cls,
        device: ttnn.MeshDevice,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = None,
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        ceil_mode: bool = False,
        slice_config: SliceConfig | None = None,
    ) -> TtMaxPool2dParameters:
        if slice_config is None:
            slice_config = SliceConfig()

        return cls(
            device=device,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            slice_config=slice_config,
        )


class TtMaxPool2d:
    """
    TTNN MaxPool2d wrapper with channel slicing support.

    Usage examples:

    # Create parameters with channel slicing
    slice_config = SliceConfig(mode=SliceMode.CHANNEL, num_slices=4)
    params = TtMaxPool2dParameters.create(
        device=device,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        slice_config=slice_config
    )
    maxpool = TtMaxPool2d(params)

    # Using convenience methods
    maxpool = TtMaxPool2d.create_with_channel_slicing(
        device=device,
        kernel_size=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        num_slices=4
    )

    # No slicing
    maxpool = TtMaxPool2d.create(device=device, kernel_size=(3, 3), stride=(2, 2))
    """

    def __init__(self, parameters: TtMaxPool2dParameters) -> None:
        self._device = parameters.device
        self._kernel_size = parameters.kernel_size
        self._stride = parameters.stride
        self._padding = parameters.padding
        self._dilation = parameters.dilation
        self._ceil_mode = parameters.ceil_mode
        self._slice_config = parameters.slice_config

    def _perform_channel_slicing(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Perform channel slicing maxpool"""
        batch_size, input_h, input_w, channels = x.shape

        logger.trace(
            f"TtMaxPool2d channel slicing - input shape: {x.shape}, slices: {self._slice_config.num_slices}, kernel: {self._kernel_size}, stride: {self._stride}, padding: {self._padding}"
        )

        assert (
            channels % self._slice_config.num_slices == 0
        ), f"Input channels ({channels}) must be divisible by num_slices ({self._slice_config.num_slices})"

        slice_channels = channels // self._slice_config.num_slices

        # Calculate output dimensions
        output_h = (input_h + 2 * self._padding[0] - self._kernel_size[0]) // self._stride[0] + 1
        output_w = (input_w + 2 * self._padding[1] - self._kernel_size[1]) // self._stride[1] + 1

        output_slices = []

        for i in range(self._slice_config.num_slices):
            logger.trace(
                f"TtMaxPool2d processing channel slice {i+1}/{self._slice_config.num_slices}, slice channels: {slice_channels}"
            )
            start_idx = i * slice_channels
            end_idx = (i + 1) * slice_channels

            # Slice input along channel dimension
            x_slice = ttnn.slice(x, [0, 0, 0, start_idx], [batch_size, input_h, input_w, end_idx])

            # Reshape for max_pool2d
            x_slice_reshaped = ttnn.reshape(x_slice, (1, 1, batch_size * input_h * input_w, slice_channels))
            ttnn.deallocate(x_slice)

            # Apply max_pool2d to slice
            x_slice_pooled = ttnn.max_pool2d(
                x_slice_reshaped,
                batch_size=batch_size,
                input_h=input_h,
                input_w=input_w,
                channels=slice_channels,
                kernel_size=list(self._kernel_size),
                stride=list(self._stride),
                padding=list(self._padding),
                dilation=list(self._dilation),
                ceil_mode=self._ceil_mode,
            )
            ttnn.deallocate(x_slice_reshaped)

            # Reshape back to output dimensions
            x_slice_output = ttnn.reshape(x_slice_pooled, (batch_size, output_h, output_w, slice_channels))
            ttnn.deallocate(x_slice_pooled)
            x_slice_output = ttnn.to_memory_config(x_slice_output, ttnn.DRAM_MEMORY_CONFIG)
            output_slices.append(x_slice_output)

        # Concatenate slices along channel dimension
        x_pooled = ttnn.concat(output_slices, dim=3)
        logger.trace(
            f"TtMaxPool2d channel slicing complete - output shape: {x_pooled.shape}, memory_config: {x_pooled.memory_config()}"
        )

        # Deallocate slice tensors
        for slice_tensor in output_slices:
            ttnn.deallocate(slice_tensor)

        return x_pooled

    def _perform_standard_maxpool(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Perform standard maxpool without slicing"""
        batch_size, input_h, input_w, channels = x.shape

        logger.trace(
            f"TtMaxPool2d standard pooling - input shape: {x.shape}, kernel: {self._kernel_size}, stride: {self._stride}, padding: {self._padding}"
        )

        # Reshape for max_pool2d
        x_reshaped = ttnn.reshape(x, (1, 1, batch_size * input_h * input_w, channels))
        x_pooled = ttnn.max_pool2d(
            x_reshaped,
            batch_size=batch_size,
            input_h=input_h,
            input_w=input_w,
            channels=channels,
            kernel_size=list(self._kernel_size),
            stride=list(self._stride),
            padding=list(self._padding),
            dilation=list(self._dilation),
            ceil_mode=self._ceil_mode,
        )
        ttnn.deallocate(x_reshaped)

        # Calculate output dimensions
        output_h = (input_h + 2 * self._padding[0] - self._kernel_size[0]) // self._stride[0] + 1
        output_w = (input_w + 2 * self._padding[1] - self._kernel_size[1]) // self._stride[1] + 1
        x_pooled = ttnn.reshape(x_pooled, (batch_size, output_h, output_w, channels))
        logger.trace(
            f"TtMaxPool2d standard pooling complete - output shape: {x_pooled.shape}, memory_config: {x_pooled.memory_config()}"
        )
        return x_pooled

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Apply maxpool operation with optional channel slicing"""
        batch_size, input_h, input_w, channels = x.shape

        # Check if channels are divisible by slice factor for channel slicing
        if self._slice_config.mode == SliceMode.CHANNEL and channels % self._slice_config.num_slices == 0:
            return self._perform_channel_slicing(x)
        else:
            if self._slice_config.mode == SliceMode.CHANNEL:
                print(
                    f"Warning: Channel slicing requested but channels ({channels}) not divisible by {self._slice_config.num_slices}. Using standard maxpool."
                )
            return self._perform_standard_maxpool(x)

    @property
    def kernel_size(self) -> tuple[int, int]:
        return self._kernel_size

    @property
    def stride(self) -> tuple[int, int]:
        return self._stride

    @property
    def padding(self) -> tuple[int, int]:
        return self._padding

    @classmethod
    def create_with_channel_slicing(
        cls,
        device: ttnn.MeshDevice,
        kernel_size: tuple[int, int],
        num_slices: int,
        stride: tuple[int, int] = None,
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        ceil_mode: bool = False,
    ) -> "TtMaxPool2d":
        """Create TtMaxPool2d with channel slicing configuration."""
        slice_config = SliceConfig(mode=SliceMode.CHANNEL, num_slices=num_slices)
        parameters = TtMaxPool2dParameters.create(
            device=device,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            slice_config=slice_config,
        )
        return cls(parameters)

    @classmethod
    def create(
        cls,
        device: ttnn.MeshDevice,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = None,
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        ceil_mode: bool = False,
    ) -> "TtMaxPool2d":
        """Create TtMaxPool2d without any slicing."""
        slice_config = SliceConfig(mode=SliceMode.NONE)
        parameters = TtMaxPool2dParameters.create(
            device=device,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            slice_config=slice_config,
        )
        return cls(parameters)
