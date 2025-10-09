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
    ) -> "TtMaxPool2dParameters":
        return cls(
            device=device,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            slice_config=slice_config or SliceConfig(),
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
            f"Channel slicing maxpool - input: {x.shape}, slices: {self._slice_config.num_slices}, "
            f"kernel: {self._kernel_size}, stride: {self._stride}"
        )

        if channels % self._slice_config.num_slices != 0:
            raise ValueError(
                f"Input channels ({channels}) must be divisible by num_slices ({self._slice_config.num_slices})"
            )

        slice_channels = channels // self._slice_config.num_slices
        output_h, output_w = self._calculate_output_dims(input_h, input_w)
        output_slices = []

        try:
            for i in range(self._slice_config.num_slices):
                logger.trace(f"Processing slice {i + 1}/{self._slice_config.num_slices}")

                start_idx = i * slice_channels
                end_idx = (i + 1) * slice_channels

                # Process slice
                x_slice = ttnn.slice(x, [0, 0, 0, start_idx], [batch_size, input_h, input_w, end_idx])
                x_slice_output = self._apply_maxpool_to_slice(
                    x_slice, batch_size, input_h, input_w, slice_channels, output_h, output_w
                )
                output_slices.append(x_slice_output)

            # Concatenate slices
            result = ttnn.concat(output_slices, dim=3)
            logger.trace(f"Channel slicing complete - output: {result.shape}")

            return result
        finally:
            # Cleanup
            for slice_tensor in output_slices:
                if slice_tensor is not None:
                    ttnn.deallocate(slice_tensor)

    def _apply_maxpool_to_slice(self, x_slice, batch_size, input_h, input_w, slice_channels, output_h, output_w):
        """Apply maxpool to a single channel slice"""
        # Reshape for max_pool2d
        x_slice_reshaped = ttnn.reshape(x_slice, (1, 1, batch_size * input_h * input_w, slice_channels))
        ttnn.deallocate(x_slice)

        # Apply max_pool2d
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

        # Reshape back
        x_slice_output = ttnn.reshape(x_slice_pooled, (batch_size, output_h, output_w, slice_channels))
        ttnn.deallocate(x_slice_pooled)

        return ttnn.to_memory_config(x_slice_output, ttnn.DRAM_MEMORY_CONFIG)

    def _perform_standard_maxpool(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Perform standard maxpool without slicing"""
        batch_size, input_h, input_w, channels = x.shape

        logger.trace(f"Standard maxpool - input: {x.shape}, kernel: {self._kernel_size}, stride: {self._stride}")

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

        # Reshape back to output dimensions
        output_h, output_w = self._calculate_output_dims(input_h, input_w)
        result = ttnn.reshape(x_pooled, (batch_size, output_h, output_w, channels))
        logger.trace(f"Standard maxpool complete - output: {result.shape}")

        return result

    def _calculate_output_dims(self, input_h: int, input_w: int) -> tuple[int, int]:
        """Calculate output dimensions for maxpool operation"""
        output_h = (input_h + 2 * self._padding[0] - self._kernel_size[0]) // self._stride[0] + 1
        output_w = (input_w + 2 * self._padding[1] - self._kernel_size[1]) // self._stride[1] + 1
        return output_h, output_w

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Apply maxpool operation with optional channel slicing"""
        if self._should_use_channel_slicing(x):
            return self._perform_channel_slicing(x)
        else:
            return self._perform_standard_maxpool(x)

    def _should_use_channel_slicing(self, x: ttnn.Tensor) -> bool:
        """Determine if channel slicing should be used"""
        if self._slice_config.mode != SliceMode.CHANNEL:
            return False

        _, _, _, channels = x.shape

        if channels % self._slice_config.num_slices != 0:
            logger.warning(
                f"Channel slicing requested but channels ({channels}) not divisible by "
                f"{self._slice_config.num_slices}. Using standard maxpool."
            )
            return False

        return True

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
        parameters = TtMaxPool2dParameters.create(
            device=device,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )
        return cls(parameters)
