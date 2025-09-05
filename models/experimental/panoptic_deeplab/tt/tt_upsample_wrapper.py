# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Union
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
class TtUpsampleParameters:
    device: ttnn.MeshDevice
    scale_factor: Union[int, tuple[int, int]]
    mode: str = "bilinear"
    slice_config: SliceConfig = None

    def __post_init__(self):
        if self.slice_config is None:
            self.slice_config = SliceConfig()

        # Normalize scale_factor to tuple
        if isinstance(self.scale_factor, int):
            self.scale_factor = (self.scale_factor, self.scale_factor)

        # Validate mode
        if self.mode not in ["bilinear", "nearest"]:
            raise ValueError(f"Mode must be 'bilinear' or 'nearest', got '{self.mode}'")

    @classmethod
    def create(
        cls,
        device: ttnn.MeshDevice,
        scale_factor: Union[int, tuple[int, int]],
        mode: str = "bilinear",
        slice_config: SliceConfig | None = None,
    ) -> TtUpsampleParameters:
        if slice_config is None:
            slice_config = SliceConfig()

        return cls(
            device=device,
            scale_factor=scale_factor,
            mode=mode,
            slice_config=slice_config,
        )


class TtUpsample:
    """
    TTNN Upsample wrapper with channel slicing support.

    Supports bilinear and nearest interpolation modes.

    Usage examples:

    # Create parameters with channel slicing
    slice_config = SliceConfig(mode=SliceMode.CHANNEL, num_slices=2)
    params = TtUpsampleParameters.create(
        device=device,
        scale_factor=(2, 2),
        mode="bilinear",
        slice_config=slice_config
    )
    upsample = TtUpsample(params)

    # Using convenience methods
    upsample = TtUpsample.create_with_channel_slicing(
        device=device,
        scale_factor=(2, 2),
        mode="bilinear",
        num_slices=2
    )

    # No slicing
    upsample = TtUpsample.create(device=device, scale_factor=(2, 2), mode="nearest")
    """

    def __init__(self, parameters: TtUpsampleParameters) -> None:
        self._device = parameters.device
        self._scale_factor = parameters.scale_factor
        self._mode = parameters.mode
        self._slice_config = parameters.slice_config

    def _perform_channel_slicing(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Perform channel slicing upsample"""
        orig_batch, orig_height, orig_width, orig_channels = x.shape

        logger.trace(
            f"TtUpsample channel slicing - input shape: {x.shape}, slices: {self._slice_config.num_slices}, scale_factor: {self._scale_factor}, mode: {self._mode}"
        )

        assert (
            orig_channels % self._slice_config.num_slices == 0
        ), f"Input channels ({orig_channels}) must be divisible by num_slices ({self._slice_config.num_slices})"

        channels_per_slice = orig_channels // self._slice_config.num_slices
        sliced_results = []

        for slice_idx in range(self._slice_config.num_slices):
            logger.trace(
                f"TtUpsample processing channel slice {slice_idx+1}/{self._slice_config.num_slices}, channels_per_slice: {channels_per_slice}"
            )
            start_ch = slice_idx * channels_per_slice
            end_ch = (slice_idx + 1) * channels_per_slice

            # Slice input along channel dimension
            x_slice = ttnn.slice(x, [0, 0, 0, start_ch], [orig_batch, orig_height, orig_width, end_ch])

            # Apply upsample to slice
            x_slice_upsampled = ttnn.upsample(x_slice, scale_factor=self._scale_factor, mode=self._mode)
            x_slice_upsampled = ttnn.to_memory_config(x_slice_upsampled, ttnn.DRAM_MEMORY_CONFIG)

            sliced_results.append(x_slice_upsampled)
            ttnn.deallocate(x_slice)

        # Concatenate slices along channel dimension
        x_upsampled = ttnn.concat(sliced_results, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        logger.trace(
            f"TtUpsample channel slicing complete - output shape: {x_upsampled.shape}, memory_config: {x_upsampled.memory_config()}"
        )

        # Deallocate slice tensors
        for slice_result in sliced_results:
            ttnn.deallocate(slice_result)

        return x_upsampled

    def _perform_standard_upsample(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Perform standard upsample without slicing"""
        logger.trace(
            f"TtUpsample standard upsampling - input shape: {x.shape}, scale_factor: {self._scale_factor}, mode: {self._mode}"
        )
        x_upsampled = ttnn.upsample(x, scale_factor=self._scale_factor, mode=self._mode)
        logger.trace(
            f"TtUpsample standard upsampling complete - output shape: {x_upsampled.shape}, memory_config: {x_upsampled.memory_config()}"
        )
        return x_upsampled

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Apply upsample operation with optional channel slicing"""

        # Convert to ROW_MAJOR layout for upsampling if needed
        original_layout = x.layout
        if original_layout != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        batch_size, height, width, channels = x.shape

        # Check if channels are divisible by slice factor for channel slicing
        if self._slice_config.mode == SliceMode.CHANNEL and channels % self._slice_config.num_slices == 0:
            result = self._perform_channel_slicing(x)
        else:
            if self._slice_config.mode == SliceMode.CHANNEL:
                print(
                    f"Warning: Channel slicing requested but channels ({channels}) not divisible by {self._slice_config.num_slices}. Using standard upsample."
                )
            result = self._perform_standard_upsample(x)

        # Ensure proper memory configuration after upsampling
        result = ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)

        # Convert back to original layout if needed
        if original_layout != ttnn.ROW_MAJOR_LAYOUT:
            result = ttnn.to_layout(result, original_layout)

        return result

    @property
    def scale_factor(self) -> tuple[int, int]:
        return self._scale_factor

    @property
    def mode(self) -> str:
        return self._mode

    @classmethod
    def create_with_channel_slicing(
        cls,
        device: ttnn.MeshDevice,
        scale_factor: Union[int, tuple[int, int]],
        num_slices: int,
        mode: str = "bilinear",
    ) -> "TtUpsample":
        """Create TtUpsample with channel slicing configuration."""
        slice_config = SliceConfig(mode=SliceMode.CHANNEL, num_slices=num_slices)
        parameters = TtUpsampleParameters.create(
            device=device,
            scale_factor=scale_factor,
            mode=mode,
            slice_config=slice_config,
        )
        return cls(parameters)

    @classmethod
    def create(
        cls,
        device: ttnn.MeshDevice,
        scale_factor: Union[int, tuple[int, int]],
        mode: str = "bilinear",
    ) -> "TtUpsample":
        """Create TtUpsample without any slicing."""
        slice_config = SliceConfig(mode=SliceMode.NONE)
        parameters = TtUpsampleParameters.create(
            device=device,
            scale_factor=scale_factor,
            mode=mode,
            slice_config=slice_config,
        )
        return cls(parameters)
