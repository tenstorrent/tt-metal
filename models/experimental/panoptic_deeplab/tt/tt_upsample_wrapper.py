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

    SUPPORTED_MODES = {"bilinear", "nearest"}

    def __post_init__(self):
        if self.slice_config is None:
            self.slice_config = SliceConfig()

        # Normalize scale_factor to tuple
        if isinstance(self.scale_factor, int):
            self.scale_factor = (self.scale_factor, self.scale_factor)

        # Validate mode
        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Mode must be one of {self.SUPPORTED_MODES}, got '{self.mode}'")

    @classmethod
    def create(
        cls,
        device: ttnn.MeshDevice,
        scale_factor: Union[int, tuple[int, int]],
        mode: str = "bilinear",
        slice_config: SliceConfig | None = None,
    ) -> "TtUpsampleParameters":
        return cls(
            device=device,
            scale_factor=scale_factor,
            mode=mode,
            slice_config=slice_config or SliceConfig(),
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
        batch_size, height, width, channels = x.shape

        logger.trace(
            f"Channel slicing upsample - input: {x.shape}, slices: {self._slice_config.num_slices}, "
            f"scale: {self._scale_factor}, mode: {self._mode}"
        )

        if channels % self._slice_config.num_slices != 0:
            raise ValueError(
                f"Input channels ({channels}) must be divisible by num_slices ({self._slice_config.num_slices})"
            )

        channels_per_slice = channels // self._slice_config.num_slices
        sliced_results = []

        try:
            for slice_idx in range(self._slice_config.num_slices):
                logger.trace(f"Processing slice {slice_idx + 1}/{self._slice_config.num_slices}")

                start_ch = slice_idx * channels_per_slice
                end_ch = (slice_idx + 1) * channels_per_slice

                # Slice input along channel dimension
                x_slice = ttnn.slice(x, [0, 0, 0, start_ch], [batch_size, height, width, end_ch])

                # Apply upsample to slice
                x_slice_upsampled = ttnn.upsample(x_slice, scale_factor=self._scale_factor, mode=self._mode)
                x_slice_upsampled = ttnn.to_memory_config(x_slice_upsampled, ttnn.DRAM_MEMORY_CONFIG)

                sliced_results.append(x_slice_upsampled)
                ttnn.deallocate(x_slice)

            # Concatenate slices along channel dimension
            result = ttnn.concat(sliced_results, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            logger.trace(f"Channel slicing complete - output: {result.shape}")

            return result
        finally:
            # Cleanup slice tensors
            for slice_result in sliced_results:
                if slice_result is not None:
                    ttnn.deallocate(slice_result)

    def _perform_standard_upsample(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Perform standard upsample without slicing"""
        logger.trace(f"Standard upsample - input: {x.shape}, scale: {self._scale_factor}, mode: {self._mode}")

        result = ttnn.upsample(x, scale_factor=self._scale_factor, mode=self._mode)
        logger.trace(f"Standard upsample complete - output: {result.shape}")

        return result

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Apply upsample operation with optional channel slicing"""
        original_layout = x.layout

        # Convert to ROW_MAJOR layout for upsampling if needed
        if original_layout != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Determine whether to use channel slicing
        use_channel_slicing = self._should_use_channel_slicing(x)

        if use_channel_slicing:
            result = self._perform_channel_slicing(x)
        else:
            result = self._perform_standard_upsample(x)

        # Ensure proper memory configuration
        result = ttnn.to_memory_config(result, ttnn.DRAM_MEMORY_CONFIG)

        # Convert back to original layout if needed
        if original_layout != ttnn.ROW_MAJOR_LAYOUT:
            result = ttnn.to_layout(result, original_layout)

        return result

    def _should_use_channel_slicing(self, x: ttnn.Tensor) -> bool:
        """Determine if channel slicing should be used based on configuration and tensor properties"""
        if self._slice_config.mode != SliceMode.CHANNEL:
            return False

        _, _, _, channels = x.shape

        if channels % self._slice_config.num_slices != 0:
            logger.warning(
                f"Channel slicing requested but channels ({channels}) not divisible by "
                f"{self._slice_config.num_slices}. Using standard upsample."
            )
            return False

        return True

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
        parameters = TtUpsampleParameters.create(
            device=device,
            scale_factor=scale_factor,
            mode=mode,
        )
        return cls(parameters)
