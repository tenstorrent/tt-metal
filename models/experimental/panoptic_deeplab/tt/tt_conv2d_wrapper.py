# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from enum import Enum

import ttnn

from .common import from_torch_fast


class SliceMode(Enum):
    """Enumeration for different slicing modes"""

    NONE = "none"
    CHANNEL = "channel"
    HEIGHT = "height"
    WIDTH = "width"


@dataclass
class SliceConfig:
    """Configuration for manual slicing control"""

    mode: SliceMode = SliceMode.NONE
    num_slices: int = 1

    def __post_init__(self):
        if self.mode != SliceMode.NONE and self.num_slices <= 1:
            raise ValueError(f"{self.mode.value} slicing requires num_slices > 1")


@dataclass
class TtConv2dParameters:
    weight: ttnn.Tensor
    bias: ttnn.Tensor | None
    device: ttnn.MeshDevice
    dilation: tuple[int, int] = (1, 1)
    slice_config: SliceConfig = None

    def __post_init__(self):
        if self.slice_config is None:
            self.slice_config = SliceConfig()

    @classmethod
    def from_torch(
        cls,
        state: dict[str, any],
        *,
        device: ttnn.MeshDevice,
        dtype: ttnn.DataType | None = None,
        slice_config: SliceConfig | None = None,
    ) -> TtConv2dParameters:
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None

        if slice_config is None:
            slice_config = SliceConfig()

        # Convert torch tensors to ttnn tensors
        weight = from_torch_fast(state["weight"], dtype=dtype, mesh_mapper=mesh_mapper)
        bias = None
        if "bias" in state and state["bias"] is not None:
            bias = from_torch_fast(state["bias"].reshape((1, 1, 1, -1)), dtype=dtype, mesh_mapper=mesh_mapper)

        return cls(
            weight=weight,
            bias=bias,
            device=device,
            dilation=state.get("dilation", (1, 1)),
            slice_config=slice_config,
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
    TTNN Conv2d wrapper with explicit manual control over slicing strategies.

    Supports the following slicing modes:
    - Channel slicing: Splits input channels across multiple conv operations
    - Height slicing: Uses ttnn spatial slicing along height dimension
    - Width slicing: Uses ttnn spatial slicing along width dimension

    Note: Height and width slicing cannot be used simultaneously.

    Usage examples:

    # Create parameters with channel slicing
    slice_config = SliceConfig(mode=SliceMode.CHANNEL, num_slices=4)
    params = TtConv2dParameters.from_torch(state_dict, device=device, slice_config=slice_config)
    conv = TtConv2d(params, stride=(2, 2), padding=(1, 1))

    # Or with spatial slicing
    slice_config = SliceConfig(mode=SliceMode.WIDTH, num_slices=2)
    params = TtConv2dParameters.from_torch(state_dict, device=device, slice_config=slice_config)

    # Using convenience methods
    conv = TtConv2d.create_with_channel_slicing(params, num_slices=4, stride=(2, 2))

    # Spatial slicing (height OR width, not both)
    conv = TtConv2d.create_with_width_slicing(params, num_slices=2)
    # OR
    conv = TtConv2d.create_with_height_slicing(params, num_slices=2)

    # No slicing
    conv = TtConv2d.create(params)

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

        self._dilation = parameters.dilation
        self._slice_config = parameters.slice_config
        self._weight = parameters.weight
        self._bias = parameters.bias
        self._device = parameters.device

        # Caching for sliced weights and biases
        self._weight_slices = []
        self._bias_slices = []

    def _get_conv_config(self) -> ttnn.Conv2dConfig:
        """Create default conv2d configuration"""
        return ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            enable_split_reader=False,
            output_layout=ttnn.TILE_LAYOUT,
            activation="",
            transpose_shards=False,
            in_place=False,
            enable_kernel_stride_folding=False,
            full_inner_dim=False,
            act_block_h_override=32,
        )

    def _create_spatial_slice_config(self) -> Optional[ttnn.Conv2dSliceConfig]:
        """Create spatial slice configuration based on slice_config"""
        if self._slice_config.mode == SliceMode.HEIGHT:
            return ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=self._slice_config.num_slices)
        elif self._slice_config.mode == SliceMode.WIDTH:
            return ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceWidth, num_slices=self._slice_config.num_slices)
        return None

    def _perform_channel_slicing(
        self,
        x: ttnn.Tensor,
        batch_size: int,
        height: int,
        width: int,
        conv_config: ttnn.Conv2dConfig,
        memory_config: ttnn.MemoryConfig | None = None,
    ) -> tuple[ttnn.Tensor, list[int]]:
        """Perform channel slicing convolution"""
        print(f"--- Running Conv2d with Channel Slicing (factor={self._slice_config.num_slices}) ---")

        assert (
            self._in_channels % self._slice_config.num_slices == 0
        ), f"Input channels ({self._in_channels}) must be divisible by num_slices ({self._slice_config.num_slices})"

        split_in_channels = self._in_channels // self._slice_config.num_slices

        if not ttnn.is_tensor_storage_on_device(self._weight):
            self._weight = ttnn.to_device(self._weight, self._device)

        # Create input slices
        input_slices = []
        for i in range(self._slice_config.num_slices):
            start_idx = i * split_in_channels
            end_idx = (i + 1) * split_in_channels
            slice_tensor = ttnn.slice(x, [0, 0, 0, start_idx], [batch_size, height, width, end_idx])
            input_slices.append(slice_tensor)

        # Create weight slices (cached)
        if not self._weight_slices:
            for i in range(self._slice_config.num_slices):
                start_idx = i * split_in_channels
                end_idx = (i + 1) * split_in_channels
                weight_slice = ttnn.slice(
                    self._weight,
                    [0, start_idx, 0, 0],
                    [self._out_channels, end_idx, self._kernel_size[0], self._kernel_size[1]],
                )
                self._weight_slices.append(weight_slice)

        accumulated_output = None
        output_height, output_width = 0, 0

        # Process each channel slice
        for i in range(self._slice_config.num_slices):
            print(f"  Processing channel slice {i+1}/{self._slice_config.num_slices}...")
            input_slice = input_slices[i]
            output_slice, [output_height, output_width], [self._weight_slices[i], bias_tmp] = ttnn.conv2d(
                input_tensor=input_slice,
                weight_tensor=self._weight_slices[i],
                bias_tensor=None,
                in_channels=split_in_channels,
                out_channels=self._out_channels,
                device=self._device,
                kernel_size=list(self._kernel_size),
                stride=list(self._stride),
                padding=list(self._padding),
                batch_size=batch_size,
                input_height=height,
                input_width=width,
                dilation=list(self._dilation),
                return_output_dim=True,
                return_weights_and_bias=True,
                conv_config=conv_config,
                memory_config=memory_config,
                dtype=ttnn.bfloat16,
            )
            output_slice = ttnn.move(output_slice)
            if i == 0:
                accumulated_output = ttnn.to_memory_config(output_slice, ttnn.DRAM_MEMORY_CONFIG)
            else:
                accumulated_output = ttnn.add(
                    output_slice, accumulated_output, memory_config=memory_config, output_tensor=accumulated_output
                )
                output_slice.deallocate(True)

        # Add bias if present
        if self._bias is not None:
            accumulated_output = ttnn.add(accumulated_output, self._bias, output_tensor=accumulated_output)

        final_shape = [batch_size, output_height, output_width, self._out_channels]
        return accumulated_output, final_shape

    def _perform_standard_conv(
        self,
        x: ttnn.Tensor,
        batch_size: int,
        height: int,
        width: int,
        conv_config: ttnn.Conv2dConfig,
        memory_config: ttnn.MemoryConfig | None = None,
        spatial_slice_config: Optional[ttnn.Conv2dSliceConfig] = None,
    ) -> tuple[ttnn.Tensor, list[int]]:
        """Perform standard convolution with optional spatial slicing"""
        output, [output_height, output_width], [prepared_weight, prepared_bias] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self._weight,
            bias_tensor=self._bias,
            in_channels=self._in_channels,
            out_channels=self._out_channels,
            device=self._device,
            kernel_size=list(self._kernel_size),
            stride=list(self._stride),
            padding=list(self._padding),
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            dilation=list(self._dilation),
            return_output_dim=True,
            return_weights_and_bias=True,
            conv_config=conv_config,
            memory_config=memory_config,
            slice_config=spatial_slice_config,
            dtype=ttnn.bfloat16,
        )

        # Update prepared weights and bias
        self._weight = prepared_weight
        self._bias = prepared_bias

        shape = [batch_size, output_height, output_width, self._out_channels]
        return output, shape

    def _call_without_reshape(
        self,
        x: ttnn.Tensor,
        *,
        conv_config: ttnn.Conv2dConfig | None = None,
        memory_config: ttnn.MemoryConfig | None = None,
    ) -> tuple[ttnn.Tensor, list[int]]:
        (batch_size, height, width, _) = x.shape

        # Use provided conv_config or create default
        if conv_config is None:
            conv_config = self._get_conv_config()

        # Determine slicing strategy based on slice_config
        if self._slice_config.mode == SliceMode.CHANNEL:
            return self._perform_channel_slicing(x, batch_size, height, width, conv_config, memory_config)
        else:
            # Handle spatial slicing (height, width, or both)
            spatial_slice_config = self._create_spatial_slice_config()
            return self._perform_standard_conv(
                x, batch_size, height, width, conv_config, memory_config, spatial_slice_config
            )

    def __call__(
        self,
        x: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig | None = None,
        conv_config: Optional[ttnn.Conv2dConfig] = None,
    ) -> ttnn.Tensor:
        x, shape = self._call_without_reshape(x, memory_config=memory_config, conv_config=conv_config)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # tiled tensors need to be rank 3 for reshape
        x = ttnn.squeeze(x, dim=0)
        shape = [shape[1], shape[2], shape[3]]
        x = ttnn.reshape(x, shape)
        x = ttnn.unsqueeze(x, dim=0)

        return x

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def kernel_size(self) -> tuple[int, int]:
        return self._kernel_size

    @classmethod
    def create_with_channel_slicing(
        cls,
        parameters: TtConv2dParameters,
        num_slices: int,
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
    ) -> "TtConv2d":
        """Create TtConv2d with channel slicing configuration."""
        parameters.slice_config = SliceConfig(mode=SliceMode.CHANNEL, num_slices=num_slices)
        return cls(parameters, stride=stride, padding=padding)

    @classmethod
    def create_with_height_slicing(
        cls,
        parameters: TtConv2dParameters,
        num_slices: int,
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
    ) -> "TtConv2d":
        """Create TtConv2d with height slicing configuration."""
        parameters.slice_config = SliceConfig(mode=SliceMode.HEIGHT, num_slices=num_slices)
        return cls(parameters, stride=stride, padding=padding)

    @classmethod
    def create_with_width_slicing(
        cls,
        parameters: TtConv2dParameters,
        num_slices: int,
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
    ) -> "TtConv2d":
        """Create TtConv2d with width slicing configuration."""
        parameters.slice_config = SliceConfig(mode=SliceMode.WIDTH, num_slices=num_slices)
        return cls(parameters, stride=stride, padding=padding)

    @classmethod
    def create(
        cls,
        parameters: TtConv2dParameters,
        *,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
    ) -> "TtConv2d":
        """Create TtConv2d without any slicing."""
        parameters.slice_config = SliceConfig(mode=SliceMode.NONE)
        return cls(parameters, stride=stride, padding=padding)
