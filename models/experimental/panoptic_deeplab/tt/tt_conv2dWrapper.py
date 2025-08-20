# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import ttnn

from ....experimental.stable_diffusion_35_large.tt.utils import from_torch_fast

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
    device: ttnn.MeshDevice
    dilation: tuple[int, int] = (1, 1)
    channel_slice_num: int = 1

    @classmethod
    def from_torch(
        cls,
        state: dict[str, torch.Tensor],
        *,
        device: ttnn.MeshDevice,
        dtype: ttnn.DataType | None = None,
    ) -> TtConv2dParameters:
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None

        return cls(
            weight=from_torch_fast(state["weight"], dtype=dtype, mesh_mapper=mesh_mapper),
            bias=from_torch_fast(state["bias"].reshape((1, 1, 1, -1)), dtype=dtype, mesh_mapper=mesh_mapper)
            if "bias" in state
            else None,
            device=device,
            dilation=state.get("dilation", (1, 1)),  # Extract dilation from state,
            channel_slice_num=state.get("channel_slice_num", 1),  # Extract channel_slice_num from state
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


CHANNEL_SLICE_THRESHOLD = 2048  # For now temporary
CHANNEL_SLICE_FACTOR = 4  # For now temporary


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
        slice_count: Optional[int] = None,
    ) -> None:
        self._stride = stride
        self._padding = padding

        self._in_channels = parameters.in_channels
        self._out_channels = parameters.out_channels
        self._kernel_size = parameters.kernel_size

        self._dilation = parameters.dilation
        self._channel_slice_num = parameters.channel_slice_num
        self._weight = parameters.weight
        self._bias = parameters.bias
        self._device = parameters.device
        self._weightSlices = []
        self._biasSlices = []

        self._slice_count = slice_count

    def call_without_reshape(
        self,
        x: ttnn.Tensor,
        *,
        conv_config: ttnn.Conv2dConfig | None = None,
        memory_config: ttnn.MemoryConfig | None = None,
    ) -> tuple[ttnn.Tensor, list[int]]:
        (batch_size, height, width, _) = x.shape

        if self._in_channels >= CHANNEL_SLICE_THRESHOLD:
            print(f"--- Running Conv2d with Channel Slicing (factor={self._channel_slice_num}) ---")
            assert (
                self._in_channels % self._channel_slice_num == 0
            ), f"Input channels ({self._in_channels}) must be divisible by channel_slice_num ({self._channel_slice_num})"

            split_in_channels = self._in_channels // self._channel_slice_num

            if not ttnn.is_tensor_storage_on_device(self._weight):
                self._weight = ttnn.to_device(self._weight, self._device)

            input_slices = []
            for i in range(self._channel_slice_num):
                start_idx = i * split_in_channels
                end_idx = (i + 1) * split_in_channels
                slice_tensor = ttnn.slice(x, [0, 0, 0, start_idx], [batch_size, height, width, end_idx])
                input_slices.append(slice_tensor)

            if self._weightSlices == []:
                for i in range(self._channel_slice_num):
                    start_idx = i * split_in_channels
                    end_idx = (i + 1) * split_in_channels
                    weight_slice = ttnn.slice(
                        self._weight,
                        [0, start_idx, 0, 0],
                        [self._out_channels, end_idx, self._kernel_size[0], self._kernel_size[1]],
                    )
                    self._weightSlices.append(weight_slice)

            accumulated_output = None
            output_height, output_width = 0, 0

            for i in range(self._channel_slice_num):
                print(f"  Processing channel slice {i+1}/{self._channel_slice_num}...")
                input_slice = input_slices[i]
                output_slice, [output_height, output_width], [self._weightSlices[i], self.biasTmp] = ttnn.conv2d(
                    input_tensor=input_slice,
                    weight_tensor=self._weightSlices[i],
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
            if self._bias is not None:
                accumulated_output = ttnn.add(accumulated_output, self._bias, output_tensor=accumulated_output)
            final_shape = [batch_size, output_height, output_width, self._out_channels]
            return accumulated_output, final_shape
        # No channel slicing needed, proceed with normal conv2d
        else:
            k = SLICE_COUNT_FACTOR.get((self._in_channels, self._out_channels, height * width), 1)
            slice_count = -(-k * batch_size // 1024)

            # Override with manual slice_count if provided
            if self._slice_count is not None:
                slice_count = self._slice_count
            if slice_count > 1:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

            def call_conv2d(t: ttnn.Tensor, w: ttnn.Tensor, b: ttnn.Tensor | None) -> _Conv2dRawResult:
                output, [output_height, output_width], [prepared_weight, prepared_bias] = ttnn.conv2d(
                    input_tensor=t,
                    weight_tensor=w,
                    bias_tensor=b,
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
                    slice_config=(
                        ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceWidth, num_slices=slice_count)
                        if slice_count > 1
                        else None
                    ),
                    dtype=ttnn.bfloat16,
                )
                return _Conv2dRawResult(
                    output=output,
                    output_height=output_height,
                    output_width=output_width,
                    prepared_weight=prepared_weight,
                    prepared_bias=prepared_bias,
                )

            results = call_conv2d(x, self._weight, self._bias)
            x = results.output
            self._weight = results.prepared_weight
            self._bias = results.prepared_bias
            if slice_count > 1:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            shape = [batch_size, results.output_height, results.output_width, self._out_channels]
            return x, shape

    def __call__(
        self,
        x: ttnn.Tensor,
        memory_config: ttnn.MemoryConfig | None = None,
    ) -> ttnn.Tensor:
        x, shape = self.call_without_reshape(x, memory_config=memory_config)

        # workaround
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)

        # kernel designed for rank 3 tensors error when reshaping 4D tiled tensors????
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


@dataclass
class _Conv2dRawResult:
    output: ttnn.Tensor
    output_height: int
    output_width: int
    prepared_weight: ttnn.Tensor
    prepared_bias: ttnn.Tensor | None
