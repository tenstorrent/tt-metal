# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

import ttnn

InputLayout = Literal["NLC", "NHWC"]
OutputLayout = Literal["NLC", "NHWC"]
PaddingType = int | tuple[int, int] | Literal["same"]


def input1d_to_2d(
    input_tensor: ttnn.Tensor,
) -> ttnn.Tensor:
    batch_size, input_length, input_channel = input_tensor.shape
    return ttnn.reshape(input_tensor, (batch_size, 1, input_length, input_channel))


def input_shape_to_memory_config(
    input_shape, output_length, in_channels, kernel_size, device: ttnn.MeshDevice
) -> ttnn.MemoryConfig:
    batch_size, input_height, input_width, input_channels = input_shape
    memory_cost = (
        batch_size * input_height * input_width * input_channels * 2
    )  # assuming bfloat16, so 2 bytes per element
    if (output_length, in_channels, kernel_size) in dims_to_num_slices:
        return ttnn.DRAM_MEMORY_CONFIG

    # Keep tiny-channel inputs interleaved to avoid expensive/invalid sharding setups.

    if memory_cost > 64 * 1_400_000:  # if input is larger than 1.4MB, use DRAM to avoid L1 thrashing
        return ttnn.DRAM_MEMORY_CONFIG
    if input_channels < 16:
        return ttnn.DRAM_MEMORY_CONFIG

    nhw = batch_size * input_height * input_width
    c = input_channels

    # Use best sharding strategy based on NHW-to-C ratio:
    # - HEIGHT_SHARDED if NHW >> C
    # - WIDTH_SHARDED if C >> NHW
    # - BLOCK_SHARDED if NHW ~= C
    if nhw >= 4 * c:
        strategy = ttnn.ShardStrategy.HEIGHT
    elif c >= 4 * nhw:
        strategy = ttnn.ShardStrategy.WIDTH
    else:
        strategy = ttnn.ShardStrategy.BLOCK

    grid_size = device.compute_with_storage_grid_size()
    candidate_grids = [
        ttnn.CoreGrid(y=grid_size.y, x=grid_size.x),
        ttnn.CoreGrid(y=grid_size.y, x=1),
        ttnn.CoreGrid(y=1, x=grid_size.x),
        ttnn.CoreGrid(y=1, x=1),
    ]

    for core_grid in candidate_grids:
        try:
            return ttnn.create_sharded_memory_config_(
                input_shape,
                core_grid=core_grid,
                strategy=strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
        except RuntimeError:
            continue

    return ttnn.DRAM_MEMORY_CONFIG


@dataclass(frozen=True)
class Conv1dConfiguration:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: tuple[int, int] = (0, 0)  # (padding_left, padding_right)
    dilation: int = 1
    groups: int = 1
    activation: Optional[ttnn.UnaryWithParam] = None
    activation_dtype: ttnn.DataType = ttnn.bfloat16
    weights_dtype: ttnn.DataType = ttnn.bfloat16
    dtype: ttnn.DataType = ttnn.bfloat16
    # # output_layout: ttnn.Layout = ttnn.TILE_LAYOUT
    # slice_strategy: Optional[SliceStrategy] = None
    # math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    # fp32_dest_acc_en: bool = False
    # packer_l1_acc: bool = False
    # enable_act_double_buffer: bool = False
    # enable_weights_double_buffer: bool = False
    # deallocate_activation: bool = True
    # reallocate_halo_output: bool = True
    # config_tensors_in_dram: bool = True


def resolve_padding(
    padding: PaddingType,
    kernel_size: int,
    stride: int,
    dilation: int,
) -> tuple[int, int]:
    if isinstance(padding, str):
        if padding != "same":
            raise ValueError(f"Unsupported padding mode: {padding}")
        if stride != 1:
            raise ValueError("Only stride=1 is supported for 'same' padding")
        padding_needed = dilation * (kernel_size - 1)
        left_padding = padding_needed // 2
        right_padding = padding_needed - left_padding
        return (left_padding, right_padding)

    if isinstance(padding, tuple):
        if len(padding) != 2:
            raise ValueError(f"padding tuple must contain 2 values, got {len(padding)}")
        return (padding[0], padding[1])

    return (padding, padding)


def output_length_from_input_length(input_length, conv1d_config: Conv1dConfiguration):
    padding_left, padding_right = conv1d_config.padding[0], conv1d_config.padding[1]
    return (
        input_length + padding_left + padding_right - conv1d_config.dilation * (conv1d_config.kernel_size - 1) - 1
    ) // conv1d_config.stride + 1


def input_shape_to_slice_config(input_shape, conv1d_config: Conv1dConfiguration) -> Optional[ttnn.Conv2dSliceConfig]:
    batch_size, input_height, input_width, input_channels = input_shape
    output_len = output_length_from_input_length(input_width, conv1d_config)
    # return determine_slice_strategy(batch_size, output_len, conv1d_config.in_channels, conv1d_config.kernel_size)

    if (output_len, conv1d_config.in_channels, conv1d_config.kernel_size) in dims_to_num_slices:
        num_slices = dims_to_num_slices[(output_len, conv1d_config.in_channels, conv1d_config.kernel_size)]
        # num_slices = 1
        return ttnn.Conv2dSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceWidth)

    return None


def get_conv_configs(
    input_shape, conv1d_config: Conv1dConfiguration, device: ttnn.Device
) -> tuple[ttnn.Conv2dConfig, ttnn.Conv2dSliceConfig, ttnn.DeviceComputeKernelConfig]:
    slice_config = input_shape_to_slice_config(input_shape, conv1d_config)
    # if conv1d_config.activation is not None:
    #     activation = conv1d_config.activation

    activation = conv1d_config.activation if conv1d_config.activation is not None else None
    return (
        ttnn.Conv2dConfig(
            weights_dtype=conv1d_config.weights_dtype,
            shard_layout=None,
            # output_layout=conv1d_config.output_layout,
            # deallocate_activation=conv1d_config.deallocate_activation,
            # reallocate_halo_output=conv1d_config.reallocate_halo_output,
            # enable_act_double_buffer=conv1d_config.enable_act_double_buffer,
            # enable_weights_double_buffer=conv1d_config.enable_weights_double_buffer,
            # config_tensors_in_dram=conv1d_config.config_tensors_in_dram,
            # force_split_reader=True,
            # act_block_h_override=32,
            activation=activation,
            # slice_config=slice_config,
        ),
        slice_config,
        ttnn.init_device_compute_kernel_config(
            device.arch(),
            # math_fidelity=conv1d_config.math_fidelity,
            # fp32_dest_acc_en=conv1d_config.fp32_dest_acc_en,
            # packer_l1_acc=conv1d_config.packer_l1_acc,
        ),
    )


dims_to_num_slices = {
    (56992, 512, 3): 7,
    # Conv1d: batch_size=1, input_length=569938, output_length=113986, in_channels=1, out_channels: 512, kernel_size=10, stride=5, padding=0, dilation=1
    (113986, 1, 10): 3,
    (113995, 1, 10): 64,
    # Conv1d: batch_size=1, input_length=56992, output_length=28495, in_channels=512, out_channels: 512, kernel_size=3, stride=2, padding=0, dilation=1
    (28495, 512, 3): 4,
    # Conv1d: batch_size=1, input_length=28495, output_length=14247, in_channels=512, out_channels: 512, kernel_size=3, stride=2, padding=0, dilation=1
    (14247, 512, 3): 2,
    # Conv1d: batch_size=1, input_length=1780, output_length=1781, in_channels=768, out_channels: 768, kernel_size=128, stride=1, padding=64, dilation=1
    (1781, 768, 128): 56,
    # Conv1d: batch_size=1, input_length=1708800, output_length=35600, in_channels=1, out_channels: 256, kernel_size=96, stride=48, padding=24, dilation=1
    (35600, 1, 96): 3,
    # Conv1d: batch_size=1, input_length=35600, output_length=35600, in_channels=256, out_channels: 256, kernel_size=3, stride=1, padding=1, dilation=1
    (35600, 256, 3): 2,
    # Conv1d: batch_size=1, input_length=35600, output_length=35600, in_channels=256, out_channels: 256, kernel_size=7, stride=1, padding=3, dilation=1
    (35600, 256, 7): 4,
    # Conv1d: batch_size=1, input_length=35600, output_length=35600, in_channels=256, out_channels: 256, kernel_size=11, stride=1, padding=5, dilation=1
    (35600, 256, 11): 6,
    # Conv1d: batch_size=1, input_length=1708800, output_length=213600, in_channels=1, out_channels: 128, kernel_size=16, stride=8, padding=4, dilation=1
    (213600, 1, 16): 3,
    # Conv1d: batch_size=1, input_length=213600, output_length=213600, in_channels=128, out_channels: 128, kernel_size=3, stride=1, padding=1, dilation=1
    (213600, 128, 3): 6,
    # Conv1d: batch_size=1, input_length=213600, output_length=213600, in_channels=128, out_channels: 128, kernel_size=7, stride=1, padding=3, dilation=1
    (213600, 128, 7): 14,
    # Conv1d: batch_size=1, input_length=213600, output_length=213600, in_channels=128, out_channels: 128, kernel_size=11, stride=1, padding=5, dilation=1
    (213600, 128, 11): 27,
    # Conv1d: batch_size=1, input_length=1708800, output_length=427200, in_channels=1, out_channels: 64, kernel_size=8, stride=4, padding=2, dilation=1
    (427200, 1, 8): 3,
    # Conv1d: batch_size=1, input_length=427200, output_length=427200, in_channels=64, out_channels: 64, kernel_size=3, stride=1, padding=1, dilation=1
    (427200, 64, 3): 6,
    # Conv1d: batch_size=1, input_length=427200, output_length=427200, in_channels=64, out_channels: 64, kernel_size=7, stride=1, padding=3, dilation=1
    (427200, 64, 7): 11,
    # Conv1d: batch_size=1, input_length=427200, output_length=427200, in_channels=64, out_channels: 64, kernel_size=11, stride=1, padding=5, dilation=1
    (427200, 64, 11): 18,
    # Conv1d: batch_size=1, input_length=1708800, output_length=854400, in_channels=1, out_channels: 32, kernel_size=4, stride=2, padding=1, dilation=1
    (854400, 1, 4): 3,
    # Conv1d: batch_size=1, input_length=854400, output_length=854400, in_channels=32, out_channels: 32, kernel_size=3, stride=1, padding=1, dilation=1
    (854400, 32, 3): 6,
    # Conv1d: batch_size=1, input_length=854400, output_length=854400, in_channels=32, out_channels: 32, kernel_size=7, stride=1, padding=3, dilation=1
    (854400, 32, 7): 11,
    # Conv1d: batch_size=1, input_length=854400, output_length=854400, in_channels=32, out_channels: 32, kernel_size=11, stride=1, padding=5, dilation=1
    (854400, 32, 11): 17,
    # Conv1d: batch_size=1, input_length=1708800, output_length=1708800, in_channels=16, out_channels: 16, kernel_size=3, stride=1, padding=1, dilation=1
    (1708800, 16, 3): 8,
    # Conv1d: batch_size=1, input_length=1708800, output_length=1708800, in_channels=16, out_channels: 16, kernel_size=7, stride=1, padding=3, dilation=1
    (1708800, 16, 7): 13,
    # Conv1d: batch_size=1, input_length=1708800, output_length=1708800, in_channels=16, out_channels: 16, kernel_size=11, stride=1, padding=5, dilation=1
    (1708800, 16, 11): 18,
}

# def determine_slice_strategy(
#     batch_size: int, ouput_length: int, in_channels: int, kernel_size: int
# ) -> Optional[SliceStrategy]:
#     if (ouput_length, in_channels, kernel_size) in dims_to_num_slices:
#         num_slices = dims_to_num_slices[(ouput_length, in_channels, kernel_size)]
#         return ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceWidth)
#     else:
#         return ttnn.Op2DSliceConfig(num_slices=1, slice_type=ttnn.Op2DDRAMSliceWidth)
#     l1_free_th = 1_300_000 * 60  # in bytes
#     memory_cost = batch_size * ouput_length * in_channels * kernel_size * 2  # assuming bfloat16, so 2 bytes per element
#     if memory_cost > l1_free_th:
#         num_slices = (memory_cost + l1_free_th - 1) // l1_free_th + 2
#         return ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceWidth)
#     return None


class Conv1d:
    """Stateful Conv1d wrapper around `ttnn.conv1d`."""

    def __init__(
        self,
        # configuration: Conv1dConfiguration | None = None,
        device: ttnn.MeshDevice | None = None,
        *,
        in_channels: int | None = None,
        out_channels: int | None = None,
        kernel_size: int | None = None,
        stride: int = 1,
        padding: PaddingType = 0,
        dilation: int = 1,
        groups: int = 1,
        dtype: ttnn.DataType | None = None,
    ) -> None:
        # if configuration is None:
        if device is None:
            raise ValueError("device is required")
        if in_channels is None or out_channels is None or kernel_size is None:
            raise ValueError(
                "in_channels, out_channels, and kernel_size are required when configuration is not provided"
            )
        if isinstance(padding, str) and padding != "same":
            raise ValueError(f"Unsupported padding mode: {padding}")
        padding_final = resolve_padding(
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )
        configuration = Conv1dConfiguration(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_final,
            dilation=dilation,
            groups=groups,
            activation=None,
            dtype=dtype or ttnn.bfloat16,
        )

        self.device = device
        self.configuration = configuration
        self.memory_config = ttnn.L1_MEMORY_CONFIG

    def load_parameters(
        self,
        parameters: dict[str, torch.Tensor],
        key: str,
        prefix: str = "",
    ) -> None:
        base_key = f"{prefix}{key}" if prefix else key
        weight_key = f"{base_key}.weight"
        bias_key = f"{base_key}.bias"

        wt = parameters[weight_key].reshape(
            self.configuration.out_channels,
            self.configuration.in_channels // self.configuration.groups,
            1,
            self.configuration.kernel_size,
        )
        bias = parameters[bias_key] if bias_key in parameters and parameters[bias_key] is not None else None
        self.weight_tensor = ttnn.from_torch(wt, dtype=ttnn.bfloat16)
        self.bias_tensor = None
        if bias is not None:
            self.bias_tensor = ttnn.from_torch(
                torch.reshape(bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
            )

    def __call__(
        self,
        input_tensor: ttnn.Tensor,
    ):
        input_2d = input1d_to_2d(input_tensor)
        batch_size = input_2d.shape[0]
        input_length = input_2d.shape[2]
        conv2d_config, slice_config, compute_config = get_conv_configs(input_2d.shape, self.configuration, self.device)
        conv_result, [self.weight_tensor, self.bias_tensor] = ttnn.conv2d(
            input_tensor=input_2d,
            weight_tensor=self.weight_tensor,
            return_output_dim=False,
            return_weights_and_bias=True,
            device=self.device,
            in_channels=self.configuration.in_channels,
            out_channels=self.configuration.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=input_length,
            kernel_size=[1, self.configuration.kernel_size],
            stride=[1, self.configuration.stride],
            padding=[0, 0, self.configuration.padding[0], self.configuration.padding[1]],
            dilation=[1, self.configuration.dilation],
            groups=self.configuration.groups,
            bias_tensor=self.bias_tensor,
            dtype=self.configuration.dtype,
            conv_config=conv2d_config,
            compute_config=compute_config,
            slice_config=slice_config,
        )
        output_shape = conv_result.shape
        conv_result = ttnn.to_layout(conv_result, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(conv_result, (batch_size, output_shape[2], output_shape[3]))
        return x

    def deallocate(self) -> None:
        if self.weight_tensor is not None:
            ttnn.deallocate(self.weight_tensor)
            self.weight_tensor = None
        if self.bias_tensor is not None:
            ttnn.deallocate(self.bias_tensor)
            self.bias_tensor = None
