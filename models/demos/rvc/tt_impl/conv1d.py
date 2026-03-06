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


def input1d_to_2d(
    input_tensor: ttnn.Tensor,
) -> ttnn.Tensor:
    batch_size, input_length, input_channel = input_tensor.shape
    return ttnn.reshape(input_tensor, (batch_size, 1, input_length, input_channel))


@dataclass(frozen=True)
class Conv1dConfiguration:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int | tuple[int, int] = 1
    padding: int | tuple[int, int] = 0
    dilation: int | tuple[int, int] = 1
    groups: int = 1
    activation: Optional[ttnn.UnaryWithParam] = None
    activation_dtype: ttnn.DataType = ttnn.bfloat16
    weights_dtype: ttnn.DataType = ttnn.bfloat16
    output_dtype: ttnn.DataType = ttnn.bfloat16
    # output_layout: ttnn.Layout = ttnn.TILE_LAYOUT
    slice_strategy: Optional[SliceStrategy] = None
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    fp32_dest_acc_en: bool = False
    packer_l1_acc: bool = False
    enable_act_double_buffer: bool = False
    enable_weights_double_buffer: bool = False
    deallocate_activation: bool = True
    reallocate_halo_output: bool = True
    config_tensors_in_dram: bool = True


def to_conv2d_config(configuration: Conv1dConfiguration):
    kwargs = dict(
        weights_dtype=configuration.weights_dtype,
        shard_layout=None,
        # output_layout=configuration.output_layout,
        deallocate_activation=configuration.deallocate_activation,
        reallocate_halo_output=configuration.reallocate_halo_output,
        enable_act_double_buffer=configuration.enable_act_double_buffer,
        enable_weights_double_buffer=configuration.enable_weights_double_buffer,
        config_tensors_in_dram=configuration.config_tensors_in_dram,
        force_split_reader=True,
        # act_block_h_override=32,
    )
    if configuration.activation is not None:
        kwargs["activation"] = configuration.activation
    return ttnn.Conv2dConfig(**kwargs)


def to_compute_config(configuration: Conv1dConfiguration, device: ttnn.Device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=configuration.math_fidelity,
        fp32_dest_acc_en=configuration.fp32_dest_acc_en,
        packer_l1_acc=configuration.packer_l1_acc,
    )


def to_slice_config(slice_type: str):
    if slice_type is None:
        return None
    if slice_type == "channel":
        return ttnn.Conv2dL1FullSliceConfig
    return None
    # return ttnn.Conv2dSliceConfig(
    #     slice_type=ttnn.,
    #     num_slices=slice_strategy.get_num_slices(),
    # )


dims_to_num_slices = {
    (56992, 512, 3): 7,
    # Conv1d: batch_size=1, input_length=569938, output_length=113986, in_channels=1, out_channels: 512, kernel_size=10, stride=5, padding=0, dilation=1
    (113986, 1, 10): 3,
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


# dims_to_num_slices_2 = {
#     (512, 512, 3),
#     (1, 512, 10),
#     ()
# }


def determine_slice_strategy(
    batch_size: int, ouput_length: int, in_channels: int, kernel_size: int
) -> Optional[SliceStrategy]:
    if (ouput_length, in_channels, kernel_size) in dims_to_num_slices:
        num_slices = dims_to_num_slices[(ouput_length, in_channels, kernel_size)]
        return ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceWidth)
    else:
        return ttnn.Op2DSliceConfig(num_slices=1, slice_type=ttnn.Op2DDRAMSliceWidth)
    l1_free_th = 1_300_000 * 60  # in bytes
    memory_cost = batch_size * ouput_length * in_channels * kernel_size * 2  # assuming bfloat16, so 2 bytes per element
    if memory_cost > l1_free_th:
        num_slices = (memory_cost + l1_free_th - 1) // l1_free_th + 2
        return ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceWidth)
    return None


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
        padding: int | tuple[int, int] = 0,
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
        configuration = Conv1dConfiguration(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            activation=None,
            output_dtype=dtype or ttnn.bfloat16,
        )

        self.device = device
        self.configuration = configuration
        self.in_channels = configuration.in_channels
        self.out_channels = configuration.out_channels
        self.kernel_size = configuration.kernel_size
        self.stride = configuration.stride
        self.padding = configuration.padding
        self.dilation = configuration.dilation
        self.groups = configuration.groups
        self.dtype = dtype or configuration.output_dtype
        self.memory_config = ttnn.L1_MEMORY_CONFIG

        # Keep internal naming aligned with TtConv2d while preserving old attributes.
        self.conv2d_config = to_conv2d_config(self.configuration)
        self.compute_config = to_compute_config(self.configuration, device)
        self.slice_config = to_slice_config(None)
        self.conv_config = self.conv2d_config

    def load_parameters(
        self,
        parameters: dict[str, torch.Tensor],
        key: str,
        prefix: str = "",
    ) -> None:
        base_key = f"{prefix}{key}" if prefix else key
        weight_key = f"{base_key}.weight"
        bias_key = f"{base_key}.bias"

        wt = parameters[weight_key].reshape(self.out_channels, self.in_channels // self.groups, 1, self.kernel_size)
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
        output_length = (
            input_length + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        ) // self.stride + 1
        slice_config = determine_slice_strategy(batch_size, output_length, self.in_channels, self.kernel_size)
        conv_result, [self.weight_tensor, self.bias_tensor] = ttnn.conv2d(
            input_tensor=input_2d,
            weight_tensor=self.weight_tensor,
            return_output_dim=False,
            return_weights_and_bias=True,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=1,
            input_width=input_length,
            kernel_size=[1, self.kernel_size],
            stride=[1, self.stride],
            padding=[0, self.padding],
            dilation=[1, self.dilation],
            groups=self.groups,
            bias_tensor=self.bias_tensor,
            dtype=self.dtype,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            slice_config=slice_config,
        )
        return conv_result

    def deallocate(self) -> None:
        if self.weight_tensor is not None:
            ttnn.deallocate(self.weight_tensor)
            self.weight_tensor = None
        if self.bias_tensor is not None:
            ttnn.deallocate(self.bias_tensor)
            self.bias_tensor = None
