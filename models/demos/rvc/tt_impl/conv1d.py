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


class TTConv1d:
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

        conv_result, [self.weight_tensor, self.bias_tensor] = ttnn.conv2d(
            # input_tensor = ttnn.to_memory_config(input_2d, memory_config=ttnn.DRAM_MEMORY_CONFIG),
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
            # slice_config=ttnn.Conv2dL1FullSliceConfig,
            # "slice_config=self.slice_config,
            compute_config=self.compute_config,
            # "memory_config=self.memory_config,
            # slice_config=ttnn.Conv2dDRAMSliceWidth,
            # slice_config=ttnn.Conv2dSliceConfig(
            #     slice_type=ttnn.Conv2dSliceWidth,
            #     num_slices=4,
            # )
            # slice_config=ttnn.Conv2dDRAMSliceHeight,
            # slice_config=slice_config,
        )
        return conv_result

    def deallocate(self) -> None:
        if self.weight_tensor is not None:
            ttnn.deallocate(self.weight_tensor)
            self.weight_tensor = None
        if self.bias_tensor is not None:
            ttnn.deallocate(self.bias_tensor)
            self.bias_tensor = None
