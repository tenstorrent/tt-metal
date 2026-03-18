# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn


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


params_to_config_values = {
    (512, 256, 16): (10_000, 32),
    (256, 128, 16): (10_000, 32),
    (128, 64, 4): (100_000, 32),
    (64, 32, 4): (100_000, 32),
    (32, 16, 4): (100_000, 32),
}


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
    output_layout: ttnn.Layout = ttnn.TILE_LAYOUT
    # slice_strategy: Optional[SliceStrategy] = None
    # math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    # fp32_dest_acc_en: bool = False
    # packer_l1_acc: bool = False
    # enable_act_double_buffer: bool = False
    # enable_weights_double_buffer: bool = False
    # deallocate_activation: bool = True
    # reallocate_halo_output: bool = True
    # config_tensors_in_dram: bool = True


def _normalize_conv2d_activation(activation: str | tuple[str, dict] | None) -> ttnn.UnaryWithParam | None:
    if activation is None:
        return None
    if isinstance(activation, tuple):
        activation_name, kwargs = activation
    else:
        activation_name = activation.strip().lower()
    activation_aliases = {
        "swish": "silu",
    }
    activation_name = activation_aliases.get(activation_name, activation_name)

    if activation_name == "relu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    if activation_name == "silu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
    if activation_name == "gelu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU)
    if activation_name == "sigmoid":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)
    if activation_name == "tanh":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH)
    if activation_name == "leaky_relu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, kwargs["negative_slope"])

    supported = "relu, silu (alias swish), gelu, sigmoid, tanh, leaky_relu"
    raise ValueError(f"Unsupported conv activation '{activation}'. Supported activations: {supported}")


def output_length_from_input_length(input_length, conv1d_config: Conv1dConfiguration):
    return (
        (input_length - 1) * conv1d_config.stride
        - 2 * conv1d_config.padding[0]
        + conv1d_config.dilation * (conv1d_config.kernel_size - 1)
        + 1
    )


def get_conv2d_config_values(output_length, in_channels, out_channels, kernel_size) -> tuple[int, int]:
    if (in_channels, out_channels, kernel_size) in params_to_config_values:
        len_per_slice, act_block_h_override = params_to_config_values[(in_channels, out_channels, kernel_size)]
        slice_num = (output_length + len_per_slice - 1) // len_per_slice
    else:
        slice_num = 1
        act_block_h_override = 0

    return (slice_num, act_block_h_override)


def get_conv_configs(
    input_length, conv1d_config: Conv1dConfiguration, device: ttnn.Device
) -> tuple[ttnn.Conv2dConfig, ttnn.Conv2dSliceConfig, ttnn.DeviceComputeKernelConfig]:
    output_length = output_length_from_input_length(input_length, conv1d_config)

    slice_num, act_block_h_override = get_conv2d_config_values(
        output_length, conv1d_config.in_channels, conv1d_config.out_channels, conv1d_config.kernel_size
    )
    slice_config = (
        ttnn.Conv2dSliceConfig(num_slices=slice_num, slice_type=ttnn.Op2DDRAMSliceWidth) if slice_num > 1 else None
    )

    act_block_w_div = 1
    return (
        ttnn.Conv2dConfig(
            weights_dtype=conv1d_config.weights_dtype,
            # shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            output_layout=conv1d_config.output_layout,
            # deallocate_activation=conv1d_config.deallocate_activation,
            # reallocate_halo_output=conv1d_config.reallocate_halo_output,
            # enable_act_double_buffer=conv1d_config.enable_act_double_buffer,
            # enable_weights_double_buffer=conv1d_config.enable_weights_double_buffer,
            # config_tensors_in_dram=conv1d_config.config_tensors_in_dram,
            # force_split_reader=True,
            config_tensors_in_dram=True,  # Force tensors in DRAM to avoid L1 thrashing for large activations
            act_block_h_override=act_block_h_override,
            act_block_w_div=act_block_w_div,
            # reshard_if_not_optimal=True,
            activation=conv1d_config.activation,
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


def input1d_to_2d(input_tensor: ttnn.Tensor) -> ttnn.Tensor:
    batch_size = input_tensor.shape[0]
    input_length = input_tensor.shape[1]
    input_channel = input_tensor.shape[2]
    input_t = ttnn.to_memory_config(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(input_t, (batch_size, 1, input_length, input_channel))


class ConvTranspose1d:
    """Stateful ConvTranspose1d wrapper built on top of `ttnn.conv_transpose2d`."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        weight_tensor: ttnn.Tensor | None = None,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias_tensor: ttnn.Tensor | None = None,
        dtype: ttnn.DataType | None = None,
        conv_config: ttnn.Conv2dConfig | None = None,
        compute_config: ttnn.DeviceComputeKernelConfig | None = None,
        memory_config: ttnn.MemoryConfig | None = None,
    ) -> None:
        tile_width = 32
        if out_channels % tile_width == 0:
            output_layout = ttnn.TILE_LAYOUT
        elif (tile_width - out_channels % tile_width) / out_channels < 1.2:
            output_layout = ttnn.TILE_LAYOUT
        else:
            output_layout = ttnn.ROW_MAJOR_LAYOUT

        padding_final = resolve_padding(
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        self.device = device
        self.configuration = Conv1dConfiguration(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_final,
            dilation=dilation,
            groups=groups,
            dtype=dtype or ttnn.bfloat16,
            output_layout=output_layout,
        )
        self.weight_tensor = weight_tensor
        self.bias_tensor = bias_tensor
        self.compute_config = compute_config
        self.memory_config = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], key: str, module_prefix: str = "") -> None:
        base_key = f"{module_prefix}{key}" if module_prefix else key
        bias_key = f"{base_key}.bias"
        wt_torch = state_dict[f"{base_key}.weight"]
        wt = wt_torch.reshape(
            self.configuration.in_channels,
            self.configuration.out_channels // self.configuration.groups,
            1,
            self.configuration.kernel_size,
        )
        self.weight_tensor = ttnn.from_torch(
            wt,
            dtype=ttnn.bfloat16,
        )

        self.bias_tensor = None
        if bias_key in state_dict and state_dict[bias_key] is not None:
            self.bias_tensor = ttnn.from_torch(
                state_dict[bias_key].reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
            )

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        input_2d = input1d_to_2d(input_tensor)
        batch_size = input_2d.shape[0]
        input_length = input_2d.shape[2]
        conv2d_config, slice_config, compute_config = get_conv_configs(input_length, self.configuration, self.device)
        output, [self.weight_tensor, self.bias_tensor] = ttnn.conv_transpose2d(
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
            dram_slice_config=slice_config,
        )
        output_shape = output.shape
        x = ttnn.reshape(output, (batch_size, output_shape[2], output_shape[3]))
        return x
