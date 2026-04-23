# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

from .utils import ConvConfiguration, get_shard_strategy_for_conv, resolve_padding_1d

PARAMS_TO_CONFIG_VALUES = {
    (512, 256, 16): (4_000, 32),
    (256, 128, 16): (10_000, 32),
    (128, 64, 4): (100_000, 32),
    (64, 32, 4): (100_000, 32),
    (32, 16, 4): (100_000, 32),
}


def output_length_from_input_length(input_length, conv1d_config: ConvConfiguration):
    return (
        (input_length - 1) * conv1d_config.stride
        - 2 * conv1d_config.padding[0]
        + conv1d_config.dilation * (conv1d_config.kernel_size - 1)
        + 1
    )


def get_conv2d_config_values(output_length, in_channels, out_channels, kernel_size) -> tuple[int, int]:
    if (in_channels, out_channels, kernel_size) in PARAMS_TO_CONFIG_VALUES:
        len_per_slice, act_block_h_override = PARAMS_TO_CONFIG_VALUES[(in_channels, out_channels, kernel_size)]
        slice_num = (output_length + len_per_slice - 1) // len_per_slice
    else:
        slice_num = 1
        act_block_h_override = 0

    return (slice_num, act_block_h_override)


def get_conv_configs(
    input_shape, conv1d_config: ConvConfiguration, device: ttnn.Device
) -> tuple[ttnn.Conv2dConfig, ttnn.Conv2dSliceConfig, ttnn.DeviceComputeKernelConfig]:
    input_length = input_shape[1]

    output_length = output_length_from_input_length(input_length, conv1d_config)

    slice_num, act_block_h_override = get_conv2d_config_values(
        output_length, conv1d_config.in_channels, conv1d_config.out_channels, conv1d_config.kernel_size
    )
    slice_config = (
        ttnn.Conv2dSliceConfig(num_slices=slice_num, slice_type=ttnn.Op2DDRAMSliceWidth) if slice_num > 1 else None
    )

    act_block_w_div = 1
    shard_layout = get_shard_strategy_for_conv(input_shape) if slice_config is None else None

    return (
        ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=shard_layout,
            output_layout=conv1d_config.output_layout,
            deallocate_activation=conv1d_config.deallocate_input,
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
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
    )


class ConvTranspose1d:
    """Stateful ConvTranspose1d wrapper built on top of `ttnn.conv_transpose2d`."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        dtype: ttnn.DataType | None = None,
        deallocate_input: bool = False,
    ) -> None:
        if isinstance(padding, str):
            raise ValueError("String padding mode is not supported for ConvTranspose1d")
        tile_width = 32
        if out_channels % tile_width == 0:
            output_layout = ttnn.TILE_LAYOUT
        elif (tile_width - out_channels % tile_width) / out_channels < 1.2:
            output_layout = ttnn.TILE_LAYOUT
        else:
            output_layout = ttnn.ROW_MAJOR_LAYOUT

        padding_final = resolve_padding_1d(
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        self.device = device
        self.configuration = ConvConfiguration(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_final,
            dilation=dilation,
            groups=groups,
            dtype=dtype or ttnn.bfloat16,
            output_layout=output_layout,
            deallocate_input=deallocate_input,
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], key: str, module_prefix: str | None = None) -> None:
        if module_prefix is None:
            module_prefix = ""
        base_key = f"{module_prefix}{key}" if module_prefix else key
        bias_key = f"{base_key}.bias"
        reshaped_weight_torch = state_dict[f"{base_key}.weight"]
        reshaped_weight = reshaped_weight_torch.reshape(
            self.configuration.in_channels,
            self.configuration.out_channels // self.configuration.groups,
            1,
            self.configuration.kernel_size,
        )
        self.weight_tensor = ttnn.from_torch(
            reshaped_weight,
            dtype=ttnn.bfloat16,
        )

        self.bias_tensor = None
        if bias_key in state_dict and state_dict[bias_key] is not None:
            self.bias_tensor = ttnn.from_torch(
                state_dict[bias_key].reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
            )

    def __call__(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        batch_size, input_length, _ = input_tensor.shape
        conv2d_config, slice_config, compute_config = get_conv_configs(
            input_tensor.shape, self.configuration, self.device
        )
        out, [self.weight_tensor, self.bias_tensor] = ttnn.conv_transpose2d(
            input_tensor=ttnn.unsqueeze(input_tensor, dim=1),
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
        return ttnn.squeeze(out, dim=1)
