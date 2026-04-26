# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from .utils import ConvConfiguration, _normalize_conv2d_activation, get_shard_strategy_for_conv, resolve_padding_1d

PARAMS_TO_CONFIG_VALUES = {
    # (1, 512, 10): (100_000, 32 * 8),
    (1, 512, 10): (100_000, 32),
    (512, 512, 3): (20_000, 32),
    (768, 768, 128): (50, 32),
    # (1, 256, 96): (100_000, 32 * 4),
    # (256, 256, 3): (20_000, 32 * 4),
    # (256, 256, 7): (20_000, 32 * 8),
    # (256, 256, 11): (10_000, 32 * 8),
    # (1, 128, 16): (100_000, 32 * 16),
    # (128, 128, 3): (90_000, 32 * 8),
    # (128, 128, 7): (90_000, 32 * 4),
    (1, 256, 96): (100_000, 32),
    (256, 256, 3): (20_000, 32),
    (256, 256, 7): (20_000, 32),
    (256, 256, 11): (10_000, 32),
    (1, 128, 16): (100_000, 32),
    (128, 128, 3): (90_000, 32),
    (128, 128, 7): (90_000, 32),
    (128, 128, 11): (90_000, 32),
    (1, 64, 8): (200_000, 32),
    (64, 64, 3): (200_000, 32),
    (64, 64, 7): (200_000, 32),
    (64, 64, 11): (200_000, 32),
    (1, 32, 4): (400_000, 32 * 4),
    (32, 32, 3): (400_000, 32 * 24),
    (32, 32, 7): (400_000, 32),
    (32, 32, 11): (400_000, 32 * 8),
    (16, 16, 3): (50_000, 32 * 0),
    (16, 16, 7): (50_000, 32),
    (16, 16, 11): (50_000, 32 * 0),
    (16, 1, 7): (50_000, 32 * 0),
}


def output_length_from_input_length(input_length, conv1d_config: ConvConfiguration):
    padding_left, padding_right = conv1d_config.padding[0], conv1d_config.padding[1]
    return (
        input_length + padding_left + padding_right - conv1d_config.dilation * (conv1d_config.kernel_size - 1) - 1
    ) // conv1d_config.stride + 1


def get_conv2d_config_values(output_length, in_channels, out_channels, kernel_size) -> tuple[int, int]:
    if (in_channels, out_channels, kernel_size) in PARAMS_TO_CONFIG_VALUES:
        len_per_slice, act_block_h_override = PARAMS_TO_CONFIG_VALUES[(in_channels, out_channels, kernel_size)]
        slice_num = (output_length + len_per_slice - 1) // len_per_slice
    else:
        act_block_h_override = 0
        slice_num = 1

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
    # if conv1d_config.kernel_size == 128:
    #     slice_config = ttnn.Conv2dSliceConfig(num_slices=slice_num, slice_type=ttnn.Op2DDRAMSliceWidth)
    #     shard_layout = None

    act_block_w_div = 1
    shard_layout = get_shard_strategy_for_conv(input_shape) if slice_config is None else None

    return (
        ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            # shard_layout=shard_layout,
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
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )


class Conv1d:
    """Stateful Conv1d wrapper around `ttnn.conv1d`."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        *,
        in_channels: int | None = None,
        out_channels: int | None = None,
        kernel_size: int | None = None,
        stride: int = 1,
        padding: PaddingType = 0,
        dilation: int = 1,
        groups: int = 1,
        dtype: ttnn.DataType | None = None,
        activation: str | tuple[str, dict] | None = None,
        deallocate_input: bool = False,
    ) -> None:
        self.device = device
        if isinstance(padding, str) and padding != "same":
            raise ValueError(f"Unsupported padding mode: {padding}")
        padding_final = resolve_padding_1d(
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        TILE_WIDTH = 32
        if out_channels % TILE_WIDTH == 0:
            output_layout = ttnn.TILE_LAYOUT
        elif (TILE_WIDTH - out_channels % TILE_WIDTH) / out_channels < 1.2:
            output_layout = ttnn.TILE_LAYOUT
        else:
            output_layout = ttnn.ROW_MAJOR_LAYOUT

        self.configuration = ConvConfiguration(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding_final,
            dilation=dilation,
            groups=groups,
            activation=_normalize_conv2d_activation(activation),
            dtype=dtype or ttnn.bfloat16,
            output_layout=output_layout,
            deallocate_input=deallocate_input,
        )

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        key: str,
        module_prefix: str | None = None,
    ) -> None:
        if module_prefix is None:
            module_prefix = ""
        base_key = f"{module_prefix}{key}" if module_prefix else key
        weight_key = f"{base_key}.weight"
        bias_key = f"{base_key}.bias"

        reshaped_weight = state_dict[weight_key].reshape(
            self.configuration.out_channels,
            self.configuration.in_channels // self.configuration.groups,
            1,
            self.configuration.kernel_size,
        )
        # Keep a torch-reference copy for internal F.conv1d parity check.
        self.torch_weight = state_dict[weight_key].detach().to(torch.float32).contiguous()
        bias = state_dict[bias_key] if bias_key in state_dict and state_dict[bias_key] is not None else None
        self.torch_bias = None if bias is None else bias.detach().to(torch.float32).contiguous()
        self.weight_tensor = ttnn.from_torch(reshaped_weight, dtype=ttnn.bfloat16)
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
        batch_size, input_length, _ = input_tensor.shape
        conv2d_config, slice_config, compute_config = get_conv_configs(
            input_tensor.shape, self.configuration, self.device
        )
        out, [self.weight_tensor, self.bias_tensor] = ttnn.conv2d(
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
            slice_config=slice_config,
        )
        if out.shape[2] > output_length_from_input_length(input_length, self.configuration):
            out = out[:, :, : output_length_from_input_length(input_length, self.configuration), :]
        return ttnn.squeeze(out, dim=1)

    def _check_against_torch(self, input_tensor: ttnn.Tensor, tt_output: ttnn.Tensor) -> None:
        # Compare TT Conv1d output against torch.nn.functional.conv1d reference.
        torch_input = ttnn.to_torch(ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)).to(torch.float32)
        if not hasattr(self, "torch_weight"):
            raise ValueError("Conv1d torch reference weight is not initialized. Call load_state_dict first.")
        torch_weight = self.torch_weight
        torch_bias = self.torch_bias

        # TT interface uses NLC, while torch conv1d expects NCL.
        torch_input_ncl = torch_input.permute(0, 2, 1).contiguous()
        pad_left, pad_right = self.configuration.padding
        if pad_left != 0 or pad_right != 0:
            torch_input_ncl = F.pad(torch_input_ncl, (pad_left, pad_right))
        torch_ref = F.conv1d(
            torch_input_ncl,
            torch_weight,
            bias=torch_bias,
            stride=self.configuration.stride,
            padding=0,
            dilation=self.configuration.dilation,
            groups=self.configuration.groups,
        )
        torch_ref_nlc = torch_ref.permute(0, 2, 1).contiguous()
        tt_output_torch = ttnn.to_torch(ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)).to(torch.float32)
        assert_with_pcc(torch_ref_nlc, tt_output_torch, pcc=0.99)

    def deallocate(self) -> None:
        if self.weight_tensor is not None:
            ttnn.deallocate(self.weight_tensor)
            self.weight_tensor = None
        if self.bias_tensor is not None:
            ttnn.deallocate(self.bias_tensor)
            self.bias_tensor = None
