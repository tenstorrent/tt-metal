# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from typing import Tuple
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_conv_params,
    prepare_split_conv_params,
    split_conv2d,
)


@dataclass
class ConvConfig:
    split_conv: bool = False
    split_in: int = 1
    split_out: int = 1
    # conv_act_dtype, also output_dtype in tests
    conv_output_dtype = ttnn.bfloat16
    # weights_dtype
    conv_w_dtype = ttnn.bfloat16

    groups: int = 1
    kernel_size: Tuple[int, int] = (3, 3)
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (1, 1)
    dilation: Tuple[int, int] = (1, 1)
    act_block_h_override: int = 0
    act_block_w_div: int = 1
    deallocate_activation: bool = True
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    fp32_accum: bool = True
    packer_l1_acc: bool = False
    enable_split_reader: bool = False
    split_input_channels_factor: int = 1
    split_output_channels_factor: int = 1
    act_db: bool = False
    w_db: bool = False

    conv2d_config: ttnn.Conv2dConfig = None
    compute_config: ttnn.WormholeComputeKernelConfig = None

    def __post_init__(self):
        if self.compute_config is None:
            self.compute_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=self.math_fidelity,
                fp32_dest_acc_en=self.fp32_accum,
                math_approx_mode=False,
                packer_l1_acc=self.packer_l1_acc,
            )
        if self.conv2d_config is None:
            self.conv2d_config = ttnn.Conv2dConfig(
                weights_dtype=self.conv_w_dtype,
                deallocate_activation=self.deallocate_activation,
                act_block_w_div=self.act_block_w_div,
                act_block_h_override=self.act_block_h_override,
            )


def make_conv_config(
    kernel: Tuple[int, int] = (3, 3),
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (1, 1),
    split_in: int = 1,
) -> ConvConfig:
    return ConvConfig(kernel_size=kernel, stride=stride, padding=padding, split_in=split_in)


class ConvolutionLayer:
    def __init__(self, device, weights, bias, conv_config=make_conv_config()):
        self.device = device
        self.conv_config = conv_config
        self.split_in = conv_config.split_in
        self.split_out = conv_config.split_out

        self.split_conv = self.split_in > 1 or self.split_out > 1

        # Prepare conv parameters
        if self.split_conv:
            self.tt_weights, self.tt_bias, self.conv_params = prepare_split_conv_params(
                weights, bias, conv_config.conv_w_dtype, self.split_in, self.split_out
            )
        else:
            self.tt_weights, self.tt_bias, self.conv_params = prepare_conv_params(
                weights, bias, conv_config.conv_w_dtype
            )

    def forward(self, hidden_states, B, C, H, W):
        if self.split_conv:
            return self._apply_split_conv(hidden_states, B, C, H, W)
        else:
            return self._apply_regular_conv(hidden_states, B, C, H, W)

    def _apply_split_conv(self, hidden_states, B, C, H, W):
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states, [C, H, W], [self.tt_weights, self.tt_bias] = split_conv2d(
            device=self.device,
            hidden_states=hidden_states,
            input_shape=[B, C, H, W],
            conv_weights=self.tt_weights,
            conv_bias=self.tt_bias,
            split_in=self.split_in,
            split_out=self.split_out,
            compute_config=self.conv_config.compute_config,
            conv_config=self.conv_config.conv2d_config,
            conv_params=self.conv_params,
            conv_dtype=self.conv_config.conv_output_dtype,
            stride=self.conv_config.stride,
            padding=self.conv_config.padding,
            dilation=self.conv_config.dilation,
            groups=self.conv_config.groups,
        )
        return hidden_states, [C, H, W]

    def _apply_regular_conv(self, hidden_states, B, C, H, W):
        [hidden_states, [H, W], [self.tt_weights, self.tt_bias]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.tt_weights,
            in_channels=self.conv_params["input_channels"],
            out_channels=self.conv_params["output_channels"],
            device=self.device,
            bias_tensor=self.tt_bias,
            kernel_size=self.conv_params["kernel_size"],
            stride=self.conv_config.stride,
            padding=self.conv_config.padding,
            dilation=self.conv_config.dilation,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv_config.conv2d_config,
            compute_config=self.conv_config.compute_config,
            groups=self.conv_config.groups,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.conv_config.conv_output_dtype,
        )
        C = self.conv_params["output_channels"]
        return hidden_states, [C, H, W]
