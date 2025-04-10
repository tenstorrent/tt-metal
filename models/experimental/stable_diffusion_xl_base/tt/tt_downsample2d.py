# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_conv_params


class TtDownsample2D(nn.Module):
    def __init__(self, device, state_dict, module_path, stride, padding, dilation, groups):
        super().__init__()

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        weights = state_dict[f"{module_path}.conv.weight"]
        bias = state_dict[f"{module_path}.conv.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        self.compute_config, self.conv_config, self.tt_weights, self.tt_bias = prepare_conv_params(
            device, weights, bias, ttnn.bfloat8_b
        )

        self.input_channels = self.tt_weights.shape[1]
        self.output_channels = self.tt_weights.shape[0]
        self.kernel_w = self.tt_weights.shape[2]
        self.kernel_h = self.tt_weights.shape[3]

    def forward(self, hidden_states, input_shape):
        B, C, H, W = input_shape

        [tt_output_tensor_on_device, [out_height, out_width], [d_w, d_b]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.tt_weights,
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            device=self.device,
            bias_tensor=self.tt_bias,
            kernel_size=(self.kernel_h, self.kernel_w),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=1,
            memory_config=None,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG)
        return tt_output_tensor_on_device, [self.output_channels, out_height, out_width]
