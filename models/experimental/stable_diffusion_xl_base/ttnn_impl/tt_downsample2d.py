# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn


class TtDownsample2D(nn.Module):
    def __init__(self, device, state_dict, module_path, stride, padding, dilation, groups):
        super().__init__()

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        weights = state_dict[f"{module_path}.conv.weight"]
        bias = state_dict[f"{module_path}.conv.bias"]

        self.tt_weights = ttnn.from_torch(weights, ttnn.bfloat16)
        self.tt_bias = (
            ttnn.from_torch(bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), ttnn.bfloat16) if bias is not None else None
        )

        self.input_channels = self.tt_weights.shape[1]
        self.output_channels = self.tt_weights.shape[0]
        self.kernel_w = self.tt_weights.shape[2]
        self.kernel_h = self.tt_weights.shape[3]

    def forward(self, hidden_states, input_shape):
        B, C, H, W = input_shape

        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            shard_layout=None,
            input_channels_alignment=32,
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True,
            act_block_w_div=1,
            act_block_h_override=0,
            preprocess_weights_on_device=True,
            always_preprocess_weights=True,
            transpose_shards=True,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

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
            conv_config=conv_config,
            compute_config=compute_config,
            conv_op_cache={},
            debug=False,
            groups=1,
            memory_config=None,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        return tt_output_tensor_on_device, [self.output_channels, out_height, out_width]
