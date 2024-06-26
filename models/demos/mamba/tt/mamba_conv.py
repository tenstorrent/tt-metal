# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch

import ttnn
import torch.nn as nn


class MambaConv:
    def __init__(self, device, args, load_fn, conv1d_weight_name):
        self.device = device
        self.args = args
        self.load_fn = load_fn
        self.conv1d_weight_name = conv1d_weight_name
        self.weights_dtype = ttnn.bfloat8_b
        self.activations_dtype = ttnn.bfloat16
        self.output_dtype = ttnn.bfloat8_b
        self.math_fidelity = ttnn.MathFidelity.LoFi
        self.prepare_conv_config()
        self.prepare_weights()

    def prepare_weights(self):
        # conv1d_weights (2E, 1, 4)->(1, 2E, 4)->(1, 2E, 4, 1)->(N, 2E, 4, 1)
        torch_conv1d_weights = self.load_fn(
            self.conv1d_weight_name,
            lambda x: x.transpose(1, 0).unsqueeze(-1).repeat(self.args.batch_size * self.args.seq_len, 1, 1, 1),
            return_as_torch=True,
        )
        self.tt_weight_tensor = ttnn.from_torch(
            torch_conv1d_weights, self.weights_dtype if self.weights_dtype != ttnn.bfloat8_b else ttnn.float32
        )

    def prepare_conv_config(self):
        self.input_channels = self.args.d_inner
        self.output_channels = 1
        self.input_height = self.args.seq_len
        self.input_width = 1
        self.filter_height = 4
        self.filter_width = 1
        self.stride_h = 1
        self.stride_w = 1
        self.pad_h = 3
        self.pad_w = 0

        self.conv_config = ttnn.DepthwiseConv1dConfig(
            dtype=self.output_dtype,
            weights_dtype=self.weights_dtype,
            math_fidelity=self.math_fidelity,
            height_sharding=True,
            input_channels_alignment=32,
            deallocate_activation=True,
        )

    def prepare_input(self, input_tensor):
        # input_tensor (1, 1, B, 2E)
        # convert from tile layout to row major layout
        # input_tensor = ttnn.to_dtype(input_tensor, self.activations_dtype)
        return ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)

    def __call__(self, input_tensor):
        [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.depthwise_conv1d(
            input_tensor=self.prepare_input(input_tensor),
            weight_tensor=self.tt_weight_tensor,
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            device=self.device,
            bias_tensor=None,
            kernel_size=(self.filter_height, self.filter_width),
            stride=(self.stride_h, self.stride_w),
            padding=(self.pad_h, self.pad_w),
            batch_size=self.args.batch_size,
            input_height=self.input_height,
            input_width=self.input_width,
            conv_config=self.conv_config,
            conv_op_cache={},
            debug=False,
            groups=self.input_channels,
        )
        self.tt_weight_tensor = weights_device

        return tt_output_tensor_on_device
