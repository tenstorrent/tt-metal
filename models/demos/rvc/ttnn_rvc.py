// File: models/demos/rvc/ttnn_rvc.py
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: MIT

import torch
import ttnn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import json


class TtnnPosteriorEncoder:
    def __init__(self, device, state_dict, layer_name, model_config, dtype=ttnn.bfloat16):
        self.device = device
        self.state_dict = state_dict
        self.layer_name = layer_name
        self.model_config = model_config
        self.dtype = dtype

    def _conv1d(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        weight = ttnn.to_device(weight, self.device)
        if bias is not None:
            bias = ttnn.to_device(bias, self.device)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, [1, 1, x.shape[0], x.shape[1]])  # [B, C, H, W]
        weight = ttnn.reshape(weight, [weight.shape[0], weight.shape[1], 1, 1])
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            in_channels=weight.shape[1],
            out_channels=weight.shape[0],
            kernel_size=(1, 1),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            batch_size=1,
            input_height=x.shape[2],
            input_width=x.shape[3],
            device=self.device,
            use_1d_systolic_array=True,
            weight_dtype=self.dtype,
            output_dtype=self.dtype,
            reader_patterns_cache=self.model_config["reader_patterns_cache"],
            deallocate_activation=True,
            conv_op_cache=self.model_config["conv_cache"],
        )[0]
        x = ttnn.reshape(x, [x.shape[2], x.shape[3]])
        if bias is not None:
            x = ttnn.add(x, bias)
            ttnn.deallocate(bias)
        ttnn.deallocate(weight)
        return x

    def __call__(self, x, g):
        x = ttnn.to_device(x, self.device)
        g = ttnn.to_device(g, self.device)

        # Initial conv
        conv1_weight = self.state_dict[f"{self.layer_name}.pre_net.conv_0.weight"]
        conv1_bias = self.state_dict.get(f"{self.layer_name}.pre_net.conv_0.bias", None)
        x = self._conv1d(x, conv1_weight, conv1_bias, padding=1)
        x = ttnn.tanh(x)

        # Residual blocks
        for i in range(2):
            x_skip = x
            x = ttnn.tanh(x)
            x = self._conv1d(x, self.state_dict[f"{self.layer_name}.resblocks.{i}.conv1.weight"], padding=1)
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            g_conv = self._conv1d(g, self.state_dict[f"{self.layer_name}.resblocks.{i}.cond_layer_norm.weight"], bias=None)
            g_conv = ttnn.reshape(g_conv, [1, -1, 1])
            g_conv = ttnn.to_layout(g_conv, ttnn.TILE_LAYOUT)
            x = ttnn.add(x, g_conv)
            ttnn.deallocate(g_conv)
            x = ttnn.tanh(x)
            x = self._conv1d(x, self.state_dict[f"{self.layer_name}.resblocks.{i}.conv2.weight"], padding=1)
            x = ttnn.add(x, x_skip)
            ttnn.deallocate(x_skip)