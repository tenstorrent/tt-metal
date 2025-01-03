# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
from typing import Optional
import torch.nn as nn
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    run_ttnn_conv_with_pre_and_post_tensor_formatting,
)
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import conv_cache

import math


def permute_conv_parameters(weight, bias):
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_torch(bias)
    return weight, bias


config_override = {
    (320, 320, 64, 64): {"act_block_h": 64},
    (640, 640, 32, 32): {"act_block_h": 64},
    (640, 1920, 32, 32): {"act_block_h": 32},
    (640, 1280, 32, 32): {"act_block_h": 32},
    (1280, 1920, 16, 16): {"act_block_h": 32},
    (1280, 1280, 32, 32): {"act_block_h": 32},
    (320, 960, 64, 64): {"act_block_h": 32},
    (640, 960, 32, 32): {"act_block_h": 32},
    (320, 640, 64, 64): {"act_block_h": 32},
    (640, 640, 64, 64): {"act_block_h": 64},
    (640, 320, 64, 64): {"act_block_h": 64},
}


class downsample_2d:
    def __init__(
        self,
        device,
        parameters,
        reader_patterns_cache,
        batch_size,
        input_height,
        input_width,
        compute_kernel_config,
    ):
        self.device = device
        self.parameters = parameters
        parameters.conv.weight, parameters.conv.bias = permute_conv_parameters(
            parameters.conv.weight, parameters.conv.bias
        )
        parameters.conv.bias = torch.reshape(parameters.conv.bias, (1, 1, 1, parameters.conv.bias.shape[-1]))

        self.conv_weights = ttnn.from_torch(parameters.conv.weight, ttnn.float32)
        self.conv_bias = ttnn.from_torch(parameters.conv.bias, ttnn.float32)
        self.input_height = input_height
        self.input_width = input_width
        self.batch_size = batch_size
        self.out_channels = parameters.conv.weight.shape[0]
        self.in_channels = parameters.conv.weight.shape[1]
        self.conv_config_override = {}
        if (self.out_channels, self.in_channels, input_height, input_width) in config_override:
            self.conv_config_override = config_override[
                (self.out_channels, self.in_channels, input_height, input_width)
            ]

        self.stride = 2

        self.output_height = ttnn.get_conv_output_dim(input_height, 3, self.stride, 1)
        self.output_width = ttnn.get_conv_output_dim(input_width, 3, self.stride, 1)
        self.shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if self.in_channels < 320 else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )

        self.input_memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
            tensor_shape=ttnn.Shape(
                [
                    1,
                    1,
                    self.batch_size * self.input_height * self.input_width,
                    self.out_channels,
                ]
            ),
            parallel_config=ttnn._ttnn.operations.conv.determine_parallel_config(
                shard_layout=self.shard_layout,
                batch_size=self.batch_size,
                input_channels=self.in_channels,
                output_height=self.output_height,
                output_width=self.output_width,
                output_channels=self.out_channels,
                compute_grid_size=self.device.compute_with_storage_grid_size(),
                block_shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
                enable_channels_padding=False,
                is_out_tiled=True,
            ),
            tile_size=32,
        )

    def __call__(
        self,
        in_channels,
        hidden_states,
        use_conv=False,
        out_channels=None,
        padding=1,
        dtype: Optional[ttnn.DataType] = None,
    ):
        stride = 2
        assert padding == 1

        assert hidden_states.shape[-1] == in_channels

        if use_conv and padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = ttnn.pad(hidden_states, pad, value=0)

        # if ttnn.get_memory_config(hidden_states) != self.conv.conv.input_sharded_memory_config:
        #     hidden_states = ttnn.to_memory_config(hidden_states, self.conv.conv.input_sharded_memory_config)
        # hidden_states = self.conv(hidden_states)
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat8_b,
            activation="",
            shard_layout=self.shard_layout,
            input_channels_alignment=32,
            transpose_shards=False,
            reshard_if_not_optimal=False,
        )

        if hidden_states.memory_config() != self.input_memory_config:
            hidden_states = ttnn.to_memory_config(hidden_states, self.input_memory_config)

        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        if self.conv_config_override and "act_block_h" in self.conv_config_override:
            conv_config.act_block_h_override = self.conv_config_override["act_block_h"]

        conv_kwargs = {
            "in_channels": in_channels,
            "out_channels": self.out_channels,
            "batch_size": hidden_states.shape[0],
            "input_height": hidden_states.shape[1],
            "input_width": hidden_states.shape[2],
            "kernel_size": (3, 3),
            "stride": (self.stride, self.stride),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": self.device,
            "conv_config": conv_config,
        }

        if not ttnn.is_tensor_storage_on_device(self.conv_weights):
            self.conv_weights = ttnn.prepare_conv_weights(
                weight_tensor=self.conv_weights,
                weights_format="OIHW",
                input_memory_config=hidden_states.memory_config(),
                input_layout=hidden_states.get_layout(),
                **conv_kwargs,
            )
            self.conv_bias = (
                ttnn.prepare_conv_bias(
                    bias_tensor=self.conv_bias,
                    input_memory_config=hidden_states.memory_config(),
                    input_layout=hidden_states.get_layout(),
                    **conv_kwargs,
                )
                if self.conv_bias is not None
                else None
            )
            self.conv_weights = ttnn.to_device(self.conv_weights, self.device)
            self.conv_bias = ttnn.to_device(self.conv_bias, self.device)

        [hidden_states, [self.conv_weights, self.conv_bias]] = ttnn.conv2d(
            input_tensor=hidden_states,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            stride=(self.stride, self.stride),
            padding=(1, 1),
            device=self.device,
            batch_size=self.batch_size,
            input_height=self.input_height,
            input_width=self.input_width,
            weight_tensor=self.conv_weights,
            bias_tensor=self.conv_bias,
            conv_config=conv_config,
            compute_config=compute_config,
            conv_op_cache=conv_cache,
            return_output_dim=False,
            return_weights_and_bias=True,
        )
        # hidden_states = run_ttnn_conv_with_pre_and_post_tensor_formatting(
        #     self.device,
        #     self.conv,
        #     hidden_states,
        #     self.conv.batch_size,
        #     self.conv.output_height,
        #     self.conv.output_width,
        #     self.conv.out_channels,
        # )

        return hidden_states
