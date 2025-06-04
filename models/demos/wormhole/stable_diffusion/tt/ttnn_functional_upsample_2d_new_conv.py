# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_upsample_nearest_2d import upsample_nearest2d
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import (
    get_default_compute_config,
    permute_conv_parameters,
)

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
}


class upsample2d:
    def __init__(self, device, parameters, batch_size, input_height, input_width, compute_kernel_config):
        self.input_height = input_height
        self.input_width = input_width
        self.device = device
        self.parameters = parameters
        parameters.conv.weight, parameters.conv.bias = permute_conv_parameters(
            parameters.conv.weight, parameters.conv.bias
        )
        self.batch_size = batch_size
        self.scale_factor = 2

        out_channels = parameters.conv.weight.shape[0]
        in_channels = parameters.conv.weight.shape[1]
        self.upsample_nearest2d = upsample_nearest2d(input_height, input_width, in_channels, self.scale_factor)

        input_height = input_height * self.scale_factor
        input_width = input_width * self.scale_factor

        out_channels = parameters.conv.weight.shape[0]
        in_channels = parameters.conv.weight.shape[1]
        # breakpoint()
        parameters.conv.bias = torch.reshape(parameters.conv.bias, (1, 1, 1, out_channels))
        tt_weight_tensor = ttnn.from_torch(parameters.conv.weight, ttnn.float32)
        tt_bias_tensor = ttnn.from_torch(parameters.conv.bias, ttnn.float32)
        self.conv_config_override = {}
        if (out_channels, in_channels, input_height, input_width) in config_override:
            self.conv_config_override = config_override[(out_channels, in_channels, input_height, input_width)]

        self.conv_in_channels = in_channels
        self.conv_out_channels = out_channels
        self.conv_weight_tensor = tt_weight_tensor
        self.conv_bias_tensor = tt_bias_tensor
        self.conv_input_height = input_height
        self.conv_input_width = input_width

        self.output_height = ttnn.get_conv_output_dim(input_height, 3, 1, 1)
        self.output_width = ttnn.get_conv_output_dim(input_width, 3, 1, 1)
        logger.info(f"Upsample Output = {self.output_height}x{self.output_width}")

    def __call__(self, input, in_channels, out_channels):
        if input.layout == ttnn.TILE_LAYOUT:
            input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
        # # slice out batch
        input = ttnn.reshape(input, (2, self.input_height, self.input_width, input.shape[3]))
        tt_out = self.upsample_nearest2d(input)
        del input
        tt_out = ttnn.reshape(tt_out, (1, 1, tt_out.shape[0] * tt_out.shape[1] * tt_out.shape[2], tt_out.shape[3]))
        # if ttnn.get_memory_config(tt_out) != self.conv.conv.input_sharded_memory_config:
        #     tt_out = ttnn.to_memory_config(tt_out, self.conv.conv.input_sharded_memory_config)
        # tt_out = self.conv(tt_out)
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat8_b,
            activation="",
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            transpose_shards=False,
            reshard_if_not_optimal=False,  # Reshard has error : 1616 Bytes unique+common runtime args targeting kernel reshard_reader on (x=0,y=0) are too large. Cannot be written as they will run into memory region reserved for result. Max allowable size is 1024 Bytes
        )
        compute_config = get_default_compute_config(self.device)
        if self.conv_config_override and "act_block_h" in self.conv_config_override:
            conv_config.act_block_h_override = self.conv_config_override["act_block_h"]

        conv_kwargs = {
            "in_channels": self.conv_in_channels,
            "out_channels": self.conv_out_channels,
            "batch_size": self.batch_size,
            "input_height": self.conv_input_height,
            "input_width": self.conv_input_width,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": self.device,
            "conv_config": conv_config,
        }
        if not ttnn.is_tensor_storage_on_device(self.conv_weight_tensor):
            self.conv_weight_tensor = ttnn.prepare_conv_weights(
                weight_tensor=self.conv_weight_tensor,
                weights_format="OIHW",
                input_layout=tt_out.get_layout(),
                input_memory_config=tt_out.memory_config(),
                has_bias=True,
                **conv_kwargs,
            )
            self.conv_bias_tensor = ttnn.prepare_conv_bias(
                bias_tensor=self.conv_bias_tensor,
                input_memory_config=tt_out.memory_config(),
                input_layout=tt_out.get_layout(),
                **conv_kwargs,
            )

            self.conv_weight_tensor = ttnn.to_device(self.conv_weight_tensor, self.device)
            self.conv_bias_tensor = ttnn.to_device(self.conv_bias_tensor, self.device)

        tt_out = ttnn.conv2d(
            input_tensor=tt_out,
            weight_tensor=self.conv_weight_tensor,
            bias_tensor=self.conv_bias_tensor,
            **conv_kwargs,
            compute_config=compute_config,
        )
        return tt_out
