# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

import torch

import ttnn
from models.demos.vision.generative.stable_diffusion.wormhole.sd_helper_funcs import (
    reshard_for_output_channels_divisibility,
)
from models.demos.vision.generative.stable_diffusion.wormhole.tt.ttnn_functional_utility_functions import (
    get_default_compute_config,
    permute_conv_parameters,
    weight_to_bfp8,
)

config_override = {
    (320, 320, 64, 64): {"act_block_h": 32 * 16},
    (640, 1920, 32, 32): {"act_block_h": 32 * 4},
    (640, 1280, 32, 32): {"act_block_h": 32 * 4},
    (320, 960, 64, 64): {"act_block_h": 32 * 4},
    (320, 640, 64, 64): {"act_block_h": 32 * 8},
    (640, 640, 32, 32): {"act_block_h": 32 * 4},
}


class resnetBlock2D:
    def __init__(
        self,
        device,
        parameters,
        batch_size,
        input_height,
        input_width,
        compute_kernel_config,
        group_norm_on_device=True,
    ):
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.device = device
        self.parameters = parameters
        self.compute_kernel_config = compute_kernel_config
        self.group_norm_on_device = group_norm_on_device

        # Correctly extract in_channels and out_channels
        conv1_weight_shape = parameters.conv1.weight.shape
        self.in_channels = conv1_weight_shape[1]  # Input channels
        self.out_channels = conv1_weight_shape[0]  # Output channels

        # Permute and convert weights for conv1
        (
            self.conv1_weight,
            self.conv1_bias,
        ) = permute_conv_parameters(parameters.conv1.weight, parameters.conv1.bias)
        self.conv1_weight = weight_to_bfp8(self.conv1_weight)
        self.conv1_bias = ttnn.from_torch(
            self.conv1_bias,
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Permute and convert weights for conv2
        conv2_weight_shape = parameters.conv2.weight.shape
        self.conv2_in_channels = conv2_weight_shape[1]
        self.conv2_out_channels = conv2_weight_shape[0]
        (
            self.conv2_weight,
            self.conv2_bias,
        ) = permute_conv_parameters(parameters.conv2.weight, parameters.conv2.bias)
        self.conv2_weight = weight_to_bfp8(self.conv2_weight)
        self.conv2_bias = ttnn.from_torch(
            self.conv2_bias,
            dtype=ttnn.bfloat16,
            device=self.device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Initialize groupnorm parameters if used on device
        if self.group_norm_on_device:
            self.norm1_weight = ttnn.from_torch(
                parameters.norm1.weight,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.norm1_bias = ttnn.from_torch(
                parameters.norm1.bias,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.norm2_weight = ttnn.from_torch(
                parameters.norm2.weight,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.norm2_bias = ttnn.from_torch(
                parameters.norm2.bias,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Time embedding projection if exists
        if hasattr(parameters, "time_emb_proj") and parameters.time_emb_proj is not None:
            self.time_emb_proj_weight = ttnn.from_torch(
                parameters.time_emb_proj.weight,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.time_emb_proj_bias = ttnn.from_torch(
                parameters.time_emb_proj.bias,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.time_emb_proj_weight = None
            self.time_emb_proj_bias = None

        # Shortcut conv if needed
        if hasattr(parameters, "conv_shortcut") and parameters.conv_shortcut is not None:
            shortcut_weight, shortcut_bias = permute_conv_parameters(
                parameters.conv_shortcut.weight, parameters.conv_shortcut.bias
            )
            self.conv_shortcut_weight = weight_to_bfp8(shortcut_weight)
            self.conv_shortcut_bias = ttnn.from_torch(
                shortcut_bias,
                dtype=ttnn.bfloat16,
                device=self.device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.conv_shortcut_weight = None
            self.conv_shortcut_bias = None

    def __call__(self, input_tensor, temb=None):
        hidden_tensor = input_tensor

        # GroupNorm + Swish for first conv
        if self.group_norm_on_device:
            hidden_tensor = ttnn.group_norm(
                hidden_tensor,
                weight=self.norm1_weight,
                bias=self.norm1_bias,
                epsilon=1e-06,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            raise NotImplementedError("Host-based group norm not supported in this implementation")

        hidden_tensor = ttnn.mul(hidden_tensor, ttnn.erf(hidden_tensor))
        hidden_tensor = ttnn.mul(hidden_tensor, 0.5)

        # First convolution
        _, _, height, width = hidden_tensor.shape
        conv1_output_shape = (self.batch_size, self.out_channels, height, width)

        hidden_tensor = ttnn.conv2d(
            input_tensor=hidden_tensor,
            weight=self.conv1_weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=self.conv1_bias,
            conv_config=ttnn.Conv2dConfig(
                dtype=ttnn.bfloat8_b,
                weights_dtype=ttnn.bfloat8_b,
                math_fidelity=ttnn.MathFidelity.LoFi,
                activation="silu",
            ),
            compute_kernel_config=self.compute_kernel_config,
            conv_input_face_shape=(height, width),
            output_tensor_shape=conv1_output_shape,
            reshard_if_necessary=True,
            deallocate_activation=True,
        )

        # Add time embedding if present
        if temb is not None and self.time_emb_proj_weight is not None:
            temb = ttnn.linear(
                temb,
                self.time_emb_proj_weight,
                bias=self.time_emb_proj_bias,
                compute_kernel_config=self.compute_kernel_config,
            )
            temb = ttnn.reshape(temb, (self.batch_size, 1, 1, -1))
            hidden_tensor = ttnn.add(hidden_tensor, temb)

        # Second groupnorm + swish
        if self.group_norm_on_device:
            hidden_tensor = ttnn.group_norm(
                hidden_tensor,
                weight=self.norm2_weight,
                bias=self.norm2_bias,
                epsilon=1e-06,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        hidden_tensor = ttnn.mul(hidden_tensor, ttnn.erf(hidden_tensor))
        hidden_tensor = ttnn.mul(hidden_tensor, 0.5)

        # Second convolution
        _, _, height, width = hidden_tensor.shape
        conv2_output_shape = (self.batch_size, self.conv2_out_channels, height, width)

        hidden_tensor = ttnn.conv2d(
            input_tensor=hidden_tensor,
            weight=self.conv2_weight,
            in_channels=self.conv2_in_channels,
            out_channels=self.conv2_out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=self.conv2_bias,
            conv_config=ttnn.Conv2dConfig(
                dtype=ttnn.bfloat8_b,
                weights_dtype=ttnn.bfloat8_b,
                math_fidelity=ttnn.MathFidelity.LoFi,
                activation="silu",
            ),
            compute_kernel_config=self.compute_kernel_config,
            conv_input_face_shape=(height, width),
            output_tensor_shape=conv2_output_shape,
            reshard_if_necessary=True,
            deallocate_activation=True,
        )

        # Shortcut connection
        if self.conv_shortcut_weight is not None:
            _, _, height, width = input_tensor.shape
            shortcut_output_shape = (self.batch_size, self.conv2_out_channels, height, width)
            residual = ttnn.conv2d(
                input_tensor=input_tensor,
                weight=self.conv_shortcut_weight,
                in_channels=self.in_channels,
                out_channels=self.conv2_out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=self.conv_shortcut_bias,
                conv_config=ttnn.Conv2dConfig(
                    dtype=ttnn.bfloat8_b,
                    weights_dtype=ttnn.bfloat8_b,
                    math_fidelity=ttnn.MathFidelity.LoFi,
                ),
                compute_kernel_config=self.compute_kernel_config,
                conv_input_face_shape=(height, width),
                output_tensor_shape=shortcut_output_shape,
                reshard_if_necessary=True,
                deallocate_activation=True,
            )
        else:
            residual = input_tensor

        # Final add
        hidden_tensor = ttnn.add(hidden_tensor, residual)

        return hidden_tensor