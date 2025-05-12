# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_conv_block import ConvBlock
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_configs import GROUPNORM_EPSILON, GROUPNORM_GROUPS
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_utils import prepare_group_norm


class ResnetBlock:
    def __init__(
        self,
        torch_resnet,
        device,
        in_channels,
        input_height,
        input_width,
        out_channels,
        norm1_num_blocks=1,
        norm2_num_blocks=1,
        conv1_channel_split_factors=(1, 1),
        conv2_channel_split_factors=(1, 1),
    ):
        self.device = device
        self.in_channels = in_channels
        self.input_height = input_height
        self.input_width = input_width

        # groupnorm 1
        self.norm1_num_blocks = norm1_num_blocks
        self.norm1_grid_core = ttnn.CoreGrid(y=4, x=8) if in_channels == 128 else ttnn.CoreGrid(y=8, x=8)
        (
            self.norm1_input_mask,
            self.norm1_weights,
            self.norm1_bias,
        ) = prepare_group_norm(
            self.device,
            in_channels,
            self.norm1_grid_core,
            torch_resnet.norm1.weight,
            torch_resnet.norm1.bias,
        )

        # conv 1
        self.conv1 = ConvBlock(
            torch_resnet.conv1,
            device,
            in_channels,
            input_height,
            input_width,
            out_channels,
            conv1_channel_split_factors[0],
            conv1_channel_split_factors[1],
        )

        # groupnorm 2
        self.norm2_num_blocks = norm2_num_blocks
        self.norm2_grid_core = ttnn.CoreGrid(y=4, x=8) if out_channels == 128 else ttnn.CoreGrid(y=8, x=8)
        (
            self.norm2_input_mask,
            self.norm2_weights,
            self.norm2_bias,
        ) = prepare_group_norm(
            self.device,
            out_channels,
            self.norm2_grid_core,
            torch_resnet.norm2.weight,
            torch_resnet.norm2.bias,
        )

        # conv 2
        self.conv2 = ConvBlock(
            torch_resnet.conv2,
            device,
            out_channels,
            input_height,
            input_width,
            out_channels,
            conv2_channel_split_factors[0],
            conv2_channel_split_factors[1],
        )

        # conv shortcut
        self.has_conv_shortcut = False
        if torch_resnet.conv_shortcut:
            self.conv_shortcut = ConvBlock(
                torch_resnet.conv_shortcut,
                device,
                in_channels,
                input_height,
                input_width,
                out_channels,
                conv1_channel_split_factors[0],
                conv1_channel_split_factors[1],
                kernel_size=1,
                padding=0,
            )
            self.has_conv_shortcut = True

    def __call__(self, input_tensor):
        hidden_states = input_tensor

        # prepare groupnorm 1
        if hidden_states.shape[1] != 1:
            hidden_states = ttnn.reshape(hidden_states, [1, 1, self.input_height * self.input_width, self.in_channels])
        if hidden_states.dtype != ttnn.bfloat16:
            hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=GROUPNORM_GROUPS,
            input_mask=self.norm1_input_mask,
            weight=self.norm1_weights,
            bias=self.norm1_bias,
            epsilon=GROUPNORM_EPSILON,
            core_grid=self.norm1_grid_core,
            dtype=ttnn.bfloat16,
            inplace=False,
            num_out_blocks=self.norm1_num_blocks,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        hidden_states = ttnn.silu(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = ttnn.group_norm(
            hidden_states,
            num_groups=GROUPNORM_GROUPS,
            input_mask=self.norm2_input_mask,
            weight=self.norm2_weights,
            bias=self.norm2_bias,
            epsilon=GROUPNORM_EPSILON,
            core_grid=self.norm2_grid_core,
            dtype=ttnn.bfloat16,
            inplace=False,
            num_out_blocks=self.norm2_num_blocks,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        hidden_states = ttnn.silu(hidden_states)

        hidden_states = self.conv2(hidden_states)

        if self.has_conv_shortcut:
            input_tensor = self.conv_shortcut(input_tensor)

        if hidden_states.shape != input_tensor.shape:
            hidden_states = ttnn.reshape(hidden_states, input_tensor.shape)

        hidden_states = ttnn.add(hidden_states, input_tensor, output_tensor=hidden_states)
        input_tensor.deallocate(True)

        return hidden_states
