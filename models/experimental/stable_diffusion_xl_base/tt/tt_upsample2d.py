# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import (
    prepare_conv_params,
)


class TtUpsample2D(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        stride,
        padding,
        dilation,
        groups,
        model_config,
    ):
        super().__init__()

        self.device = device

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.scale_factor = 2  # fixed number for now

        weights = state_dict[f"{module_path}.conv.weight"]
        bias = state_dict[f"{module_path}.conv.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        self.conv_config = model_config.get_conv_config(conv_path=module_path)
        self.compute_config, self.tt_weights, self.tt_bias, self.conv_params = prepare_conv_params(
            device,
            weights,
            bias,
            self.conv_config.weights_dtype,
            fp32_dest_acc_en=(self.conv_config.weights_dtype == ttnn.bfloat8_b)
            and (self.conv_config.shard_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
        )

    def interpolate(self, hidden_states):
        memory_config = ttnn.create_sharded_memory_config(
            shape=hidden_states.shape,
            core_grid=ttnn.CoreGrid(y=8, x=5),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        hidden_states = ttnn.to_memory_config(hidden_states, memory_config)
        hidden_states = ttnn.upsample(hidden_states, (self.scale_factor, self.scale_factor))
        B, H, W, C = list(hidden_states.shape)
        return hidden_states, [B, C, H, W]

    def forward(self, hidden_states):
        hidden_states, input_shape = self.interpolate(hidden_states)
        B, C, H, W = input_shape

        [hidden_states, [H, W], [self.tt_weights, self.tt_bias]] = ttnn.conv2d(
            input_tensor=hidden_states,
            weight_tensor=self.tt_weights,
            in_channels=self.conv_params["input_channels"],
            out_channels=self.conv_params["output_channels"],
            device=self.device,
            bias_tensor=self.tt_bias,
            kernel_size=self.conv_params["kernel_size"],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            batch_size=B,
            input_height=H,
            input_width=W,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            memory_config=None,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        C = self.conv_params["output_channels"]

        hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
        return hidden_states, [C, H, W]
