# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtSqueezeExcitation(torch.nn.Module):
    def __init__(
        self,
        config,
        in_channels: int,
        fc_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        state_dict=None,
        base_address="",
        device=None,
    ) -> None:
        super().__init__()
        self.device = device
        self.in_channels = in_channels
        self.fc_channels = fc_channels

        # Convert weights to TTNN tensors
        weight_fc1 = state_dict[f"{base_address}.fc1.weight"]
        bias_fc1 = state_dict[f"{base_address}.fc1.bias"]
        self.fc1_weight = ttnn.from_torch(
            weight_fc1,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.fc1_bias = ttnn.from_torch(
            bias_fc1,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        weight_fc2 = state_dict[f"{base_address}.fc2.weight"]
        bias_fc2 = state_dict[f"{base_address}.fc2.bias"]
        self.fc2_weight = ttnn.from_torch(
            weight_fc2,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.fc2_bias = ttnn.from_torch(
            bias_fc2,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

        self.activation = ttnn.relu
        self.scale_activation = ttnn.hardsigmoid

    def forward(self, input: ttnn.Tensor) -> ttnn.Tensor:
        # Global average pooling
        scale = ttnn.global_avg_pool2d(input)

        # First fully connected layer
        batch_size = scale.shape[0]
        scale = ttnn.linear(
            scale,
            self.fc1_weight,
            bias=self.fc1_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scale = self.activation(scale)

        # Second fully connected layer
        scale = ttnn.linear(
            scale,
            self.fc2_weight,
            bias=self.fc2_bias,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scale = self.scale_activation(scale)

        # Element-wise multiplication
        final_out = ttnn.multiply(input, scale)
        return final_out
