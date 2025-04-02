# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import ttnn


class TtTimestepEmbedding(nn.Module):
    def __init__(self, device, state_dict, module_path):
        super().__init__()

        self.device = device
        weights_1 = state_dict[f"{module_path}.linear_1.weight"].unsqueeze(0).unsqueeze(0)
        bias_1 = state_dict[f"{module_path}.linear_1.bias"]

        weights_2 = state_dict[f"{module_path}.linear_2.weight"].unsqueeze(0).unsqueeze(0)
        bias_2 = state_dict[f"{module_path}.linear_2.bias"]

        self.tt_weights_1 = ttnn.from_torch(
            torch.permute(weights_1, (0, 1, 3, 2)), ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )
        self.tt_bias_1 = (
            ttnn.from_torch(bias_1, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            if bias_1 is not None
            else None
        )

        self.tt_weights_2 = ttnn.from_torch(
            torch.permute(weights_2, (0, 1, 3, 2)), ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )
        self.tt_bias_2 = (
            ttnn.from_torch(bias_2, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            if bias_1 is not None
            else None
        )

    def forward(self, sample):
        sample = ttnn.linear(sample, self.tt_weights_1, bias=self.tt_bias_1, activation="silu")
        sample = ttnn.linear(
            sample,
            self.tt_weights_2,
            bias=self.tt_bias_2,
        )
        return sample
