# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import ttnn


class TtGEGLU(nn.Module):
    def __init__(self, device, state_dict, module_path):
        super().__init__()

        self.device = device
        weights = state_dict[f"{module_path}.proj.weight"].unsqueeze(0).unsqueeze(0)
        bias = state_dict[f"{module_path}.proj.bias"]

        self.tt_weights = ttnn.from_torch(
            torch.permute(weights, (0, 1, 3, 2)), ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )
        self.tt_bias = (
            ttnn.from_torch(bias, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT) if bias is not None else None
        )

    def forward(self, hidden_states):
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_weights,
            bias=self.tt_bias,
        )
        gate = hidden_states[:, :, :, hidden_states.shape[3] // 2 :]
        hidden_states = hidden_states[:, :, :, : hidden_states.shape[3] // 2]

        # ttnn.split not working properly
        # hidden_states, gate = ttnn.split(hidden_states, ceil(hidden_states.shape[3] / 2), 3)

        return ttnn.multiply(hidden_states, ttnn.gelu(gate))
