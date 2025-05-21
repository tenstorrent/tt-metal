# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


class TtGEGLU(nn.Module):
    def __init__(self, device, state_dict, module_path, weights_dtype=ttnn.bfloat16):
        super().__init__()

        self.device = device
        weights = state_dict[f"{module_path}.proj.weight"].unsqueeze(0).unsqueeze(0)
        bias = state_dict[f"{module_path}.proj.bias"]

        self.tt_weights, self.tt_bias = prepare_linear_params(device, weights, bias, weights_dtype)

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
