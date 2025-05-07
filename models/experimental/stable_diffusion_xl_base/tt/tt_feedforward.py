# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn

from models.experimental.stable_diffusion_xl_base.tt.tt_geglu import TtGEGLU
from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params


class TtFeedForward(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        weights_dtype=ttnn.bfloat16,
    ):
        super().__init__()

        self.device = device
        self.tt_geglu = TtGEGLU(device, state_dict, f"{module_path}.net.0", weights_dtype=weights_dtype)

        weights = state_dict[f"{module_path}.net.2.weight"].unsqueeze(0).unsqueeze(0)
        bias = state_dict[f"{module_path}.net.2.bias"]

        self.tt_weights, self.tt_bias = prepare_linear_params(device, weights, bias, weights_dtype)

    def forward(self, hidden_states):
        hidden_states = self.tt_geglu(hidden_states)
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_weights,
            bias=self.tt_bias,
        )

        return hidden_states
