# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import ttnn

from models.experimental.stable_diffusion_xl_base.ttnn_impl.tt_geglu import TtGEGLU


class TtFeedForward(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
    ):
        super().__init__()

        self.device = device
        self.tt_geglu = TtGEGLU(device, state_dict, f"{module_path}.net.0")

        weights = state_dict[f"{module_path}.net.2.weight"].unsqueeze(0).unsqueeze(0)
        bias = state_dict[f"{module_path}.net.2.bias"]

        self.tt_weights = ttnn.from_torch(
            torch.permute(weights, (0, 1, 3, 2)), ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )
        self.tt_bias = (
            ttnn.from_torch(bias, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT) if bias is not None else None
        )

    def forward(self, hidden_states):
        hidden_states = self.tt_geglu(hidden_states)
        hidden_states = ttnn.linear(
            hidden_states,
            self.tt_weights,
            bias=self.tt_bias,
        )

        return hidden_states
