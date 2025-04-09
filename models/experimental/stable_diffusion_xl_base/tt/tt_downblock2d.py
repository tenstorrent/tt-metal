# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from models.experimental.stable_diffusion_xl_base.tt.tt_resnetblock2d import TtResnetBlock2D
from models.experimental.stable_diffusion_xl_base.tt.tt_downsample2d import TtDownsample2D


class TtDownBlock2D(nn.Module):
    def __init__(self, device, state_dict, module_path):
        super().__init__()

        num_layers = 2
        self.resnets = []

        for i in range(num_layers):
            self.resnets.append(TtResnetBlock2D(device, state_dict, f"{module_path}.resnets.{i}"))

        self.downsamplers = TtDownsample2D(
            device, state_dict, f"{module_path}.downsamplers.0", (2, 2), (1, 1), (1, 1), 1
        )

    def forward(self, hidden_states, temb, input_shape):
        B, C, H, W = input_shape
        output_states = ()

        for resnet in self.resnets:
            hidden_states, [C, H, W] = resnet.forward(hidden_states, temb, [B, C, H, W])
            output_states = output_states + (hidden_states,)

        hidden_states, [C, H, W] = self.downsamplers.forward(hidden_states, [B, C, H, W])
        output_states = output_states + (hidden_states,)
        return hidden_states, [C, H, W], output_states
