# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_resnetblock2d import TtResnetBlock2D


class TtUpBlock2D(nn.Module):
    def __init__(self, device, state_dict, module_path, model_config):
        super().__init__()

        num_layers = 3
        self.resnets = []

        for i in range(num_layers):
            self.resnets.append(
                TtResnetBlock2D(
                    device,
                    state_dict,
                    f"{module_path}.resnets.{i}",
                    model_config,
                    True,
                    2,
                )
            )

    def forward(self, hidden_states, res_hidden_states_tuple, input_shape, temb):
        B, C, H, W = input_shape

        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = ttnn.concat([hidden_states, res_hidden_states], dim=3)
            C = list(hidden_states.shape)[3]

            hidden_states, [C, H, W] = resnet.forward(hidden_states, temb, [B, C, H, W])

        return hidden_states, [C, H, W]
