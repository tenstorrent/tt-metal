# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_attention import TtAttention
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_resnetblock2d import TtResnetBlock2D


class TtUNetMidBlock2D(nn.Module):
    def __init__(self, device, state_dict, module_path, model_config):
        super().__init__()

        num_layers_attn = 1
        num_layers_resn = num_layers_attn + 1
        self.attentions = []
        self.resnets = []

        for i in range(num_layers_attn):
            self.attentions.append(
                TtAttention(device, state_dict, f"{module_path}.attentions.{i}", 512, 1, 512, None, 512)
            )

        for i in range(num_layers_resn):
            self.resnets.append(TtResnetBlock2D(device, state_dict, f"{module_path}.resnets.{i}", model_config))

    def forward(self, input_tensor, input_shape):
        B, C, H, W = input_shape
        hidden_states = input_tensor

        hidden_states, [C, H, W] = self.resnets[0].forward(hidden_states, [B, C, H, W])

        tt_blocks = list(zip(self.resnets[1:], self.attentions))
        for resnet, attn in tt_blocks:
            hidden_states = attn.forward(hidden_states, [B, C, H, W])
            hidden_states, [C, H, W] = resnet.forward(hidden_states, [B, C, H, W])

        return hidden_states, [C, H, W]
