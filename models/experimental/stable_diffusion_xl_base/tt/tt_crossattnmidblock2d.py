# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from models.experimental.stable_diffusion_xl_base.tt.tt_transformermodel import TtTransformer2DModel
from models.experimental.stable_diffusion_xl_base.tt.tt_resnetblock2d import TtResnetBlock2D


class TtUNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
        query_dim,
        num_attn_heads,
        out_dim,
    ):
        super().__init__()

        num_layers_attn = 1
        num_layers_resn = num_layers_attn + 1
        self.attentions = []
        self.resnets = []

        for i in range(num_layers_attn):
            self.attentions.append(
                TtTransformer2DModel(
                    device,
                    state_dict,
                    f"{module_path}.attentions.{i}",
                    model_config,
                    query_dim,
                    num_attn_heads,
                    out_dim,
                )
            )

        for i in range(num_layers_resn):
            self.resnets.append(
                TtResnetBlock2D(device, state_dict, f"{module_path}.resnets.{i}", model_config=model_config)
            )

    def forward(self, input_tensor, input_shape, temb=None, encoder_hidden_states=None, attention_mask=None):
        B, C, H, W = input_shape
        hidden_states = input_tensor

        hidden_states, [C, H, W] = self.resnets[0].forward(hidden_states, temb, [B, C, H, W])

        tt_blocks = list(zip(self.resnets[1:], self.attentions))
        for resnet, attn in tt_blocks:
            hidden_states = attn.forward(hidden_states, [B, C, H, W], encoder_hidden_states=encoder_hidden_states)
            hidden_states, [C, H, W] = resnet.forward(hidden_states, temb, [B, C, H, W])

        return hidden_states, [C, H, W]
