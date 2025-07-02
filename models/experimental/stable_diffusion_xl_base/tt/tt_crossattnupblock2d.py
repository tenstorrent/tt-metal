# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_transformermodel import TtTransformer2DModel
from models.experimental.stable_diffusion_xl_base.tt.tt_resnetblock2d import TtResnetBlock2D
from models.experimental.stable_diffusion_xl_base.tt.tt_upsample2d import TtUpsample2D


class TtCrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
        query_dim,
        num_attn_heads,
        out_dim,
        has_upsample=False,
    ):
        super().__init__()

        num_layers = 3
        self.attentions = []
        self.resnets = []
        self.device = device

        for i in range(num_layers):
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

        for i in range(num_layers):
            self.resnets.append(
                TtResnetBlock2D(
                    device,
                    state_dict,
                    f"{module_path}.resnets.{i}",
                    model_config=model_config,
                    conv_shortcut=True,
                )
            )

        self.upsamplers = (
            TtUpsample2D(
                device, state_dict, f"{module_path}.upsamplers.0", (1, 1), (1, 1), (1, 1), 1, model_config=model_config
            )
            if has_upsample
            else None
        )

    def forward(
        self,
        input_tensor,
        res_hidden_states_tuple,
        input_shape,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        B, C, H, W = input_shape

        hidden_states = input_tensor
        tt_blocks = list(zip(self.resnets, self.attentions))
        for resnet, attn in tt_blocks:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = ttnn.concat([hidden_states, res_hidden_states], dim=3)
            C = list(hidden_states.shape)[3]

            hidden_states, [C, H, W] = resnet.forward(hidden_states, temb, [B, C, H, W])
            hidden_states = attn.forward(
                hidden_states, [B, C, H, W], encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
            )

        ttnn.DumpDeviceProfiler(self.device)

        if self.upsamplers is not None:
            hidden_states = ttnn.reshape(hidden_states, [B, H, W, C])
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
            hidden_states, [C, H, W] = self.upsamplers.forward(hidden_states)
        return hidden_states, [C, H, W]
