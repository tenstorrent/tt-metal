# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
from models.experimental.stable_diffusion_xl_base.tt.tt_transformermodel import TtTransformer2DModel
from models.experimental.stable_diffusion_xl_base.tt.tt_resnetblock2d import TtResnetBlock2D
from models.experimental.stable_diffusion_xl_base.tt.tt_downsample2d import TtDownsample2D


class TtCrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        query_dim,
        num_attn_heads,
        out_dim,
        has_downsample=False,
        transformer_weights_dtype=ttnn.bfloat16,
        conv_weights_dtype=ttnn.bfloat16,
    ):
        super().__init__()

        num_layers = 2
        self.attentions = []
        self.resnets = []

        for i in range(num_layers):
            self.attentions.append(
                TtTransformer2DModel(
                    device,
                    state_dict,
                    f"{module_path}.attentions.{i}",
                    query_dim,
                    num_attn_heads,
                    out_dim,
                    weights_dtype=transformer_weights_dtype,
                )
            )

        for i in range(num_layers):
            self.resnets.append(
                TtResnetBlock2D(
                    device, state_dict, f"{module_path}.resnets.{i}", i == 0, conv_weights_dtype=conv_weights_dtype
                )
            )

        self.downsamplers = (
            TtDownsample2D(
                device,
                state_dict,
                f"{module_path}.downsamplers.0",
                (2, 2),
                (1, 1),
                (1, 1),
                1,
                conv_weights_dtype=conv_weights_dtype,
            )
            if has_downsample
            else None
        )

    def forward(self, input_tensor, input_shape, temb=None, encoder_hidden_states=None, attention_mask=None):
        B, C, H, W = input_shape
        output_states = ()

        hidden_states = input_tensor
        tt_blocks = list(zip(self.resnets, self.attentions))
        for resnet, attn in tt_blocks:
            hidden_states, [C, H, W] = resnet.forward(hidden_states, temb, [B, C, H, W])
            hidden_states = attn.forward(hidden_states, [B, C, H, W], encoder_hidden_states=encoder_hidden_states)
            residual = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
            output_states = output_states + (residual,)

        if self.downsamplers is not None:
            hidden_states, [C, H, W] = self.downsamplers.forward(hidden_states, [B, C, H, W])
            residual = ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG)
            output_states = output_states + (residual,)
        return hidden_states, [C, H, W], output_states
