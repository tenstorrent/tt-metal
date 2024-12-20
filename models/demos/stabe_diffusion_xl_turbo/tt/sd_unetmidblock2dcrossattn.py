# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.stabe_diffusion_xl_turbo.tt import resnetblock2d, sd_transformer2d


def sd_unetmidblock2dcrossattn(device, input_tensor, temb, encoder_hidden_states, parameters, config):
    hidden_states = resnetblock2d.ResnetBlock2D(
        config, input_tensor, temb, parameters=parameters.resnets[0], device=device
    )
    hidden_states = sd_transformer2d.sd_transformer_2d(
        hidden_states,
        encoder_hidden_states,
        num_layers=10,
        attention_head_dim=20,
        device=device,
        parameters=parameters.attentions[0],
        config=config,
    )
    hidden_states = resnetblock2d.ResnetBlock2D(
        config, hidden_states[0], temb, device=device, parameters=parameters.resnets[1]
    )

    return hidden_states
