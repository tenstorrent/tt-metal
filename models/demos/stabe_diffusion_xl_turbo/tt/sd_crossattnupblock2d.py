# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.stabe_diffusion_xl_turbo.tt import resnetblock2d, sd_transformer2d, tt_upsample_2d


def sd_crossattnupblock2d(
    device,
    input_tensor,
    tt_res_hidden_states_tuple,
    temb,
    encoder_hidden_states,
    parameters,
    config,
    num_layers,
    attention_head_dim,
):
    hidden_states = ttnn.concat((input_tensor, tt_res_hidden_states_tuple[-1]), dim=1)
    hidden_states = resnetblock2d.ResnetBlock2D(
        config, hidden_states, temb, parameters=parameters.resnets[0], device=device
    )
    hidden_states = sd_transformer2d.sd_transformer_2d(
        hidden_states,
        encoder_hidden_states,
        num_layers=num_layers,
        attention_head_dim=attention_head_dim,
        device=device,
        parameters=parameters.attentions[0],
        config=config,
    )

    # if hidden_states[0].get_layout() != ttnn.ROW_MAJOR_LAYOUT:
    #     hidden_states[0] = ttnn.to_layout(hidden_states[0], ttnn.ROW_MAJOR_LAYOUT)
    # if tt_res_hidden_states_tuple[1].get_layout() != ttnn.ROW_MAJOR_LAYOUT:
    #     tt_res_hidden_states_tuple[1] = ttnn.to_layout((tt_res_hidden_states_tuple[1]), ttnn.ROW_MAJOR_LAYOUT)

    hidden_states = ttnn.concat((hidden_states[0], tt_res_hidden_states_tuple[1]), dim=1)
    hidden_states = resnetblock2d.ResnetBlock2D(
        config, hidden_states, temb, parameters=parameters.resnets[1], device=device
    )
    hidden_states = sd_transformer2d.sd_transformer_2d(
        hidden_states,
        encoder_hidden_states,
        num_layers=num_layers,
        attention_head_dim=attention_head_dim,
        device=device,
        parameters=parameters.attentions[1],
        config=config,
    )

    # if hidden_states.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
    #     hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
    # if tt_res_hidden_states_tuple[0].get_layout() != ttnn.ROW_MAJOR_LAYOUT:
    #     tt_res_hidden_states_tuple[0] = ttnn.to_layout((tt_res_hidden_states_tuple[0]), ttnn.ROW_MAJOR_LAYOUT)

    hidden_states = ttnn.concat((hidden_states[0], tt_res_hidden_states_tuple[0]), dim=1)
    hidden_states = resnetblock2d.ResnetBlock2D(
        config, hidden_states, temb, parameters=parameters.resnets[2], device=device
    )
    hidden_states = sd_transformer2d.sd_transformer_2d(
        hidden_states,
        encoder_hidden_states,
        num_layers=num_layers,
        attention_head_dim=attention_head_dim,
        device=device,
        parameters=parameters.attentions[2],
        config=config,
    )

    hidden_states = hidden_states[0]

    if hidden_states.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)

    hidden_states = tt_upsample_2d.upsample(hidden_states, parameters.upsamplers[0], device)

    return hidden_states
