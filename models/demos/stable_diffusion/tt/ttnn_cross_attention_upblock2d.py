# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.stable_diffusion.tt.ttnn_resnetblock2d import ResnetBlock2D
from models.demos.stable_diffusion.tt.ttnn_transformer2d import sd_transformer_2d
from models.demos.stable_diffusion.tt.ttnn_upsample2d import upsample


def up_block_2d(device, parameters, config, input, temb, res_sample, num_layers=3):
    hidden_states = input
    input_tuple = res_sample[-1]
    input_tuple = ttnn.to_memory_config(input_tuple, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    for i in range(num_layers):
        if i == 2:
            hidden_states = ttnn.to_torch(hidden_states)
            hidden_states = ttnn.from_torch(hidden_states, dtype=ttnn.bfloat16, device=device)
        hidden_states = ttnn.to_memory_config(hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        hidden_states = ttnn.concat([hidden_states, input_tuple], dim=1)

        hidden_states = ResnetBlock2D(
            config=config,
            input_tensor=hidden_states,
            temb=temb,
            parameters=parameters.resnets[i],
            device=device,
            eps=1e-5,
            time_embedding_norm="default",
            non_linearity="silu",
        )
    return hidden_states


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
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    res_tensor = ttnn.to_memory_config(
        tt_res_hidden_states_tuple[-1], memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )
    hidden_states = ttnn.concat((input_tensor, res_tensor), dim=1)

    hidden_states = ResnetBlock2D(config, hidden_states, temb, parameters=parameters.resnets[0], device=device)
    hidden_states = sd_transformer_2d(
        hidden_states,
        encoder_hidden_states,
        num_layers=num_layers,
        attention_head_dim=attention_head_dim,
        device=device,
        parameters=parameters.attentions[0],
        config=config,
    )
    ttnn.deallocate(res_tensor)
    res_tensor = ttnn.to_memory_config(
        tt_res_hidden_states_tuple[1], memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )
    hidden_states = ttnn.concat((hidden_states, res_tensor), dim=1)
    hidden_states = ResnetBlock2D(config, hidden_states, temb, parameters=parameters.resnets[1], device=device)
    hidden_states = sd_transformer_2d(
        hidden_states,
        encoder_hidden_states,
        num_layers=num_layers,
        attention_head_dim=attention_head_dim,
        device=device,
        parameters=parameters.attentions[1],
        config=config,
    )
    ttnn.deallocate(res_tensor)
    res_tensor = ttnn.to_memory_config(
        tt_res_hidden_states_tuple[0], memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )

    hidden_states = ttnn.concat((hidden_states, res_tensor), dim=1)

    hidden_states = ResnetBlock2D(config, hidden_states, temb, parameters=parameters.resnets[2], device=device)
    hidden_states = sd_transformer_2d(
        hidden_states,
        encoder_hidden_states,
        num_layers=num_layers,
        attention_head_dim=attention_head_dim,
        device=device,
        parameters=parameters.attentions[2],
        config=config,
    )
    ttnn.deallocate(res_tensor)
    if hidden_states.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)

    hidden_states = upsample(hidden_states, parameters.upsamplers[0], device)

    return hidden_states
