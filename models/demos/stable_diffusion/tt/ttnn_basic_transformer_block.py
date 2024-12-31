# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.demos.stable_diffusion.tt.ttnn_attention import sd_attention


def sd_geglu(
    hidden_states,
    parameters,
    device=None,
):
    x = ttnn.linear(
        hidden_states,
        parameters.proj.weight,
        bias=parameters.proj.bias,
    )
    x = ttnn.geglu(x)

    return x


def sd_feed_forward(
    hidden_states,
    parameters,
    device,
):
    hidden_states = sd_geglu(hidden_states, parameters.net[0], device)
    hidden_states = ttnn.linear(
        hidden_states,
        parameters.net[2].weight,
        bias=parameters.net[2].bias,
        dtype=ttnn.bfloat16,
    )
    return hidden_states


def sd_basic_transformer_block(
    hidden_states,
    encoder_hidden_states=None,
    timestep=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    class_labels=None,
    config=None,
    num_embeds_ada_norm=False,
    cross_attention_dim: int = None,
    only_cross_attention: bool = False,
    attention_head_dim=None,
    *,
    parameters,
    device,
):
    norm_hidden_states = ttnn.layer_norm(
        hidden_states,
        epsilon=1e-05,
        weight=parameters.norm1.weight,
        bias=parameters.norm1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
    cross_attention_dim = config.cross_attention_dim if cross_attention_dim is None else cross_attention_dim
    attn_output = sd_attention(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if only_cross_attention else None,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        cross_attention_dim=cross_attention_dim,
        heads=attention_head_dim,
        parameters=parameters.attn1,
        device=device,
    )
    hidden_states = ttnn.add(attn_output, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(norm_hidden_states)
    if cross_attention_dim is not None:
        norm_hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=1e-05,
            weight=parameters.norm2.weight,
            bias=parameters.norm2.bias,
            # memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        attn_output = sd_attention(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            cross_attention_dim=cross_attention_dim,
            heads=attention_head_dim,
            parameters=parameters.attn2,
            device=device,
        )
        hidden_states = ttnn.add(attn_output, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_output)
        norm_hidden_states = ttnn.layer_norm(
            hidden_states,
            epsilon=1e-05,
            weight=parameters.norm3.weight,
            bias=parameters.norm3.bias,
            # memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ff_output = sd_feed_forward(hidden_states=norm_hidden_states, parameters=parameters.ff, device=device)
        hidden_states = ttnn.add(ff_output, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG)
        return hidden_states
