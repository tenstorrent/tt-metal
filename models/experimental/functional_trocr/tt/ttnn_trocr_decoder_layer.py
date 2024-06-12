# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.functional_trocr.tt.ttnn_trocr_attention import trocr_attention
import tt_lib as ttl


def trocr_decoder_layer(
    hidden_states,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    layer_head_mask=None,
    cross_attn_layer_head_mask=None,
    past_key_value=None,
    output_attentions=None,
    use_cache=None,
    config=None,
    parameters=None,
    device=None,
):
    embed_dim = config.hidden_size

    residual = hidden_states

    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    ttl.device.DumpDeviceProfiler(device)
    hidden_states, self_attn_weights, present_key_value = trocr_attention(
        config=config,
        embed_dim=embed_dim,
        num_heads=config.decoder_attention_heads,
        is_decoder=True,
        hidden_states=hidden_states,
        past_key_value=self_attn_past_key_value,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
        parameters=parameters.self_attn,
        device=device,
    )

    hidden_states = ttnn.add(residual, hidden_states)

    hidden_states = ttnn.layer_norm(
        hidden_states, weight=parameters.self_attn_layer_norm.weight, bias=parameters.self_attn_layer_norm.bias
    )

    # Cross-Attention Block
    cross_attn_present_key_value = None
    cross_attn_weights = None

    if encoder_hidden_states is not None:
        residual = hidden_states

        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        hidden_states, cross_attn_weights, cross_attn_present_key_value = trocr_attention(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,
            output_attentions=output_attentions,
            config=config,
            embed_dim=embed_dim,
            num_heads=config.decoder_attention_heads,
            kdim=config.cross_attention_hidden_size,
            vdim=config.cross_attention_hidden_size,
            is_decoder=True,
            is_cross_attention=True,
            parameters=parameters.encoder_attn,
            device=device,
        )

        hidden_states = ttnn.add(residual, hidden_states)
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=parameters.encoder_attn_layer_norm.weight,
            bias=parameters.encoder_attn_layer_norm.bias,
        )

        present_key_value = present_key_value + cross_attn_present_key_value

    residual = hidden_states

    hidden_states = ttnn.linear(hidden_states, parameters.fc1.weight, bias=parameters.fc1.bias)

    hidden_states = ttnn.gelu(hidden_states)

    hidden_states = ttnn.linear(hidden_states, parameters.fc2.weight, bias=parameters.fc2.bias)

    hidden_states = ttnn.add(residual, hidden_states)

    hidden_states = ttnn.layer_norm(
        hidden_states, weight=parameters.final_layer_norm.weight, bias=parameters.final_layer_norm.bias
    )

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)

    if use_cache:
        outputs += (present_key_value,)

    return outputs
