# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math

from models.experimental.functional_trocr.tt.ttnn_trocr_decoder_layer import trocr_decoder_layer
from models.experimental.functional_trocr.ttnn_trocr_utils import _prepare_4d_causal_attention_mask, _expand_mask
from models.experimental.functional_trocr.tt.ttnn_trocr_learned_positional_embedding import TtTrOCRPositionalEmbedding
import tt_lib as ttl


def trocr_decoder(
    input_ids=None,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    head_mask=None,
    cross_attn_head_mask=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    config=None,
    parameters=None,
    device=None,
):
    embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

    output_attentions = output_attentions if output_attentions is not None else config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else config.output_hidden_states
    use_cache = use_cache if use_cache is not None else config.use_cache
    return_dict = return_dict if return_dict is not None else config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None:
        input = input_ids
        input_ids = ttnn.from_device(input_ids)
        input_ids = ttnn.to_layout(input_ids, ttnn.ROW_MAJOR_LAYOUT)
        input_ids = ttnn.reshape(input_ids, (-1, input.shape[-1]))
        input_ids = ttnn.to_torch(input_ids)
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.shape[:-1]
        input = inputs_embeds[:, :, -1]

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    if inputs_embeds is None:
        input_ids = ttnn.from_torch(input_ids, device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        inputs_embeds = ttnn.embedding(input_ids, weight=parameters.embed_tokens.weight)
        inputs_embeds = ttnn.to_layout(inputs_embeds, ttnn.TILE_LAYOUT)
        inputs_embeds = inputs_embeds * embed_scale

    if config.use_learned_position_embeddings:
        input = ttnn.to_torch(input)
        embed_pos = TtTrOCRPositionalEmbedding(
            input_ids=input,
            parameters=parameters.embed_positions,
            device=device,
        )

    hidden_states = ttnn.add(inputs_embeds, embed_pos)

    hidden_states = ttnn.layer_norm(hidden_states, weight=parameters.layernorm_embedding.weight)

    input_shape = input_ids.shape

    inputs_embeds = ttnn.to_torch(inputs_embeds)
    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )

    # expand encoder attention mask
    if encoder_hidden_states is not None and encoder_attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
    next_decoder_cache = () if use_cache else None

    for idx in range(config.decoder_layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None
        ttl.device.DumpDeviceProfiler(device)
        layer_outputs = trocr_decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            config=config,
            parameters=parameters.layers[idx],
            device=device,
        )
        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[2],)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
            if v is not None
        )
    return (
        hidden_states,
        next_cache,
        all_hidden_states,
        all_self_attns,
        all_cross_attentions,
    )
