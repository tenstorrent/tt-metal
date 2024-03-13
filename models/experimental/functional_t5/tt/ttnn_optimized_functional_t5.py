# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Optional

import torch

import ttnn

from models.experimental.functional_common.attention_mask_functions import (
    get_extended_attention_mask,
    invert_attention_mask,
)


def t5_layer_norm(config, hidden_states, *, weight):
    return ttnn.rms_norm(hidden_states, weight, epsilon=config.layer_norm_epsilon)


def get_activation_function(dense_act_fn):
    if dense_act_fn == "relu":
        return ttnn.relu
    elif dense_act_fn == "gelu_new":
        return ttnn.gelu
    else:
        raise RuntimeError(f"Unsupported activation function: {dense_act_fn}")


def t5_dense_act_dense(config, hidden_states, parameters):
    if config.dense_act_fn == "relu":
        ff1_activation = "relu"
    elif config.dense_act_fn == "gelu_new":
        ff1_activation = "gelu"
    else:
        raise RuntimeError(f"Unsupported activation function: {config.dense_act_fn}")

    _, height, _ = hidden_states.shape
    hidden_states = ttnn.linear(
        hidden_states,
        parameters.wi.weight,
        dtype=ttnn.bfloat8_b,
        activation=ff1_activation,
        core_grid=ttnn.CoreGrid(y=height // 32, x=12),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    hidden_states = ttnn.linear(
        hidden_states,
        parameters.wo.weight,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=9, x=12),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return hidden_states


def t5_dense_gated_act_dense(config, hidden_states, parameters):
    activation_function = get_activation_function(config.dense_act_fn)

    hidden_gelu = hidden_states @ parameters.wi_0.weight
    hidden_gelu = activation_function(hidden_gelu)
    hidden_linear = hidden_states @ parameters.wi_1.weight
    hidden_states = hidden_gelu * hidden_linear

    hidden_states = hidden_states @ parameters.wo.weight
    return hidden_states


def t5_layer_ff(config, hidden_states, parameters):
    forwarded_states = t5_layer_norm(config, hidden_states, weight=parameters.layer_norm.weight)
    if config.is_gated_act:
        forwarded_states = t5_dense_gated_act_dense(config, forwarded_states, parameters.DenseReluDense)
    else:
        forwarded_states = t5_dense_act_dense(config, forwarded_states, parameters.DenseReluDense)
    hidden_states = ttnn.add(hidden_states, forwarded_states, memory_config=ttnn.L1_MEMORY_CONFIG)
    return hidden_states


def t5_attention(
    config,
    hidden_states,
    key_value_states=None,
    mask=None,
    layer_head_mask=None,
    *,
    parameters,
    num_cores_x=12,
):
    batch_size, *_ = hidden_states.shape

    if key_value_states is None:
        query_key_value_output = ttnn.linear(
            hidden_states,
            parameters.query_key_value.weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
        )

        (
            query,
            key,
            value,
        ) = ttnn.transformer.split_query_key_value_and_split_heads(
            query_key_value_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            num_heads=config.num_heads,
        )
        ttnn.deallocate(query_key_value_output)

    else:
        query_proj = ttnn.linear(
            hidden_states,
            parameters.q.weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
        )

        key_value_proj = ttnn.linear(
            key_value_states,
            parameters.key_value.weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            # dtype=ttnn.bfloat8_b,
            core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
        )
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            query_proj, key_value_proj, num_heads=config.num_heads
        )
        ttnn.deallocate(query_proj)
        ttnn.deallocate(key_value_proj)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        # core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    if mask is None:
        attention_probs = ttnn.softmax(attention_scores, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        attention_probs = ttnn.transformer.attention_softmax_(attention_scores, attention_mask=mask, head_size=None)

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        # dtype=ttnn.bfloat8_b,
        # core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    self_output = ttnn.linear(
        context_layer,
        parameters.o.weight,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        # core_grid=ttnn.CoreGrid(y=batch_size, x=num_cores_x),
    )
    ttnn.deallocate(context_layer)

    return self_output


def t5_layer_self_attention(
    config,
    hidden_states,
    attention_mask=None,
    *,
    parameters,
):
    normed_hidden_states = t5_layer_norm(config, hidden_states, weight=parameters.layer_norm.weight)
    attention_output = t5_attention(
        config,
        normed_hidden_states,
        mask=attention_mask,
        parameters=parameters.SelfAttention,
    )
    hidden_states = ttnn.add(hidden_states, attention_output, memory_config=ttnn.L1_MEMORY_CONFIG)
    return hidden_states


def t5_layer_cross_attention(config, hidden_states, key_value_states, attention_mask=None, *, parameters):
    normed_hidden_states = t5_layer_norm(config, hidden_states, weight=parameters.layer_norm.weight)
    attention_output = t5_attention(
        config,
        normed_hidden_states,
        key_value_states=key_value_states,
        mask=attention_mask,
        parameters=parameters.EncDecAttention,
    )
    layer_output = ttnn.add(hidden_states, attention_output, memory_config=ttnn.L1_MEMORY_CONFIG)
    return layer_output


def t5_block(
    config,
    hidden_states,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    *,
    parameters,
):
    hidden_states = t5_layer_self_attention(
        config,
        hidden_states,
        attention_mask=attention_mask,
        parameters=parameters.layer[0],
    )

    do_cross_attention = encoder_hidden_states is not None
    if do_cross_attention:
        hidden_states = t5_layer_cross_attention(
            config,
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            parameters=parameters.layer[1],
        )

    # Apply Feed Forward layer
    hidden_states = t5_layer_ff(config, hidden_states, parameters.layer[-1])

    return hidden_states  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


def t5_stack(
    config,
    input_ids,
    shared_embedding_weight,
    encoder_hidden_states=None,
    *,
    parameters,
):
    input_shape = tuple(input_ids.shape)

    hidden_states = ttnn.embedding(
        input_ids, shared_embedding_weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    attention_mask = create_attention_mask(
        input_shape, input_ids.device(), is_decoder=encoder_hidden_states is not None
    )
    if encoder_hidden_states is not None:
        encoder_attention_mask = create_encoder_attention_mask(input_shape, input_ids.device())
    else:
        encoder_attention_mask = None

    for block_parameters in parameters.block:
        hidden_states = t5_block(
            config,
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            parameters=block_parameters,
        )

    hidden_states = t5_layer_norm(config, hidden_states, weight=parameters.final_layer_norm.weight)

    return hidden_states


def t5_for_conditional_generation(
    config,
    input_ids: Optional[torch.LongTensor],
    decoder_input_ids: Optional[torch.LongTensor],
    parameters,
    *,
    encoder_last_hidden_state=None,
) -> torch.FloatTensor:
    # Encode
    if encoder_last_hidden_state is None:
        encoder_last_hidden_state = t5_stack(
            config,
            input_ids=input_ids,
            shared_embedding_weight=parameters.shared.weight,
            parameters=parameters.encoder,
        )

    # Decode
    sequence_output = t5_stack(
        config,
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_last_hidden_state,
        shared_embedding_weight=parameters.shared.weight,
        parameters=parameters.decoder,
    )

    lm_logits = ttnn.linear(sequence_output, parameters.lm_head.weight, memory_config=ttnn.L1_MEMORY_CONFIG)

    return lm_logits, encoder_last_hidden_state


@functools.lru_cache
def create_attention_mask(input_shape, device, is_decoder):
    batch_size, seq_length = input_shape

    attention_mask = torch.ones(batch_size, seq_length)

    extended_attention_mask = get_extended_attention_mask(
        attention_mask, input_shape, is_decoder=is_decoder, dtype=torch.bfloat16
    )

    extended_attention_mask = extended_attention_mask.expand((-1, -1, seq_length, -1))
    extended_attention_mask = ttnn.from_torch(extended_attention_mask)
    extended_attention_mask = ttnn.to_layout(extended_attention_mask, ttnn.TILE_LAYOUT)
    extended_attention_mask = ttnn.to_device(extended_attention_mask, device)
    return extended_attention_mask


@functools.lru_cache
def create_encoder_attention_mask(input_shape, device):
    batch_size, seq_length = input_shape

    encoder_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)

    encoder_extended_attention_mask = encoder_extended_attention_mask.expand((-1, -1, seq_length, -1))
    encoder_extended_attention_mask = ttnn.from_torch(encoder_extended_attention_mask)
    encoder_extended_attention_mask = ttnn.to_layout(encoder_extended_attention_mask, ttnn.TILE_LAYOUT)
    encoder_extended_attention_mask = ttnn.to_device(encoder_extended_attention_mask, device)
    return encoder_extended_attention_mask


def custom_preprocessor(model, name):
    import transformers
    from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_layernorm_parameter

    parameters = {}
    if isinstance(model, transformers.models.t5.modeling_t5.T5LayerNorm):
        parameters["weight"] = preprocess_layernorm_parameter(model.weight, dtype=ttnn.bfloat16)

    elif isinstance(model, transformers.models.t5.modeling_t5.T5Attention):
        if "EncDecAttention" in name:
            # Cross Attention
            preprocessed_kv_weight = torch.cat([model.k.weight, model.v.weight], dim=0)
            parameters = {
                "q": {"weight": preprocess_linear_weight(model.q.weight, dtype=ttnn.bfloat16)},
                "key_value": {"weight": preprocess_linear_weight(preprocessed_kv_weight, dtype=ttnn.bfloat16)},
                "o": {"weight": preprocess_linear_weight(model.o.weight, dtype=ttnn.bfloat16)},
            }
        else:
            # Self Attention
            preprocessed_qkv_weight = torch.cat([model.q.weight, model.k.weight, model.v.weight], dim=0)
            parameters = {
                "query_key_value": {"weight": preprocess_linear_weight(preprocessed_qkv_weight, dtype=ttnn.bfloat16)},
                "o": {"weight": preprocess_linear_weight(model.o.weight, dtype=ttnn.bfloat16)},
            }

    return parameters
