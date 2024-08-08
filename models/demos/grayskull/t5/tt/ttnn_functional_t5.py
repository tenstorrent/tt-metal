# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import functools
import math

import torch

import ttnn

from models.experimental.functional_common.attention_mask_functions import (
    get_extended_attention_mask,
    invert_attention_mask,
)


def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on

    Args:
        relative_position: an int32 Tensor
        bidirectional: a boolean - whether the attention is bidirectional
        num_buckets: an integer
        max_distance: an integer

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
    """
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        relative_position = torch.abs(relative_position)
    else:
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
        torch.log(relative_position.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


def compute_bias(config, query_length, key_length, *, is_decoder, parameters):
    """Compute binned relative position bias"""
    context_position = torch.arange(query_length, dtype=torch.long)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
    relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_bucket = _relative_position_bucket(
        relative_position,  # shape (query_length, key_length)
        bidirectional=(not is_decoder),
        num_buckets=config.relative_attention_num_buckets,
        max_distance=config.relative_attention_max_distance,
    )
    values = torch.nn.functional.embedding(
        relative_position_bucket, parameters.relative_attention_bias.weight
    )  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    return values


def t5_layer_norm(config, hidden_states, *, weight):
    # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
    # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
    # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
    # half-precision inputs is done in fp32

    squared_hidden_states = ttnn.pow(hidden_states, 2)
    averaged_squared_hidden_states = ttnn.mean(
        squared_hidden_states,
        dim=-1,
    )

    variance = averaged_squared_hidden_states + config.layer_norm_epsilon
    std = ttnn.rsqrt(variance)

    hidden_states = hidden_states * std
    hidden_states = hidden_states * weight

    return hidden_states


def get_activation_function(dense_act_fn):
    if dense_act_fn == "relu":
        return ttnn.relu
    elif dense_act_fn == "gelu_new":
        return ttnn.gelu
    else:
        raise RuntimeError(f"Unsupported activation function: {dense_act_fn}")


def t5_dense_act_dense(config, hidden_states, parameters):
    activation_function = get_activation_function(config.dense_act_fn)

    hidden_states = hidden_states @ parameters.wi.weight
    hidden_states = activation_function(hidden_states)
    hidden_states = hidden_states @ parameters.wo.weight
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
    position_bias=None,
    *,
    is_decoder,
    parameters,
):
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length, _ = hidden_states.shape

    real_seq_length = seq_length
    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states, head_size, is_key=False):
        """projection"""
        states = ttnn.to_layout(states, layout=ttnn.ROW_MAJOR_LAYOUT)
        states = ttnn.get_fallback_function(ttnn.reshape)(states, (batch_size, seq_length, config.num_heads, head_size))
        if is_key:
            states = ttnn.permute(states, (0, 2, 3, 1))
        else:
            states = ttnn.permute(states, (0, 2, 1, 3))
        states = ttnn.to_layout(states, ttnn.TILE_LAYOUT)
        return states

    def unshape(states, hidden_size):
        """reshape"""
        states = ttnn.permute(states, (0, 2, 1, 3))
        states = ttnn.to_layout(states, ttnn.ROW_MAJOR_LAYOUT)
        states = ttnn.get_fallback_function(ttnn.reshape)(states, (batch_size, seq_length, hidden_size))
        states = ttnn.to_layout(states, ttnn.TILE_LAYOUT)
        return states

    def project(hidden_states, weight, is_key=False):
        hidden_size = weight.shape[-1]
        head_size = hidden_size // config.num_heads
        """projects hidden states correctly to key/query states"""
        # self-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(hidden_states @ weight, head_size, is_key=is_key)
        return hidden_states

    # get query states
    hidden_size = parameters.q.weight.shape[-1]
    query_states = project(hidden_states, parameters.q.weight)  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states if key_value_states is None else key_value_states,
        parameters.k.weight,
        is_key=True,
    )
    value_states = project(
        hidden_states if key_value_states is None else key_value_states,
        parameters.v.weight,
    )

    # compute scores
    scores = ttnn.matmul(query_states, key_states)

    if position_bias is None:
        if "relative_attention_bias" in parameters:
            position_bias = compute_bias(
                config, real_seq_length, key_length, is_decoder=is_decoder, parameters=parameters
            )
        else:
            position_bias = torch.zeros((1, config.num_heads, real_seq_length, key_length), dtype=torch.float32)

        position_bias = ttnn.from_torch(
            position_bias, dtype=ttnn.bfloat16, device=scores.device(), layout=ttnn.TILE_LAYOUT
        )

        if mask is not None:
            position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

    scores += position_bias

    attn_weights = ttnn.softmax(scores, dim=-1)

    # Mask heads if we want to
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(ttnn.matmul(attn_weights, value_states), hidden_size)  # (batch_size, seq_length, dim)
    attn_output = attn_output @ parameters.o.weight

    return attn_output, position_bias


def t5_layer_self_attention(
    config,
    hidden_states,
    attention_mask=None,
    position_bias=None,
    *,
    is_decoder,
    parameters,
):
    normed_hidden_states = t5_layer_norm(config, hidden_states, weight=parameters.layer_norm.weight)
    attention_output, position_bias = t5_attention(
        config,
        normed_hidden_states,
        mask=attention_mask,
        position_bias=position_bias,
        is_decoder=is_decoder,
        parameters=parameters.SelfAttention,
    )
    hidden_states = hidden_states + attention_output
    return hidden_states, position_bias


def t5_layer_cross_attention(
    config, hidden_states, key_value_states, attention_mask=None, position_bias=None, *, is_decoder, parameters
):
    normed_hidden_states = t5_layer_norm(config, hidden_states, weight=parameters.layer_norm.weight)
    attention_output, position_bias = t5_attention(
        config,
        normed_hidden_states,
        key_value_states,
        mask=attention_mask,
        position_bias=position_bias,
        is_decoder=is_decoder,
        parameters=parameters.EncDecAttention,
    )
    layer_output = hidden_states + attention_output
    return layer_output, position_bias


def t5_block(
    config,
    hidden_states,
    attention_mask=None,
    position_bias=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    encoder_decoder_position_bias=None,
    *,
    is_decoder,
    parameters,
):
    hidden_states, position_bias = t5_layer_self_attention(
        config,
        hidden_states,
        attention_mask=attention_mask,
        position_bias=position_bias,
        is_decoder=is_decoder,
        parameters=parameters.layer[0],
    )

    do_cross_attention = encoder_hidden_states is not None
    if do_cross_attention:
        hidden_states, encoder_decoder_position_bias = t5_layer_cross_attention(
            config,
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            position_bias=encoder_decoder_position_bias,
            is_decoder=is_decoder,
            parameters=parameters.layer[1],
        )

    # Apply Feed Forward layer
    hidden_states = t5_layer_ff(config, hidden_states, parameters.layer[-1])

    return hidden_states, position_bias, encoder_decoder_position_bias


def t5_stack(
    config,
    input_ids,
    shared_embedding_weight,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    *,
    parameters,
):
    input_shape = tuple(input_ids.shape)

    hidden_states = ttnn.embedding(input_ids, shared_embedding_weight, layout=ttnn.TILE_LAYOUT)

    is_decoder = encoder_hidden_states is not None
    if attention_mask is None:
        attention_mask = create_attention_mask(input_shape, config.num_heads, input_ids.device(), is_decoder=is_decoder)
    if encoder_hidden_states is not None:
        encoder_attention_mask = create_encoder_attention_mask(input_shape, config.num_heads, input_ids.device())
    else:
        encoder_attention_mask = None

    position_bias = None
    encoder_decoder_position_bias = None

    for block_parameters in parameters.block:
        hidden_states, position_bias, encoder_decoder_position_bias = t5_block(
            config,
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_decoder_position_bias=encoder_decoder_position_bias,
            is_decoder=is_decoder,
            parameters=block_parameters,
        )

    hidden_states = t5_layer_norm(config, hidden_states, weight=parameters.final_layer_norm.weight)

    return hidden_states


def t5_for_conditional_generation(
    config,
    input_ids: ttnn.Tensor,
    decoder_input_ids: ttnn.Tensor,
    parameters,
    *,
    encoder_last_hidden_state=None,
) -> ttnn.Tensor:
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

    if config.tie_word_embeddings:
        sequence_output *= config.d_model**-0.5

    lm_logits = sequence_output @ parameters.lm_head.weight

    return lm_logits, encoder_last_hidden_state


@functools.lru_cache
def create_attention_mask(input_shape, num_heads, device, is_decoder):
    batch_size, seq_length = input_shape

    attention_mask = torch.ones(batch_size, seq_length)

    extended_attention_mask = get_extended_attention_mask(
        attention_mask, input_shape, is_decoder=is_decoder, dtype=torch.bfloat16
    )

    extended_attention_mask = extended_attention_mask.expand((-1, num_heads, seq_length, -1))
    extended_attention_mask = ttnn.from_torch(extended_attention_mask)
    extended_attention_mask = ttnn.to_layout(extended_attention_mask, ttnn.TILE_LAYOUT)
    extended_attention_mask = ttnn.to_device(extended_attention_mask, device)
    return extended_attention_mask


@functools.lru_cache
def create_encoder_attention_mask(input_shape, num_heads, device):
    batch_size, seq_length = input_shape

    encoder_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)

    encoder_extended_attention_mask = encoder_extended_attention_mask.expand((-1, num_heads, seq_length, -1))
    encoder_extended_attention_mask = ttnn.from_torch(encoder_extended_attention_mask)
    encoder_extended_attention_mask = ttnn.to_layout(encoder_extended_attention_mask, ttnn.TILE_LAYOUT)
    encoder_extended_attention_mask = ttnn.to_device(encoder_extended_attention_mask, device)
    return encoder_extended_attention_mask


def convert_to_ttnn(model, name):
    return "relative_attention_bias" not in name


def custom_preprocessor(model, name):
    import transformers
    from ttnn.model_preprocessing import preprocess_layernorm_parameter

    parameters = {}
    if isinstance(model, transformers.models.t5.modeling_t5.T5LayerNorm):
        parameters["weight"] = preprocess_layernorm_parameter(model.weight, dtype=ttnn.bfloat16)

    return parameters
