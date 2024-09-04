# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Tuple

import transformers
import torch
from torch.nn import functional as F
from transformers.models.bloom.configuration_bloom import BloomConfig

import ttnn
from ttnn.model_preprocessing import (
    ParameterDict,
    preprocess_linear_weight,
    preprocess_linear_bias,
)


# From transformers/models/bloom/modeling_bloom.py
def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size, num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        device=attention_mask.device,
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            device=attention_mask.device,
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1,
            1 + 2 * num_remaining_heads,
            2,
            device=attention_mask.device,
            dtype=torch.int32,
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size, num_heads, 1, seq_length).to(dtype)


# Based on transformers/models/bloom/modeling_bloom.py
def split_query_key_value_and_split_heads(
    query_key_value: ttnn.Tensor, num_heads: int
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    device = query_key_value.device()

    batch_size, seq_length, three_times_hidden_size = query_key_value.shape
    hidden_size = three_times_hidden_size // 3
    head_size = hidden_size // num_heads

    query_key_value = ttnn.to_layout(query_key_value, ttnn.ROW_MAJOR_LAYOUT)
    query_key_value = ttnn.from_device(query_key_value)
    query_key_value = ttnn.to_torch(query_key_value)

    query_key_value = query_key_value.view(batch_size, seq_length, 3, num_heads, head_size)
    query_layer, key_layer, value_layer = (
        query_key_value[..., 0, :, :],
        query_key_value[..., 1, :, :],
        query_key_value[..., 2, :, :],
    )

    query_layer = torch.reshape(query_layer, (batch_size, seq_length, num_heads, head_size))
    query_layer = torch.permute(query_layer, (0, 2, 1, 3)).contiguous()

    key_layer = torch.reshape(key_layer, (batch_size, seq_length, num_heads, head_size))
    key_layer = torch.permute(key_layer, (0, 2, 3, 1)).contiguous()

    value_layer = torch.reshape(value_layer, (batch_size, seq_length, num_heads, head_size))
    value_layer = torch.permute(value_layer, (0, 2, 1, 3)).contiguous()

    query_layer = ttnn.from_torch(query_layer)
    query_layer = ttnn.to_device(query_layer, device)
    query_layer = ttnn.to_layout(query_layer, ttnn.TILE_LAYOUT)

    key_layer = ttnn.from_torch(key_layer)
    key_layer = ttnn.to_device(key_layer, device)
    key_layer = ttnn.to_layout(key_layer, ttnn.TILE_LAYOUT)

    value_layer = ttnn.from_torch(value_layer)
    value_layer = ttnn.to_device(value_layer, device)
    value_layer = ttnn.to_layout(value_layer, ttnn.TILE_LAYOUT)

    return query_layer, key_layer, value_layer


# Based on transformers/models/bloom/modeling_bloom.py
def merge_heads(x: ttnn.Tensor) -> ttnn.Tensor:
    # What we want to achieve is:
    # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
    batch_size, num_heads, seq_length, head_size = x.shape

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    x = ttnn.permute(x, (0, 2, 1, 3))

    # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.get_fallback_function(ttnn.reshape)(x, shape=(batch_size, seq_length, num_heads * head_size))
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    return x


# TODO: issue 4172
def create_query_key_value(config: BloomConfig, hidden_states, *, use_core_grid=True, parameters: ParameterDict):
    if use_core_grid:
        query_key_value = ttnn.matmul(
            hidden_states, parameters.query_key_value.weight, core_grid=ttnn.CoreGrid(y=9, x=12)
        )
    else:
        query_key_value = hidden_states @ parameters.query_key_value.weight
    query_key_value = query_key_value + parameters.query_key_value.bias
    return split_query_key_value_and_split_heads(query_key_value, config.n_head)


def bloom_attention(
    config: BloomConfig,
    hidden_states: ttnn.Tensor,
    residual: ttnn.Tensor,
    alibi: ttnn.Tensor,
    attention_mask: ttnn.Tensor,
    *,
    use_core_grid=True,  # TODO: issue 4172
    parameters: ParameterDict,
):
    query_layer, key_layer, value_layer = create_query_key_value(
        config,
        hidden_states,
        parameters=parameters,
        use_core_grid=use_core_grid,
    )

    *_, head_size = query_layer.shape
    beta = 1.0
    inv_norm_factor = 1.0 / math.sqrt(head_size)
    attention_scores = beta * alibi + inv_norm_factor * (query_layer @ key_layer)

    fill_value = -1024
    attention_weights = attention_scores * (1 + (attention_mask * -1)) + attention_mask * fill_value
    attention_probs = ttnn.softmax(attention_weights, dim=-1)

    context_layer = merge_heads(attention_probs @ value_layer)
    output_tensor = context_layer @ parameters.dense.weight + parameters.dense.bias
    output_tensor += residual

    return output_tensor


def bloom_mlp(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    *,
    parameters: ParameterDict,
):
    output_tensor = hidden_states @ parameters.dense_h_to_4h.weight
    output_tensor += parameters.dense_h_to_4h.bias
    output_tensor = ttnn.gelu(output_tensor)
    output_tensor = output_tensor @ parameters.dense_4h_to_h.weight
    output_tensor += parameters.dense_4h_to_h.bias
    output_tensor += residual

    return output_tensor


def bloom_block(
    config: BloomConfig,
    hidden_states: ttnn.Tensor,
    alibi: ttnn.Tensor,
    attention_mask: ttnn.Tensor,
    *,
    parameters: ParameterDict,
) -> ttnn.Tensor:
    layernorm_output = ttnn.layer_norm(
        input_tensor=hidden_states,
        weight=parameters.input_layernorm.weight,
        bias=parameters.input_layernorm.bias,
    )

    if config.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = hidden_states

    attention_output = bloom_attention(
        config,
        layernorm_output,
        residual,
        alibi,
        attention_mask,
        parameters=parameters.self_attention,
    )

    layernorm_output = ttnn.layer_norm(
        input_tensor=attention_output,
        weight=parameters.post_attention_layernorm.weight,
        bias=parameters.post_attention_layernorm.bias,
    )

    if config.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = attention_output

    mlp_output = bloom_mlp(
        layernorm_output,
        residual,
        parameters=parameters.mlp,
    )

    return mlp_output


def bloom(
    config: BloomConfig,
    input_ids,
    alibi,
    causal_mask,
    *,
    parameters: ParameterDict,
):
    inputs_embeds = ttnn.embedding(
        input_ids,
        parameters.word_embeddings.weight,
        layout=ttnn.TILE_LAYOUT,
    )

    hidden_states = ttnn.layer_norm(
        inputs_embeds,
        weight=parameters.word_embeddings_layernorm.weight,
        bias=parameters.word_embeddings_layernorm.bias,
    )

    for layer_parameters in parameters.h:
        hidden_states = bloom_block(
            config,
            hidden_states,
            alibi,
            causal_mask,
            parameters=layer_parameters,
        )

    # Add last hidden state
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.ln_f.weight,
        bias=parameters.ln_f.bias,
    )
    return hidden_states


def bloom_for_causal_lm(config, input_ids, alibi, causal_mask, *, parameters):
    hidden_states = bloom(
        config,
        input_ids,
        alibi,
        causal_mask,
        parameters=parameters.transformer,
    )

    # Unfortunately we do not have the ability to handle large tensors yet. So running final matmul using torch as a workaround.
    hidden_states = ttnn.from_device(hidden_states)
    hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
    hidden_states = ttnn.to_torch(hidden_states).to(torch.float32)
    output = hidden_states @ parameters.lm_head.weight

    return output


def bloom_for_question_answering(
    config,
    input_ids,
    alibi,
    causal_mask,
    *,
    parameters,
):
    hidden_states = bloom(
        config,
        input_ids,
        alibi,
        causal_mask,
        parameters=parameters.transformer,
    )

    hidden_states = ttnn.linear(hidden_states, parameters.qa_outputs.weight, bias=parameters.qa_outputs.bias)
    return hidden_states


def preprocess_inputs(
    *,
    input_ids,
    device,
    num_heads,
    max_length=384,
    attention_mask=None,
):
    num_input_tokens = input_ids.shape[-1]
    padding_needed = (max_length - (num_input_tokens % max_length)) % max_length
    padded_input_ids = F.pad(input_ids, (0, padding_needed, 0, 0))
    padded_input_ids = ttnn.from_torch(padded_input_ids, dtype=ttnn.uint32)
    padded_input_ids = ttnn.to_device(padded_input_ids, device)

    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    attention_mask = F.pad(attention_mask, (0, padding_needed, 0, 0))

    alibi = build_alibi_tensor(attention_mask, num_heads, dtype=torch.float32)
    alibi = ttnn.from_torch(alibi, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    batch_size, padded_seq_length = attention_mask.shape
    mask = torch.empty((padded_seq_length, padded_seq_length), dtype=torch.bool)
    seq_ids = torch.arange(padded_seq_length)
    mask[:, :] = seq_ids[:, None] < seq_ids[None, :]
    causal_mask = mask[None, None, :, :].expand(batch_size, num_heads, padded_seq_length, padded_seq_length)

    expanded_mask = ~(attention_mask[:, None, None, :].to(torch.bool))
    expanded_mask = expanded_mask.expand(batch_size, num_heads, padded_seq_length, padded_seq_length)

    causal_mask = expanded_mask | causal_mask
    causal_mask = causal_mask.float()
    causal_mask = ttnn.from_torch(causal_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return padded_input_ids, alibi, causal_mask


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, transformers.models.bloom.modeling_bloom.BloomAttention):
        weight = torch_model.query_key_value.weight
        bias = torch_model.query_key_value.bias

        assert weight.shape[-1] == 1024
        num_heads = 16

        three_times_hidden_size, _ = weight.shape
        hidden_size = three_times_hidden_size // 3
        head_size = hidden_size // num_heads

        # Store QKV one after another instead of interleaving heads
        weight = weight.view(num_heads, 3, head_size, hidden_size)
        query, key, value = weight[:, 0], weight[:, 1], weight[:, 2]
        query = torch.reshape(query, (hidden_size, hidden_size))
        key = torch.reshape(key, (hidden_size, hidden_size))
        value = torch.reshape(value, (hidden_size, hidden_size))
        preprocessed_weight = torch.cat([query, key, value], dim=0)

        # Store QKV one after another instead of interleaving heads
        bias = bias.view(num_heads, 3, head_size)
        query, key, value = bias[:, 0], bias[:, 1], bias[:, 2]
        query = torch.reshape(query, (hidden_size,))
        key = torch.reshape(key, (hidden_size,))
        value = torch.reshape(value, (hidden_size,))
        preprocessed_bias = torch.cat([query, key, value], dim=0)

        parameters = {"query_key_value": {}, "dense": {}}

        parameters["query_key_value"]["weight"] = preprocess_linear_weight(preprocessed_weight, dtype=ttnn.bfloat16)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(preprocessed_bias, dtype=ttnn.bfloat16)

        parameters["dense"]["weight"] = preprocess_linear_weight(torch_model.dense.weight, dtype=ttnn.bfloat16)
        parameters["dense"]["bias"] = preprocess_linear_bias(torch_model.dense.bias, dtype=ttnn.bfloat16)
    return parameters
