# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import torch
import time
import math
from typing import Tuple

import torch.utils.checkpoint
from torch.nn import functional as F
import transformers


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
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
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


# From transformers/models/bloom/modeling_bloom.py
def split_heads(query_key_value: torch.Tensor, num_heads) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
    storage as `query_key_value`

    Args:
        query_key_value (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

    Returns:
        query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
        value: [batch_size, seq_length, num_heads, head_dim]
    """
    batch_size, sequence_size, three_times_hidden_size = query_key_value.shape
    hidden_size = three_times_hidden_size // 3
    head_size = hidden_size // num_heads

    query_key_value = query_key_value.view(batch_size, sequence_size, 3, num_heads, head_size)
    return query_key_value[..., 0, :, :], query_key_value[..., 1, :, :], query_key_value[..., 2, :, :]


# From transformers/models/bloom/modeling_bloom.py
def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def create_query_key_value(hidden_states, weight, bias, num_heads):
    query_key_value = hidden_states @ weight
    query_key_value += bias
    query_layer, key_layer, value_layer = split_heads(query_key_value, num_heads)
    query_layer = torch.permute(query_layer, (0, 2, 1, 3))
    key_layer = torch.permute(key_layer, (0, 2, 3, 1))
    value_layer = torch.permute(value_layer, (0, 2, 1, 3))
    return query_layer, key_layer, value_layer


def compute_attention_scores(query_layer, key_layer, alibi):
    *_, head_size = query_layer.shape
    beta = 1.0
    inv_norm_factor = 1.0 / math.sqrt(head_size)
    matmul_result = beta * alibi + inv_norm_factor * (query_layer @ key_layer)
    return matmul_result


def compute_attention_probs(attention_scores, causal_mask):
    input_dtype = attention_scores.dtype
    fill_value = -100
    attention_weights = attention_scores * (1 + (causal_mask * -1)) + causal_mask * fill_value
    attention_probs = F.softmax(attention_weights, dim=-1, dtype=input_dtype)
    return attention_probs


# From transformers/models/bloom/modeling_bloom.py
def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """
    Merge heads together over the last dimension

    Args:
        x: (`torch.tensor`, *required*): [batch_size, num_heads, seq_length, head_dim]

    Returns:
        torch.tensor: [batch_size, seq_length, num_heads * head_dim]
    """
    # What we want to achieve is:
    # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
    batch_size, num_heads, seq_length, head_size = x.shape

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    x = x.permute(0, 2, 1, 3)

    # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
    return x.reshape(batch_size, seq_length, num_heads * head_size)


def compute_context_layer(attention_probs, value_layer):
    context_layer = attention_probs @ value_layer
    return merge_heads(context_layer)


def finalize_output(context_layer, output_weight, output_bias):
    output_tensor = context_layer @ output_weight
    output_tensor = output_tensor + output_bias
    return output_tensor


def multi_head_attention(
    hidden_states,
    alibi,
    causal_mask,
    query_key_value_weight,
    query_key_value_bias,
    output_weight,
    output_bias,
    *,
    num_heads,
):
    query_layer, key_layer, value_layer = create_query_key_value(
        hidden_states, query_key_value_weight, query_key_value_bias, num_heads
    )
    attention_scores = compute_attention_scores(query_layer, key_layer, alibi)
    attention_probs = compute_attention_probs(attention_scores, causal_mask)
    context_layer = compute_context_layer(attention_probs, value_layer)
    output_tensor = finalize_output(context_layer, output_weight, output_bias)
    return output_tensor


def mlp(
    hidden_states,
    dense_h_to_4h_weight,
    dense_h_to_4h_bias,
    dense_4h_to_h_weight,
    dense_4h_to_h_bias,
):
    hidden_states = hidden_states @ dense_h_to_4h_weight
    hidden_states = hidden_states + dense_h_to_4h_bias
    hidden_states = bloom_gelu_forward(hidden_states)
    hidden_states = hidden_states @ dense_4h_to_h_weight
    hidden_states = hidden_states + dense_4h_to_h_bias
    # hidden_states = F.dropout(hidden_states, p=0.0, training=False)
    return hidden_states


def bloom(input_ids, alibi, causal_mask, parameters, num_heads):
    inputs_embeds = F.embedding(input_ids, parameters.transformer.word_embeddings.weight)
    hidden_size = inputs_embeds.shape[2]

    hidden_states = F.layer_norm(
        inputs_embeds,
        (hidden_size,),
        parameters.transformer.word_embeddings_layernorm.weight,
        parameters.transformer.word_embeddings_layernorm.bias,
    )

    for layer_parameters in parameters.transformer.h:
        normalized_hidden_states = F.layer_norm(
            hidden_states,
            (hidden_size,),
            layer_parameters.input_layernorm.weight,
            layer_parameters.input_layernorm.bias,
        )

        attention_output = multi_head_attention(
            normalized_hidden_states,
            alibi,
            causal_mask,
            layer_parameters.self_attention.query_key_value.weight,
            layer_parameters.self_attention.query_key_value.bias,
            layer_parameters.self_attention.dense.weight,
            layer_parameters.self_attention.dense.bias,
            num_heads=num_heads,
        )
        attention_output += hidden_states

        normalized_attention_output = F.layer_norm(
            attention_output,
            (hidden_size,),
            layer_parameters.post_attention_layernorm.weight,
            layer_parameters.post_attention_layernorm.bias,
        )

        mlp_output = mlp(
            normalized_attention_output,
            layer_parameters.mlp.dense_h_to_4h.weight,
            layer_parameters.mlp.dense_h_to_4h.bias,
            layer_parameters.mlp.dense_4h_to_h.weight,
            layer_parameters.mlp.dense_4h_to_h.bias,
        )
        mlp_output += attention_output
        hidden_states = mlp_output

    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_size,),
        parameters.transformer.ln_f.weight,
        parameters.transformer.ln_f.bias,
    )
    return hidden_states


def bloom_for_causal_lm(input_ids, alibi, causal_mask, parameters, num_heads):
    start = time.time()
    hidden_states = bloom(input_ids, alibi, causal_mask, parameters, num_heads)
    end = time.time()
    batch_size, _ = input_ids.shape
    duration = end - start
    logger.info(f"Duration: {duration}")
    logger.info(f"Samples per second: {1 / duration * batch_size}")

    # return logits
    return hidden_states @ parameters.lm_head.weight


def preprocess_inputs(
    *,
    input_ids,
    num_heads,
    max_length,
    attention_mask=None,
):
    num_tokens = input_ids.shape[-1]
    padding_needed = (max_length - (num_tokens % max_length)) % max_length
    padded_input_ids = F.pad(input_ids, (0, padding_needed, 0, 0))

    if attention_mask is None:
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    attention_mask = F.pad(attention_mask, (0, padding_needed, 0, 0))

    alibi = build_alibi_tensor(attention_mask, num_heads, dtype=torch.float)

    batch_size, padded_seq_length = padded_input_ids.shape
    mask = torch.empty((padded_seq_length, padded_seq_length), dtype=torch.bool)
    seq_ids = torch.arange(padded_seq_length)
    mask[:, 0:] = seq_ids[:, None] < seq_ids[None, :]
    causal_mask = mask[None, None, :, :].expand(batch_size, num_heads, padded_seq_length, padded_seq_length)
    causal_mask = causal_mask.float()

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

        parameters["query_key_value"]["weight"] = preprocessed_weight.T.contiguous()
        parameters["query_key_value"]["bias"] = preprocessed_bias

        parameters["dense"]["weight"] = torch_model.dense.weight.T.contiguous()
        parameters["dense"]["bias"] = torch_model.dense.bias
    return parameters
