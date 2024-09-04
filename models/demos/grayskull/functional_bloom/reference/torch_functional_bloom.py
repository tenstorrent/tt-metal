# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from typing import Tuple

import transformers
from ttnn.model_preprocessing import ParameterDict
from torch.nn import functional as F
from transformers.models.bloom.configuration_bloom import BloomConfig


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


# From transformers/models/bloom/modeling_bloom.py
def split_heads(query_key_value: torch.Tensor, num_heads: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
    storage as `query_key_value`

    Args:
        query_key_value (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

    Returns:
        query: [batch_size, seq_length, num_heads, head_dim]
        key: [batch_size, seq_length, num_heads, head_dim]
        value: [batch_size, seq_length, num_heads, head_dim]
    """
    batch_size, sequence_size, three_times_hidden_size = query_key_value.shape
    hidden_size = three_times_hidden_size // 3
    head_size = hidden_size // num_heads

    query_key_value = query_key_value.view(batch_size, sequence_size, 3, num_heads, head_size)
    return (
        query_key_value[..., 0, :, :],
        query_key_value[..., 1, :, :],
        query_key_value[..., 2, :, :],
    )


# From transformers/models/bloom/modeling_bloom.py
def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """
    Merge heads together over the last dimension

    Args:
        x: (`torch.tensor`, *required*): [batch_size, num_heads, seq_length, head_dim]

    Returns:
        torch.tensor: [batch_size, seq_length, num_heads * head_dim]
    """
    batch_size, num_heads, seq_length, head_size = x.shape

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    x = x.permute(0, 2, 1, 3)

    # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
    return x.reshape(batch_size, seq_length, num_heads * head_size)


def make_causal_mask(attention_mask: torch.Tensor, input_ids_shape: torch.Size):
    batch_size, target_length = input_ids_shape
    mask = torch.empty((target_length, target_length), dtype=torch.bool)
    seq_ids = torch.arange(target_length)
    mask[:, :] = seq_ids[:, None] < seq_ids[None, :]
    causal_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length)

    expanded_mask = ~(attention_mask[:, None, None, :].to(torch.bool))
    expanded_mask = expanded_mask.expand(batch_size, 1, target_length, target_length)

    return expanded_mask | causal_mask


# From transformers/models/bloom/modeling_bloom.py
def bloom_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def create_query_key_value(
    config: BloomConfig, hidden_states: torch.Tensor, *, parameters: ParameterDict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_key_value = hidden_states @ parameters.query_key_value.weight
    query_key_value += parameters.query_key_value.bias

    query_layer, key_layer, value_layer = split_heads(query_key_value, config.n_head)
    query_layer = torch.permute(query_layer, (0, 2, 1, 3))
    key_layer = torch.permute(key_layer, (0, 2, 3, 1))
    value_layer = torch.permute(value_layer, (0, 2, 1, 3))

    return query_layer, key_layer, value_layer


def bloom_attention(
    config: BloomConfig,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    parameters: ParameterDict,
) -> torch.Tensor:
    query_layer, key_layer, value_layer = create_query_key_value(config, hidden_states, parameters=parameters)

    *_, head_size = query_layer.shape
    beta = 1.0
    inv_norm_factor = 1.0 / math.sqrt(head_size)
    attention_scores = beta * alibi + inv_norm_factor * (query_layer @ key_layer)

    fill_value = -100
    attention_weights = attention_scores * (1 + (attention_mask * -1)) + attention_mask * fill_value
    attention_probs = F.softmax(attention_weights, dim=-1, dtype=attention_scores.dtype)

    context_layer = merge_heads(attention_probs @ value_layer)
    output_tensor = context_layer @ parameters.dense.weight + parameters.dense.bias
    output_tensor += residual

    return output_tensor


def bloom_mlp(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    *,
    parameters: ParameterDict,
) -> torch.Tensor:
    output_tensor = hidden_states @ parameters.dense_h_to_4h.weight
    output_tensor += parameters.dense_h_to_4h.bias
    output_tensor = bloom_gelu(output_tensor)
    output_tensor = output_tensor @ parameters.dense_4h_to_h.weight
    output_tensor += parameters.dense_4h_to_h.bias
    output_tensor += residual

    return output_tensor


def bloom_block(
    config: BloomConfig,
    hidden_states: torch.Tensor,
    alibi: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    parameters: ParameterDict,
) -> torch.Tensor:
    layernorm_output = F.layer_norm(
        input=hidden_states,
        normalized_shape=(config.hidden_size,),
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

    layernorm_output = F.layer_norm(
        input=attention_output,
        normalized_shape=(config.hidden_size,),
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
    inputs_embeds = F.embedding(input_ids, parameters.word_embeddings.weight)
    hidden_size = inputs_embeds.shape[2]

    hidden_states = F.layer_norm(
        inputs_embeds,
        (hidden_size,),
        parameters.word_embeddings_layernorm.weight,
        parameters.word_embeddings_layernorm.bias,
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
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_size,),
        parameters.ln_f.weight,
        parameters.ln_f.bias,
    )

    return hidden_states


def bloom_for_causal_lm(config: BloomConfig, input_ids, alibi, causal_mask, *, parameters):
    bloom_output = bloom(
        config,
        input_ids,
        alibi,
        causal_mask,
        parameters=parameters.transformer,
    )

    # return logits
    return bloom_output @ parameters.lm_head.weight


def bloom_for_question_answering(
    config,
    input_ids,
    alibi,
    causal_mask,
    *,
    parameters,
):
    bloom_output = bloom(
        config,
        input_ids,
        alibi,
        causal_mask,
        parameters=parameters.transformer,
    )

    qa_outputs = bloom_output
    qa_outputs = qa_outputs @ parameters.qa_outputs.weight
    qa_outputs = qa_outputs + parameters.qa_outputs.bias
    return qa_outputs


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

        parameters["query_key_value"]["weight"] = preprocessed_weight.T
        parameters["query_key_value"]["bias"] = preprocessed_bias

        parameters["dense"]["weight"] = torch_model.dense.weight.T
        parameters["dense"]["bias"] = torch_model.dense.bias
    return parameters
