# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import transformers

from models.experimental.functional_common.attention_mask_functions import (
    get_extended_attention_mask,
    invert_attention_mask,
)


def t5_layer_norm(config, hidden_states, *, weight):
    # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
    # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
    # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
    # half-precision inputs is done in fp32

    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + config.layer_norm_epsilon)

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        hidden_states = hidden_states.to(weight.dtype)

    return weight * hidden_states


def gelu_new(input_tensor):
    # TODO: compare against torch.nn.functional.gelu
    return transformers.activations.NewGELUActivation()(input_tensor)


def get_activation_function(dense_act_fn):
    if dense_act_fn == "relu":
        return torch.nn.functional.relu
    elif dense_act_fn == "gelu_new":
        return gelu_new
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
    hidden_states = hidden_states + forwarded_states
    return hidden_states


def t5_attention(
    config,
    hidden_states,
    key_value_states=None,
    mask=None,
    layer_head_mask=None,
    *,
    parameters,
):
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length, _ = hidden_states.shape

    def shape(states, head_size):
        """projection"""
        return states.view(batch_size, -1, config.num_heads, head_size).transpose(1, 2)

    def unshape(states, hidden_size):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, hidden_size)

    def project(hidden_states, weight):
        hidden_size = weight.shape[-1]
        head_size = hidden_size // config.num_heads
        """projects hidden states correctly to key/query states"""
        # self-attn
        # (batch_size, n_heads, seq_length, dim_per_head)
        hidden_states = shape(hidden_states @ weight, head_size)
        return hidden_states

    # get query states
    hidden_size = parameters.q.weight.shape[-1]
    query_states = project(hidden_states, parameters.q.weight)  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states if key_value_states is None else key_value_states,
        parameters.k.weight,
    )
    value_states = project(
        hidden_states if key_value_states is None else key_value_states,
        parameters.v.weight,
    )

    # compute scores
    scores = torch.matmul(
        query_states, key_states.transpose(3, 2)
    )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
    if mask is not None:
        scores += mask

    attn_weights = torch.nn.functional.softmax(scores.float(), dim=-1).type_as(
        scores
    )  # (batch_size, n_heads, seq_length, key_length)

    # Mask heads if we want to
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(torch.matmul(attn_weights, value_states), hidden_size)  # (batch_size, seq_length, dim)
    attn_output = attn_output @ parameters.o.weight

    return attn_output


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
    hidden_states = hidden_states + attention_output
    return hidden_states


def t5_layer_cross_attention(config, hidden_states, key_value_states, attention_mask=None, *, parameters):
    normed_hidden_states = t5_layer_norm(config, hidden_states, weight=parameters.layer_norm.weight)
    attention_output = t5_attention(
        config,
        normed_hidden_states,
        key_value_states,
        mask=attention_mask,
        parameters=parameters.EncDecAttention,
    )
    layer_output = hidden_states + attention_output
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

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16:
        clamp_value = torch.where(
            torch.isinf(hidden_states).any(),
            torch.finfo(hidden_states.dtype).max - 1000,
            torch.finfo(hidden_states.dtype).max,
        )
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    do_cross_attention = encoder_hidden_states is not None
    if do_cross_attention:
        hidden_states = t5_layer_cross_attention(
            config,
            hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            parameters=parameters.layer[1],
        )

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    # Apply Feed Forward layer
    hidden_states = t5_layer_ff(config, hidden_states, parameters.layer[-1])

    # clamp inf values to enable fp16 training
    if hidden_states.dtype == torch.float16:
        clamp_value = torch.where(
            torch.isinf(hidden_states).any(),
            torch.finfo(hidden_states.dtype).max - 1000,
            torch.finfo(hidden_states.dtype).max,
        )
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    return hidden_states  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


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
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])

    hidden_states = torch.nn.functional.embedding(input_ids, shared_embedding_weight)

    batch_size, seq_length = input_shape

    # required mask seq length can be calculated via length of past
    mask_seq_length = seq_length

    if attention_mask is None:
        attention_mask = torch.ones(batch_size, mask_seq_length, device=hidden_states.device)

    extended_attention_mask = get_extended_attention_mask(
        attention_mask, input_shape, is_decoder=encoder_hidden_states is not None
    )

    if encoder_hidden_states is not None:
        if encoder_attention_mask is None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=hidden_states.device, dtype=torch.long
            )

        encoder_extended_attention_mask = invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    for block_parameters in parameters.block:
        hidden_states = t5_block(
            config,
            hidden_states,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
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

    lm_logits = sequence_output @ parameters.lm_head.weight

    return lm_logits, encoder_last_hidden_state
