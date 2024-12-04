# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch import nn
from typing import Callable, Optional


def get_head_mask(
    head_mask: Optional[torch.Tensor],
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
) -> torch.Tensor:
    if head_mask is None:
        torch_head_mask = None

    if torch_head_mask is not None:
        torch_head_mask = self._convert_head_mask_to_5d(torch_head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            torch_head_mask = torch_head_mask.unsqueeze(-1)

    else:
        head_mask = [
            None,
        ] * num_hidden_layers

    return head_mask


def attention(config, x, mask, head_mask=None, output_attentions=None, *, parameters):
    query, key, value = x, x, x
    bs, q_length, dim = query.size()
    k_length = key.size(1)

    dim_per_head = config.dim // config.n_heads

    mask_reshp = (bs, 1, 1, k_length)

    def shape(x: torch.Tensor) -> torch.Tensor:
        """separate heads"""
        return x.view(bs, -1, config.n_heads, dim_per_head).transpose(1, 2)

    def unshape(x: torch.Tensor) -> torch.Tensor:
        """group heads"""
        return x.transpose(1, 2).contiguous().view(bs, -1, config.n_heads * dim_per_head)

    q = query @ parameters.q_lin.weight
    q = q + parameters.q_lin.bias
    q = shape(q)

    k = key @ parameters.k_lin.weight
    k = k + parameters.k_lin.bias
    k = shape(k)

    v = value @ parameters.v_lin.weight
    v = v + parameters.v_lin.bias
    v = shape(v)

    q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
    scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
    mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
    scores = scores.masked_fill(mask, torch.tensor(torch.finfo(scores.dtype).min))  # (bs, n_heads, q_length, k_length)

    weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)

    # Mask heads if we want to
    if head_mask is not None:
        weights = weights * head_mask

    context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
    context = unshape(context)  # (bs, q_length, dim)
    context = context @ parameters.out_lin.weight
    context = context + parameters.out_lin.bias

    return context


def ffn(configs, hidden_state, *, parameters):
    x = hidden_state @ parameters.lin1.weight
    x = x + parameters.lin1.bias

    x = torch.nn.functional.gelu(x)
    x = x @ parameters.lin2.weight
    x = x + parameters.lin2.bias
    return x


def transformer_block(
    config,
    x,
    attention_mask=None,
    head_mask=None,
    output_attentions: bool = False,
    *,
    parameters,
):
    sa_output = attention(config, x, attention_mask, head_mask, output_attentions, parameters=parameters.attention)
    if output_attentions:
        (sa_output, sa_weight) = sa_output
    else:
        sa_output = sa_output[0]
    sa_output = torch.nn.functional.layer_norm(
        sa_output + x, (config.dim,), parameters.sa_layer_norm.weight, parameters.sa_layer_norm.bias, eps=1e-12
    )

    ffn_output = ffn(config, sa_output, parameters=parameters.ffn)

    ffn_output = torch.nn.functional.layer_norm(
        ffn_output + sa_output,
        (config.dim,),
        parameters.output_layer_norm.weight,
        parameters.output_layer_norm.bias,
        eps=1e-12,
    )

    return ffn_output


def transformer(
    config,
    x,
    attention_mask=None,
    head_mask=None,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    *,
    parameters,
):
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    hidden_state = x
    i = 0
    for params in parameters.layer:
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        layer_outputs = transformer_block(
            config=config,
            x=hidden_state,
            attention_mask=attention_mask,
            head_mask=head_mask[i],
            output_attentions=output_attentions,
            parameters=params,
        )
        hidden_state = layer_outputs
        i += 1

    return hidden_state


def distilbert(
    config,
    input_ids=None,
    attention_mask=None,
    head_mask=None,
    inputs_embeds=None,
    output_attentions=None,
    output_hidden_states=None,
    position_ids=None,
    *,
    parameters,
):
    output_attentions = output_attentions if output_attentions is not None else config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else config.output_hidden_states

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    head_mask = get_head_mask(head_mask, config.num_hidden_layers)
    if attention_mask is None:
        attention_mask = torch.ones(input_shape, device=device)

    if input_ids is not None:
        inputs_embeds = torch.nn.functional.embedding(input_ids, parameters.embeddings.word_embeddings.weight)
    seq_length = inputs_embeds.size(1)
    if position_ids is not None:
        position_ids = position_ids[:, :seq_length]
    else:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

    position_embeddings = torch.nn.functional.embedding(position_ids, parameters.embeddings.position_embeddings.weight)
    embeddings = inputs_embeds + position_embeddings
    embeddings = torch.nn.functional.layer_norm(
        embeddings,
        (config.dim,),
        parameters.embeddings.LayerNorm.weight,
        parameters.embeddings.LayerNorm.bias,
        eps=1e-12,
    )
    return transformer(
        config,
        embeddings,
        attention_mask,
        head_mask,
        output_attentions,
        output_hidden_states,
        parameters=parameters.transformer,
    )


def distilbert_for_question_answering(
    config,
    input_ids,
    attention_mask,
    head_mask=None,
    inputs_embeds=None,
    start_positions=None,
    end_positions=None,
    output_attentions=None,
    output_hidden_states=None,
    position_ids=None,
    *,
    parameters,
    name="distilbert",
):
    distilbert_output = distilbert(
        config,
        input_ids,
        attention_mask,
        head_mask,
        inputs_embeds,
        output_attentions,
        output_hidden_states,
        position_ids=position_ids,
        parameters=parameters[name],
    )
    qa_outputs = distilbert_output
    qa_outputs = qa_outputs @ parameters.qa_outputs.weight
    qa_outputs = qa_outputs + parameters.qa_outputs.bias

    return qa_outputs
