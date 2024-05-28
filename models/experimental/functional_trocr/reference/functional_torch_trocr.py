# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple, Union
from loguru import logger
import torch
from torch import nn

from models.experimental.functional_trocr.activations import ACT2FN
from models.experimental.functional_trocr.trocr_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)


def _shape(tensor: torch.Tensor, seq_len: int, bsz: int, num_heads: int, head_dim: int):
    return tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()


def TrOCRAttention(
    config,
    hidden_states: torch.Tensor,
    embed_dim: int = 1024,
    num_heads: int = 16,
    kdim: int = None,
    vdim: int = None,
    dropout: float = 0.0,
    is_decoder: bool = False,
    bias: bool = True,
    is_cross_attention: bool = False,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    parameters=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    embed_dim = embed_dim
    kdim = kdim if kdim is not None else embed_dim
    vdim = vdim if vdim is not None else embed_dim
    num_heads = num_heads
    dropout = dropout
    head_dim = embed_dim // num_heads
    if not (head_dim * num_heads == embed_dim):
        raise ValueError(
            f"embed_dim must be divisible by num_heads (got `embed_dim`: {embed_dim} and `num_heads`:" f" {num_heads})."
        )
    scaling = head_dim**-0.5
    is_decoder = is_decoder

    k_proj = nn.Linear(kdim, embed_dim, bias=bias)
    v_proj = nn.Linear(vdim, embed_dim, bias=bias)
    q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    q_proj.weight = nn.Parameter(parameters.q_proj.weight)
    q_proj.bias = nn.Parameter(parameters.q_proj.bias)
    k_proj.weight = nn.Parameter(parameters.k_proj.weight)
    k_proj.bias = nn.Parameter(parameters.k_proj.bias)
    v_proj.weight = nn.Parameter(parameters.v_proj.weight)
    v_proj.bias = nn.Parameter(parameters.v_proj.bias)

    """Input shape: Batch x Time x Channel"""
    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None
    bsz, tgt_len, embed_dim = hidden_states.size()
    # get query proj
    query_states = hidden_states @ parameters.q_proj.weight
    query_states = query_states + parameters.q_proj.bias
    query_states = query_states * scaling
    # get key, value proj
    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        k_p = key_value_states @ parameters.k_proj.weight
        k_p = k_p + parameters.k_proj.bias
        v_p = key_value_states @ parameters.v_proj.weight
        v_p = v_p + parameters.v_proj.bias
        key_states = _shape(k_p, -1, bsz, num_heads, head_dim)
        value_states = _shape(v_p, -1, bsz, num_heads, head_dim)
    elif past_key_value is not None:
        # reuse k, v, self_attention
        k_p = hidden_states @ parameters.k_proj.weight
        k_p = k_p + parameters.k_proj.bias
        v_p = hidden_states @ parameters.v_proj.weight
        v_p = v_p + parameters.v_proj.bias
        key_states = _shape(k_p, -1, bsz, num_heads, head_dim)
        value_states = _shape(v_p, -1, bsz, num_heads, head_dim)
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)
    else:
        # self_attention
        k_p = hidden_states @ parameters.k_proj.weight
        k_p = k_p + parameters.k_proj.bias
        v_p = hidden_states @ parameters.v_proj.weight
        v_p = v_p + parameters.v_proj.bias
        key_states = _shape(k_p, -1, bsz, num_heads, head_dim)
        value_states = _shape(v_p, -1, bsz, num_heads, head_dim)

    if is_decoder:
        past_key_value = (key_states, value_states)

    proj_shape = (bsz * num_heads, -1, head_dim)
    query_states = _shape(query_states, tgt_len, bsz, num_heads, head_dim).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * num_heads, tgt_len, src_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        if layer_head_mask.size() != (num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(num_heads,)}, but is" f" {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_weights = attn_weights.view(bsz * num_heads, tgt_len, src_len)

    if output_attentions:
        attn_weights_reshaped = attn_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=dropout, training=False)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * num_heads, tgt_len, head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, num_heads, tgt_len, head_dim)}, but is" f" {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, num_heads, tgt_len, head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = attn_output @ parameters.out_proj.weight
    attn_output = attn_output + parameters.out_proj.bias

    return attn_output, attn_weights_reshaped, past_key_value


def TrOCRDecoderLayer(
    config,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = True,
    parameters=None,
):
    embed_dim = config.hidden_size
    dropout = config.dropout
    activation_fn = ACT2FN[config.activation_function]
    activation_dropout = config.activation_dropout

    self_attn_layer_norm = nn.LayerNorm(embed_dim)
    self_attn_layer_norm.weight = nn.Parameter(parameters.self_attn_layer_norm.weight)
    self_attn_layer_norm.bias = nn.Parameter(parameters.self_attn_layer_norm.bias)

    fc1 = nn.Linear(embed_dim, config.decoder_ffn_dim)
    fc1.weight = nn.Parameter(parameters.fc1.weight)
    fc1.bias = nn.Parameter(parameters.fc1.bias)

    fc2 = nn.Linear(config.decoder_ffn_dim, embed_dim)
    fc2.weight = nn.Parameter(parameters.fc2.weight)
    fc2.bias = nn.Parameter(parameters.fc2.bias)

    final_layer_norm = nn.LayerNorm(embed_dim)
    final_layer_norm.weight = nn.Parameter(parameters.final_layer_norm.weight)
    final_layer_norm.bias = nn.Parameter(parameters.final_layer_norm.bias)

    residual = hidden_states

    # Self Attention
    # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    # add present self-attn cache to positions 1,2 of present_key_value tuple
    hidden_states, self_attn_weights, present_key_value = TrOCRAttention(
        config=config,
        hidden_states=hidden_states,
        past_key_value=self_attn_past_key_value,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
        embed_dim=embed_dim,
        num_heads=config.decoder_attention_heads,
        dropout=config.attention_dropout,
        is_decoder=True,
        parameters=parameters.self_attn,
    )

    hidden_states = nn.functional.dropout(hidden_states, p=dropout, training=False)
    hidden_states = residual + hidden_states
    hidden_states = self_attn_layer_norm(hidden_states)

    # Cross-Attention Block
    cross_attn_present_key_value = None
    cross_attn_weights = None

    if encoder_hidden_states is not None:
        residual = hidden_states

        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
        encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
        encoder_attn_layer_norm.weight = nn.Parameter(parameters.encoder_attn_layer_norm.weight)
        encoder_attn_layer_norm.bias = nn.Parameter(parameters.encoder_attn_layer_norm.bias)

        hidden_states, cross_attn_weights, cross_attn_present_key_value = TrOCRAttention(
            config,
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=cross_attn_past_key_value,
            output_attentions=output_attentions,
            embed_dim=embed_dim,
            num_heads=config.decoder_attention_heads,
            kdim=config.cross_attention_hidden_size,
            vdim=config.cross_attention_hidden_size,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_cross_attention=True,
            parameters=parameters.encoder_attn,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=dropout, training=False)
        hidden_states = residual + hidden_states
        hidden_states = encoder_attn_layer_norm(hidden_states)

        # add cross-attn to positions 3,4 of present_key_value tuple
        present_key_value = present_key_value + cross_attn_present_key_value

    # Fully Connected
    residual = hidden_states
    hidden_states = hidden_states @ parameters.fc1.weight
    hidden_states = hidden_states + parameters.fc1.bias
    hidden_states = activation_fn(hidden_states)
    hidden_states = hidden_states @ parameters.fc2.weight
    hidden_states = hidden_states + parameters.fc2.bias

    hidden_states = residual + hidden_states
    hidden_states = final_layer_norm(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights, cross_attn_weights)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def TrOCRPositionalEmbedding(
    input_ids: torch.Tensor,
    num_embeddings: int = 512,
    embedding_dim: int = 1024,
    past_key_values_length: int = 0,
    parameters=None,
):
    offset = 2
    embed = nn.Embedding(num_embeddings + offset, embedding_dim, _weight=parameters)
    bsz, seq_len = input_ids.shape[:2]
    positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long).expand(bsz, -1)

    return embed(positions + offset)


def TrOCRDecoder(
    config,
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
    parameters=None,
):
    dropout = config.dropout
    layerdrop = config.decoder_layerdrop
    padding_idx = config.pad_token_id
    embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
    training = False
    embed_tokens = nn.Embedding(
        config.vocab_size, config.hidden_size, padding_idx, _weight=parameters.embed_tokens.weight
    )

    if config.layernorm_embedding:
        layernorm_embedding = nn.LayerNorm(config.hidden_size)
        layernorm_embedding.weight.data = parameters.layernorm_embedding.weight
        layernorm_embedding.bias.data = parameters.layernorm_embedding.bias
    else:
        layernorm_embedding = None

    gradient_checkpointing = False

    output_attentions = output_attentions if output_attentions is not None else config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else config.output_hidden_states
    use_cache = use_cache if use_cache is not None else config.use_cache
    return_dict = return_dict if return_dict is not None else config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        input = input_ids
        input_ids = input_ids.view(-1, input.shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        input = inputs_embeds[:, :, -1]
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    if inputs_embeds is None:
        inputs_embeds = embed_tokens(input_ids) * embed_scale

    if config.use_learned_position_embeddings:
        embed_pos = TrOCRPositionalEmbedding(
            input,
            config.max_position_embeddings,
            config.hidden_size,
            past_key_values_length=past_key_values_length,
            parameters=parameters.embed_positions.weight,
        )

    hidden_states = inputs_embeds + embed_pos
    hidden_states = layernorm_embedding(hidden_states)

    input_shape = input.shape

    attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )

    # expand encoder attention mask
    if encoder_hidden_states is not None and encoder_attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        encoder_attention_mask = _prepare_4d_attention_mask(
            encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        )

    if gradient_checkpointing and training:
        if use_cache:
            logger.warning_once(
                "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
    next_decoder_cache = () if use_cache else None

    # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
    for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
        if attn_mask is not None:
            if attn_mask.size()[0] != (config.de):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
    idx = 0
    for decoder_layer_params in parameters.layers:
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if training:
            dropout_probability = torch.rand([])
            if dropout_probability < layerdrop:
                continue

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        layer_outputs = TrOCRDecoderLayer(
            config,
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            parameters=decoder_layer_params,
        )
        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

            if encoder_hidden_states is not None:
                all_cross_attentions += (layer_outputs[2],)
        idx += 1

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

    return hidden_states


def TrOCRForCausalLM(
    config,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    parameters=None,
) -> Union[Tuple, torch.Tensor]:
    decoder_output = TrOCRDecoder(
        config,
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        head_mask=head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        parameters=parameters.model.decoder,
    )

    logits = decoder_output @ parameters.output_projection.weight

    return logits
