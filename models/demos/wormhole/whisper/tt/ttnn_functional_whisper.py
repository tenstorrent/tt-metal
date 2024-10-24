# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import transformers
import torch
from typing import Optional, Tuple

from torch.nn import functional as F
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias
import ttnn


def gelu(tensor):
    return ttnn.gelu(tensor)


def dropout(hidden_states, p, training):
    # ignored for inference
    return hidden_states


def calculate_key_values(config, key_value_states, *, parameters):
    bsz, tgt_len, hidden_size = key_value_states.shape
    bsz, tgt_len_padded, _ = key_value_states.shape.with_tile_padding()
    head_size = hidden_size // config.encoder_attention_heads

    fused_qkv = key_value_states @ parameters.key_value.weight + parameters.key_value.bias

    dtype = fused_qkv.dtype
    device = fused_qkv.device()
    fused_qkv = ttnn.to_torch(fused_qkv)
    fused_qkv = torch.reshape(fused_qkv, (bsz, tgt_len, 2, config.encoder_attention_heads, head_size))
    key_states, value_states = fused_qkv[..., 0, :, :], fused_qkv[..., 1, :, :]

    key_states = ttnn.from_torch(key_states, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    value_states = ttnn.from_torch(value_states, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    key_states = ttnn.permute(key_states, (0, 2, 1, 3))
    value_states = ttnn.permute(value_states, (0, 2, 1, 3))
    key_states = ttnn.to_layout(key_states, ttnn.TILE_LAYOUT)
    value_states = ttnn.to_layout(value_states, ttnn.TILE_LAYOUT)

    desired_shape = ttnn.Shape(
        [bsz, config.encoder_attention_heads, tgt_len, head_size],
        [bsz, config.encoder_attention_heads, tgt_len_padded, head_size],
    )
    key_states = ttnn.reshape(key_states, shape=desired_shape)
    value_states = ttnn.reshape(value_states, shape=desired_shape)

    return key_states, value_states


def split_query_key_value_and_split_heads(
    config, fused_qkv: ttnn.Tensor
) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    head_size = config.d_model // config.encoder_attention_heads
    batch_size, *_, seq_length, three_times_hidden_size = fused_qkv.shape
    batch_size, *_, padded_seq_length, three_times_hidden_size = fused_qkv.shape.with_tile_padding()
    hidden_size = three_times_hidden_size // 3
    encoder_attention_heads = hidden_size // head_size

    dtype = fused_qkv.dtype
    device = fused_qkv.device()
    fused_qkv = ttnn.to_torch(fused_qkv)
    fused_qkv = torch.reshape(fused_qkv, shape=(batch_size, seq_length, 3, encoder_attention_heads, head_size))
    query_states, key_states, value_states = fused_qkv[..., 0, :, :], fused_qkv[..., 1, :, :], fused_qkv[..., 2, :, :]

    query_states = ttnn.from_torch(query_states, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    key_states = ttnn.from_torch(key_states, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    value_states = ttnn.from_torch(value_states, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    query_states = ttnn.permute(query_states, (0, 2, 1, 3))
    key_states = ttnn.permute(key_states, (0, 2, 1, 3))
    value_states = ttnn.permute(value_states, (0, 2, 1, 3))

    query_states = ttnn.to_layout(query_states, ttnn.TILE_LAYOUT)
    key_states = ttnn.to_layout(key_states, ttnn.TILE_LAYOUT)
    value_states = ttnn.to_layout(value_states, ttnn.TILE_LAYOUT)

    desired_shape = ttnn.Shape(
        [batch_size, encoder_attention_heads, seq_length, head_size],
        [batch_size, encoder_attention_heads, padded_seq_length, head_size],
    )
    query_states = ttnn.reshape(query_states, shape=desired_shape)
    key_states = ttnn.reshape(key_states, shape=desired_shape)
    value_states = ttnn.reshape(value_states, shape=desired_shape)
    return query_states, key_states, value_states


def calculate_query_key_values(config, hidden_states, *, parameters):
    fused_qkv = hidden_states @ parameters.query_key_value.weight + parameters.query_key_value.bias
    return split_query_key_value_and_split_heads(config, fused_qkv)


def whisper_attention(config, hidden_states, attention_mask, key_value_states=None, *, parameters):
    head_size = config.d_model // config.encoder_attention_heads
    scaling = head_size**-0.5
    bsz, *_, padded_tgt_len, _ = hidden_states.shape.with_tile_padding()
    bsz, *_, tgt_len, _ = hidden_states.shape

    is_cross_attention = key_value_states is not None
    if is_cross_attention:
        query_states = hidden_states @ parameters.q_proj.weight + parameters.q_proj.bias

        dtype = query_states.dtype
        device = query_states.device()
        query_states = ttnn.to_torch(query_states)
        query_states = torch.reshape(query_states, (bsz, tgt_len, config.encoder_attention_heads, head_size))
        query_states = ttnn.from_torch(query_states, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))
        key_states, value_states = calculate_key_values(config, key_value_states, parameters=parameters)
        padded_key_value_tgt_len = key_states.shape.with_tile_padding()[2]
        key_value_tgt_len = key_states.shape[2]
    else:
        query_states, key_states, value_states = calculate_query_key_values(
            config, hidden_states, parameters=parameters
        )
        padded_key_value_tgt_len = padded_tgt_len
        key_value_tgt_len = tgt_len

    query_states *= scaling

    proj_shape = ttnn.Shape(
        [bsz * config.encoder_attention_heads, tgt_len, head_size],
        [bsz * config.encoder_attention_heads, padded_tgt_len, head_size],
    )
    query_states = ttnn.reshape(query_states, shape=proj_shape)
    proj_shape = ttnn.Shape(
        [bsz * config.encoder_attention_heads, key_value_tgt_len, head_size],
        [bsz * config.encoder_attention_heads, padded_key_value_tgt_len, head_size],
    )
    key_states = ttnn.reshape(key_states, shape=proj_shape)
    value_states = ttnn.reshape(value_states, shape=proj_shape)

    query_states = ttnn.to_layout(query_states, layout=ttnn.TILE_LAYOUT)
    key_states = ttnn.to_layout(key_states, layout=ttnn.TILE_LAYOUT)
    value_states = ttnn.to_layout(value_states, layout=ttnn.TILE_LAYOUT)

    attn_weights = query_states @ ttnn.permute(key_states, (0, 2, 1))
    if attention_mask is not None:
        bsz, _, tgt_len, src_len = attention_mask.shape
        attn_weights = ttnn.to_layout(attn_weights, layout=ttnn.ROW_MAJOR_LAYOUT)
        attn_weights = ttnn.reshape(attn_weights, shape=(bsz, config.encoder_attention_heads, tgt_len, src_len))
        attn_weights = ttnn.to_layout(attn_weights, layout=ttnn.TILE_LAYOUT)
        attn_weights = attn_weights + attention_mask
        attn_weights = ttnn.to_layout(attn_weights, layout=ttnn.ROW_MAJOR_LAYOUT)
        attn_weights = ttnn.reshape(attn_weights, shape=(bsz * config.encoder_attention_heads, tgt_len, src_len))
        attn_weights = ttnn.to_layout(attn_weights, layout=ttnn.TILE_LAYOUT)

    # differences in ttnn.softmax vs torch.softmax cause the attn_weights to be slightly different
    attn_weights = ttnn.softmax(attn_weights, dim=-1)

    attn_probs = dropout(attn_weights, p=0, training=False)
    attn_output = attn_probs @ value_states
    attn_output = ttnn.to_layout(attn_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    attn_output = ttnn.reshape(attn_output, shape=(bsz, config.encoder_attention_heads, tgt_len, head_size))
    attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))

    dtype = attn_output.dtype
    device = attn_output.device()
    attn_output = ttnn.to_torch(attn_output)
    attn_output = torch.reshape(attn_output, (bsz, tgt_len, config.d_model))
    attn_output = ttnn.from_torch(attn_output, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    attn_output = attn_output @ parameters.out_proj.weight + parameters.out_proj.bias
    return attn_output


def encoder_layer(config, hidden_states, *, parameters):
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.self_attn_layer_norm.weight,
        bias=parameters.self_attn_layer_norm.bias,
    )

    hidden_states = whisper_attention(config, hidden_states, attention_mask=None, parameters=parameters.self_attn)
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.final_layer_norm.weight,
        bias=parameters.final_layer_norm.bias,
    )
    hidden_states = hidden_states @ parameters.fc1.weight + parameters.fc1.bias
    hidden_states = gelu(hidden_states)
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = hidden_states @ parameters.fc2.weight + parameters.fc2.bias
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = residual + hidden_states

    # if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
    #     clamp_value = torch.finfo(hidden_states.dtype).max - 1000
    #     hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    return hidden_states


def encoder(config, inputs_embeds, *, parameters):
    hidden_states = inputs_embeds + parameters.embed_positions.weight
    hidden_states = dropout(hidden_states, p=0, training=False)

    for encoder_layer_parameter in parameters.layers:
        hidden_states = encoder_layer(config, hidden_states, parameters=encoder_layer_parameter)

    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layer_norm.weight,
        bias=parameters.layer_norm.bias,
    )
    return hidden_states


def make_causal_mask(input_ids_shape, dtype):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


def expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def decoder_layer(config, hidden_states, attention_mask, encoder_hidden_states, *, parameters):
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.self_attn_layer_norm.weight,
        bias=parameters.self_attn_layer_norm.bias,
    )

    hidden_states = whisper_attention(
        config,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        parameters=parameters.self_attn,
    )
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = residual + hidden_states

    # Cross-Attention Block
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.encoder_attn_layer_norm.weight,
        bias=parameters.encoder_attn_layer_norm.bias,
    )

    hidden_states = whisper_attention(
        config,
        hidden_states,
        attention_mask=None,
        key_value_states=encoder_hidden_states,
        parameters=parameters.encoder_attn,
    )

    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.final_layer_norm.weight,
        bias=parameters.final_layer_norm.bias,
    )
    hidden_states = hidden_states @ parameters.fc1.weight + parameters.fc1.bias
    hidden_states = gelu(hidden_states)
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = hidden_states @ parameters.fc2.weight + parameters.fc2.bias
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = residual + hidden_states

    return hidden_states


def prepare_decoder_attention_mask(attention_mask, input_shape, input_embeds):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None

    if input_shape[-1] > 1:
        combined_attention_mask = make_causal_mask(input_shape, input_embeds.dtype)

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = expand_mask(attention_mask, input_embeds.dtype, tgt_len=input_shape[-1])
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def decoder(config, hidden_states, decoder_attention_mask, encoder_hidden_states, *, parameters):
    hidden_states = dropout(hidden_states, p=0, training=False)

    for decoder_layer_parameter in parameters.layers:
        hidden_states = decoder_layer(
            config,
            hidden_states,
            decoder_attention_mask,
            encoder_hidden_states,
            parameters=decoder_layer_parameter,
        )

    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layer_norm.weight,
        bias=parameters.layer_norm.bias,
    )

    return hidden_states


def convert_to_ttnn(model, name):
    return name not in [
        "encoder.conv1",
        "encoder.conv2",
        "decoder.embed_tokens",
        "decoder.embed_positions",
    ]


def preprocess_encoder_inputs(input_features, *, parameters, device):
    def conv(input, weight, bias, stride=1, padding=1, dilation=1, groups=1):
        return F.conv1d(input, weight, bias, stride, padding, dilation, groups)

    input_embeds = torch.nn.functional.gelu(
        conv(
            input_features,
            weight=parameters.conv1.weight,
            bias=parameters.conv1.bias,
            padding=1,
        )
    )
    input_embeds = torch.nn.functional.gelu(
        conv(
            input_embeds,
            weight=parameters.conv2.weight,
            bias=parameters.conv2.bias,
            stride=2,
            padding=1,
        )
    )
    input_embeds = input_embeds.permute(0, 2, 1)
    input_embeds = ttnn.from_torch(input_embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return input_embeds


def preprocess_decoder_inputs(config, input_ids, attention_mask, *, parameters, device):
    input_shape = input_ids.size()
    input_ids = torch.reshape(input_ids, (-1, input_shape[-1]))
    inputs_embeds = F.embedding(input_ids, parameters.embed_tokens.weight)
    attention_mask = prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds)
    # ttnn cannot broadcast when adding on the batch or channel dimensions so this is a workaround
    attention_mask = attention_mask.expand(-1, config.encoder_attention_heads, -1, -1)

    positions = parameters.embed_positions.weight[0 : input_ids.shape[-1]]
    decoder_hidden_states = inputs_embeds + positions

    decoder_hidden_states = ttnn.from_torch(
        decoder_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    attention_mask = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return decoder_hidden_states, attention_mask


def preprocess_inputs(
    *,
    config,
    input_features,
    input_ids,
    attention_mask,
    parameters,
    device,
):
    input_embeds = preprocess_encoder_inputs(input_features, parameters=parameters.encoder, device=device)
    (decoder_hidden_states, attention_mask) = preprocess_decoder_inputs(
        config, input_ids, attention_mask, parameters=parameters.decoder, device=device
    )
    return input_embeds, decoder_hidden_states, attention_mask


def whisper(config, encoder_hidden_states, decoder_hidden_states, decoder_attention_mask, *, parameters):
    encoder_hidden_states = encoder(config, encoder_hidden_states, parameters=parameters.encoder)
    last_hidden_state = decoder(
        config,
        decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        parameters=parameters.decoder,
    )
    return last_hidden_state


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, transformers.models.whisper.modeling_whisper.WhisperAttention):
        height, width = torch_model.k_proj.weight.shape

        if "encoder_attn" in name:
            parameters = {"key_value": {}, "q_proj": {}, "out_proj": {}}
            preprocessed_weight = torch.cat([torch_model.k_proj.weight, torch_model.v_proj.weight], dim=0)
            preprocessed_bias = torch.cat([torch.zeros(height), torch_model.v_proj.bias], dim=0)
            parameters["key_value"]["weight"] = preprocess_linear_weight(preprocessed_weight, dtype=ttnn.bfloat16)
            parameters["key_value"]["bias"] = preprocess_linear_bias(preprocessed_bias, dtype=ttnn.bfloat16)
            parameters["q_proj"]["weight"] = preprocess_linear_weight(torch_model.q_proj.weight, dtype=ttnn.bfloat16)
            parameters["q_proj"]["bias"] = preprocess_linear_bias(torch_model.q_proj.bias, dtype=ttnn.bfloat16)
        else:
            parameters = {"query_key_value": {}, "out_proj": {}}
            preprocessed_weight = torch.cat(
                [torch_model.q_proj.weight, torch_model.k_proj.weight, torch_model.v_proj.weight], dim=0
            )
            preprocessed_bias = torch.cat(
                [torch_model.q_proj.bias, torch.zeros(height), torch_model.v_proj.bias], dim=0
            )
            parameters["query_key_value"]["weight"] = preprocess_linear_weight(preprocessed_weight, dtype=ttnn.bfloat16)
            parameters["query_key_value"]["bias"] = preprocess_linear_bias(preprocessed_bias, dtype=ttnn.bfloat16)

        parameters["out_proj"]["weight"] = preprocess_linear_weight(torch_model.out_proj.weight, dtype=ttnn.bfloat16)
        parameters["out_proj"]["bias"] = preprocess_linear_bias(torch_model.out_proj.bias, dtype=ttnn.bfloat16)
    elif name == "encoder.embed_positions" and isinstance(torch_model, torch.nn.Embedding):
        embeddings = ttnn.from_torch(torch_model.weight, dtype=ttnn.bfloat16)
        embeddings = ttnn.to_layout(embeddings, ttnn.TILE_LAYOUT)
        parameters["weight"] = embeddings
    return parameters
