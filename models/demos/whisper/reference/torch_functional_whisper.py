# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import transformers
from transformers import AutoFeatureExtractor, WhisperModel
from datasets import load_dataset
import torch
from typing import Optional

from torch.nn import functional as F
from ttnn.model_preprocessing import preprocess_model_parameters
from loguru import logger


def conv(input, weight, bias, stride=1, padding=1, dilation=1, groups=1):
    return F.conv1d(input, weight, bias, stride, padding, dilation, groups)


def gelu(tensor):
    return torch.nn.functional.gelu(tensor)


def dropout(hidden_states, p, training):
    return hidden_states
    # return torch.nn.functional.dropout(hidden_states, p=p, training=training)


def calculate_key_values(config, key_value_states, parameters):
    bsz, tgt_len, hidden_size = key_value_states.size()
    head_size = hidden_size // config.encoder_attention_heads

    fused_qkv = key_value_states @ parameters.key_value.weight + parameters.key_value.bias
    fused_qkv = torch.reshape(fused_qkv, shape=(bsz, tgt_len, 2, config.encoder_attention_heads, head_size))
    key_states, value_states = fused_qkv[..., 0, :, :], fused_qkv[..., 1, :, :]

    key_states = torch.reshape(key_states, shape=(bsz, tgt_len, config.encoder_attention_heads, head_size))
    key_states = torch.permute(key_states, (0, 2, 1, 3))
    key_states = key_states.contiguous()

    value_states = torch.reshape(value_states, shape=(bsz, tgt_len, config.encoder_attention_heads, head_size))
    value_states = torch.permute(value_states, (0, 2, 1, 3))
    value_states = value_states.contiguous()

    return key_states, value_states


def split_query_key_value_and_split_heads(config, fused_qkv):
    head_size = config.d_model // config.encoder_attention_heads
    batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
    hidden_size = three_times_hidden_size // 3
    num_heads = hidden_size // head_size

    fused_qkv = torch.reshape(fused_qkv, shape=(batch_size, seq_length, 3, num_heads, head_size))
    query_states, key_states, value_states = fused_qkv[..., 0, :, :], fused_qkv[..., 1, :, :], fused_qkv[..., 2, :, :]

    query_states = torch.reshape(query_states, shape=(batch_size, seq_length, num_heads, head_size))
    query_states = torch.permute(query_states, (0, 2, 1, 3))

    key_states = torch.reshape(key_states, shape=(batch_size, seq_length, num_heads, head_size))
    key_states = torch.permute(key_states, (0, 2, 1, 3))

    value_states = torch.reshape(value_states, shape=(batch_size, seq_length, num_heads, head_size))
    value_states = torch.permute(value_states, (0, 2, 1, 3))

    return query_states, key_states, value_states


def calculate_query_key_values(config, hidden_states, *, parameters):
    fused_qkv = hidden_states @ parameters.query_key_value.weight + parameters.query_key_value.bias
    return split_query_key_value_and_split_heads(config, fused_qkv)


def whisper_attention(config, hidden_states, attention_mask, key_value_states, *, parameters):
    head_size = config.d_model // config.encoder_attention_heads
    scaling = head_size**-0.5
    bsz, tgt_len, _ = hidden_states.size()

    is_cross_attention = key_value_states is not None
    if is_cross_attention:
        query_states = hidden_states @ parameters.q_proj.weight + parameters.q_proj.bias
        query_states = torch.reshape(query_states, shape=(bsz, tgt_len, config.encoder_attention_heads, head_size))
        query_states = torch.permute(query_states, (0, 2, 1, 3))
        key_states, value_states = calculate_key_values(config, key_value_states, parameters=parameters)
    else:
        query_states, key_states, value_states = calculate_query_key_values(
            config, hidden_states, parameters=parameters
        )
    query_states *= scaling

    proj_shape = (bsz * config.encoder_attention_heads, -1, head_size)
    query_states = torch.reshape(query_states, shape=proj_shape)
    key_states = torch.reshape(key_states, shape=proj_shape)
    value_states = torch.reshape(value_states, shape=proj_shape)

    attn_weights = query_states @ torch.permute(key_states, (0, 2, 1))
    if attention_mask is not None:
        bsz, _, tgt_len, src_len = attention_mask.size()
        attn_weights = (
            torch.reshape(attn_weights, shape=(bsz, config.encoder_attention_heads, tgt_len, src_len)) + attention_mask
        )
        attn_weights = torch.reshape(attn_weights, shape=(bsz * config.encoder_attention_heads, tgt_len, src_len))

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

    attn_probs = dropout(attn_weights, p=0, training=False)
    attn_output = attn_probs @ value_states
    attn_output = torch.reshape(attn_output, shape=(bsz, config.encoder_attention_heads, tgt_len, head_size))
    attn_output = torch.permute(attn_output, (0, 2, 1, 3))
    attn_output = attn_output.reshape(bsz, tgt_len, config.d_model)
    attn_output = attn_output @ parameters.out_proj.weight + parameters.out_proj.bias
    return attn_output


def encoder_layer(config, hidden_states, *, parameters):
    residual = hidden_states

    hidden_states = F.layer_norm(
        hidden_states,
        (config.d_model,),
        parameters.self_attn_layer_norm.weight,
        parameters.self_attn_layer_norm.bias,
    )
    hidden_states = whisper_attention(
        config, hidden_states, attention_mask=None, key_value_states=None, parameters=parameters.self_attn
    )
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = residual + hidden_states

    residual = hidden_states

    hidden_states = F.layer_norm(
        hidden_states,
        (config.d_model,),
        parameters.final_layer_norm.weight,
        parameters.final_layer_norm.bias,
    )
    hidden_states = hidden_states @ parameters.fc1.weight + parameters.fc1.bias
    hidden_states = gelu(hidden_states)
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = hidden_states @ parameters.fc2.weight + parameters.fc2.bias
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = residual + hidden_states

    if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    return hidden_states


def encoder(config, inputs_embeds, *, parameters):
    hidden_states = inputs_embeds + parameters.embed_positions.weight
    hidden_states = dropout(hidden_states, p=0, training=False)

    for encoder_layer_parameter in parameters.layers:
        hidden_states = encoder_layer(config, hidden_states, parameters=encoder_layer_parameter)

    hidden_states = F.layer_norm(
        hidden_states,
        (config.d_model,),
        parameters.layer_norm.weight,
        parameters.layer_norm.bias,
    )
    return hidden_states


def encoder_original(config, input_features, *, parameters):
    inputs_embeds = gelu(
        conv(
            input_features,
            weight=parameters.conv1.weight,
            bias=parameters.conv1.bias,
            padding=1,
        )
    )
    inputs_embeds = gelu(
        conv(
            inputs_embeds,
            weight=parameters.conv2.weight,
            bias=parameters.conv2.bias,
            stride=2,
            padding=1,
        )
    )
    inputs_embeds = inputs_embeds.permute(0, 2, 1)
    hidden_states = inputs_embeds + parameters.embed_positions.weight
    hidden_states = dropout(hidden_states, p=0, training=False)

    for encoder_layer_parameter in parameters.layers:
        hidden_states = encoder_layer(
            config,
            hidden_states,
            parameters=encoder_layer_parameter,
        )

    hidden_states = F.layer_norm(
        hidden_states,
        (config.d_model,),
        parameters.layer_norm.weight,
        parameters.layer_norm.bias,
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
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def decoder_layer(config, hidden_states, attention_mask, encoder_hidden_states, *, parameters):
    residual = hidden_states
    hidden_states = F.layer_norm(
        hidden_states,
        (config.d_model,),
        parameters.self_attn_layer_norm.weight,
        parameters.self_attn_layer_norm.bias,
    )

    hidden_states = whisper_attention(
        config,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        key_value_states=None,
        parameters=parameters.self_attn,
    )
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = residual + hidden_states

    # Cross-Attention Block
    residual = hidden_states
    hidden_states = F.layer_norm(
        hidden_states,
        (config.d_model,),
        parameters.encoder_attn_layer_norm.weight,
        parameters.encoder_attn_layer_norm.bias,
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
    hidden_states = F.layer_norm(
        hidden_states,
        (config.d_model,),
        parameters.final_layer_norm.weight,
        parameters.final_layer_norm.bias,
    )
    hidden_states = hidden_states @ parameters.fc1.weight + parameters.fc1.bias
    hidden_states = gelu(hidden_states)
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = hidden_states @ parameters.fc2.weight + parameters.fc2.bias
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = residual + hidden_states

    return hidden_states


def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None

    if input_shape[-1] > 1:
        combined_attention_mask = make_causal_mask(input_shape, inputs_embeds.dtype)

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
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

    hidden_states = F.layer_norm(
        hidden_states,
        (config.d_model,),
        parameters.layer_norm.weight,
        parameters.layer_norm.bias,
    )

    return hidden_states


def decoder_original(config, input_ids, attention_mask, encoder_hidden_states, parameters):
    input_shape = input_ids.size()
    input_ids = torch.reshape(input_ids, (-1, input_shape[-1]))
    inputs_embeds = F.embedding(input_ids, parameters.embed_tokens.weight)
    attention_mask = prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds)
    positions = parameters.embed_positions.weight[0 : input_ids.shape[-1]]

    hidden_states = inputs_embeds + positions
    hidden_states = dropout(hidden_states, p=0, training=False)

    for decoder_layer_parameter in parameters.layers:
        hidden_states = decoder_layer(
            config,
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            parameters=decoder_layer_parameter,
        )

    hidden_states = F.layer_norm(
        hidden_states,
        (config.d_model,),
        parameters.layer_norm.weight,
        parameters.layer_norm.bias,
    )

    return hidden_states


def preprocess_encoder_inputs(input_features, parameters):
    inputs_embeds = gelu(
        conv(
            input_features,
            weight=parameters.conv1.weight,
            bias=parameters.conv1.bias,
            padding=1,
        )
    )
    inputs_embeds = gelu(
        conv(
            inputs_embeds,
            weight=parameters.conv2.weight,
            bias=parameters.conv2.bias,
            stride=2,
            padding=1,
        )
    )

    inputs_embeds = inputs_embeds.permute(0, 2, 1)
    return inputs_embeds


def preprocess_decoder_inputs(input_ids, attention_mask, *, parameters):
    input_shape = input_ids.size()
    input_ids = torch.reshape(input_ids, (-1, input_shape[-1]))
    inputs_embeds = F.embedding(input_ids, parameters.embed_tokens.weight)
    attention_mask = prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds)

    positions = parameters.embed_positions.weight[0 : input_ids.shape[-1]]
    decoder_hidden_states = inputs_embeds + positions

    return decoder_hidden_states, attention_mask


def preprocess_inputs(
    *,
    input_features,
    input_ids,
    attention_mask,
    parameters,
):
    input_embeds = preprocess_encoder_inputs(input_features, parameters.encoder)
    (decoder_hidden_states, attention_mask) = preprocess_decoder_inputs(
        input_ids, attention_mask, parameters=parameters.decoder
    )
    return input_embeds, decoder_hidden_states, attention_mask


def whisper_original(config, input_features, decoder_input_ids, attention_mask, *, parameters):
    encoder_hidden_states = encoder_original(config, input_features, parameters=parameters.encoder)
    return decoder_original(
        config,
        input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        parameters=parameters.decoder,
    )


def whisper(config, input_embeds, decoder_hidden_states, decoder_attention_mask, *, parameters):
    encoder_hidden_states = encoder(config, input_embeds, parameters=parameters.encoder)
    return decoder(
        config,
        decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        parameters=parameters.decoder,
    )


def custom_preprocessor(torch_model, name):
    parameters = {}
    if isinstance(torch_model, transformers.models.whisper.modeling_whisper.WhisperAttention):
        if "encoder_attn" in name:
            parameters = {"key_value": {}, "q_proj": {}, "out_proj": {}}
            preprocessed_weight = torch.cat([torch_model.k_proj.weight, torch_model.v_proj.weight], dim=0)
            preprocessed_bias = torch.cat([torch.zeros_like(torch_model.v_proj.bias), torch_model.v_proj.bias], dim=0)
            parameters["key_value"]["weight"] = preprocessed_weight.T.contiguous()
            parameters["key_value"]["bias"] = preprocessed_bias
            parameters["q_proj"]["weight"] = torch_model.q_proj.weight.T.contiguous()
            parameters["q_proj"]["bias"] = torch_model.q_proj.bias
        else:
            parameters = {"query_key_value": {}, "out_proj": {}}
            preprocessed_weight = torch.cat(
                [torch_model.q_proj.weight, torch_model.k_proj.weight, torch_model.v_proj.weight], dim=0
            )
            preprocessed_bias = torch.cat(
                [torch_model.q_proj.bias, torch.zeros_like(torch_model.v_proj.bias), torch_model.v_proj.bias], dim=0
            )
            parameters["query_key_value"]["weight"] = preprocessed_weight.T.contiguous()
            parameters["query_key_value"]["bias"] = preprocessed_bias

        parameters["out_proj"]["weight"] = torch_model.out_proj.weight.T.contiguous()
        parameters["out_proj"]["bias"] = torch_model.out_proj.bias
    return parameters


if __name__ == "__main__":
    # The following is simply to visualize the operations from pytorch
    # sudo apt install graphviz
    # pip install graphviz torchview
    from torchview import draw_graph
    from datasets import load_dataset

    model_name = "openai/whisper-base"
    model = WhisperModel.from_pretrained(model_name).to(torch.bfloat16).eval()

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    decoder_input_ids = torch.ones(1, 1).type(torch.int32) * model.config.decoder_start_token_id

    model_graph = draw_graph(
        model,
        input_size=((1, 80, 3000), (1, 2), (1, 80)),
        dtypes=[torch.bfloat16, torch.int64, torch.int64],
        expand_nested=True,
        depth=10,
        directory="out",
    )
    model_graph.visual_graph.render(format="svg")

    # Sanity check the torch functional approach
    parameters = preprocess_model_parameters(
        model_name=f"torch_{model_name}",
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        convert_to_ttnn=lambda *_: False,
    )
    last_hidden_state = whisper_original(
        model.config, input_features, decoder_input_ids, attention_mask=None, parameters=parameters
    )
    logger.info(last_hidden_state.shape)
    last_three = last_hidden_state[0, -1, -3:]
    logger.info(last_three)
