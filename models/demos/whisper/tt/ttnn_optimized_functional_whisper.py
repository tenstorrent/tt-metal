# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import transformers
from typing import Optional
from torch.nn import functional as F
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias


WHISPER_DTYPE = ttnn.bfloat8_b


def dropout(hidden_states, p, training):
    # ignored for inference
    return hidden_states


# The split_query_key_value_and_split_heads requires the query to have the same volume as the key and values
# This is not the case however for whisper so we currently cannot swap out calculate_key_values below
# def calculate_key_values(config, query_states, key_value_states, *, parameters):
#     fused_kv = key_value_states @ parameters.key_value.weight + parameters.key_value.bias
#     head_size = config.d_model // config.encoder_attention_heads
#     batch_size, *_, _, two_times_hidden_size = fused_kv.shape.with_tile_padding()
#     hidden_size = two_times_hidden_size // 2
#     encoder_attention_heads = hidden_size // head_size
#     query_states, key_states, value_states = ttnn.transformer.split_query_key_value_and_split_heads(
#         query_states,
#         kv_input_tensor=fused_kv,
#         num_heads=encoder_attention_heads,
#         memory_config=WHISPER_MEMORY_CONFIG,
#     )
#     key_states = ttnn.permute(key_states, (0, 1, 3, 2))
#     return query_states, key_states, value_states


def calculate_key_values(config, key_value_states, *, parameters, whisper_memory_config):
    bsz, tgt_len, hidden_size = key_value_states.shape
    bsz, tgt_len_padded, _ = key_value_states.shape.with_tile_padding()
    head_size = hidden_size // config.encoder_attention_heads

    fused_qkv = ttnn.linear(
        key_value_states,
        parameters.weight,
        bias=parameters.bias,
        memory_config=whisper_memory_config,
    )
    dtype = fused_qkv.dtype
    device = fused_qkv.device()

    # fused_qkv = ttnn.to_layout(fused_qkv, layout=ttnn.ROW_MAJOR_LAYOUT)
    # fused_qkv = ttnn.from_device(fused_qkv)
    # fused_qkv = ttnn.reshape(fused_qkv, (bsz, tgt_len, config.encoder_attention_heads, 2, head_size))
    # # Without Split: 0.84 pcc
    # key_states = ttnn.reshape(fused_qkv, (bsz, tgt_len, config.encoder_attention_heads, head_size * 2))[..., :head_size]
    # value_states = ttnn.reshape(fused_qkv, (bsz, tgt_len, config.encoder_attention_heads, head_size * 2))[..., head_size:]

    # key_states = ttnn.to_device(key_states, device)
    # key_states = ttnn.to_layout(key_states, ttnn.TILE_LAYOUT)
    # key_states = ttnn.permute(key_states, (0, 2, 3, 1))

    # value_states = ttnn.to_device(value_states, device)
    # value_states = ttnn.to_layout(value_states, ttnn.TILE_LAYOUT)
    # value_states = ttnn.permute(value_states, (0, 2, 1, 3))

    fused_qkv = ttnn.to_layout(fused_qkv, layout=ttnn.ROW_MAJOR_LAYOUT)
    fused_qkv = ttnn.from_device(fused_qkv)
    fused_qkv = ttnn.reshape(fused_qkv, (bsz, tgt_len, 2, config.encoder_attention_heads, head_size))
    fused_qkv = ttnn.to_layout(fused_qkv, layout=ttnn.TILE_LAYOUT)
    fused_qkv = ttnn.to_device(fused_qkv, device=device)

    # #13672: Slice op Not supported for 5d tensors.
    fused_qkv = ttnn.to_torch(fused_qkv)
    key_states, value_states = fused_qkv[..., 0, :, :], fused_qkv[..., 1, :, :]  #
    key_states = ttnn.from_torch(key_states, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    value_states = ttnn.from_torch(value_states, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    key_states = ttnn.permute(key_states, (0, 2, 3, 1))
    value_states = ttnn.permute(value_states, (0, 2, 1, 3))
    key_states = ttnn.to_layout(key_states, ttnn.TILE_LAYOUT)
    value_states = ttnn.to_layout(value_states, ttnn.TILE_LAYOUT)

    desired_shape = ttnn.Shape(
        [bsz, config.encoder_attention_heads, head_size, tgt_len],
        [bsz, config.encoder_attention_heads, head_size, tgt_len_padded],
    )
    key_states = ttnn.reshape(key_states, shape=desired_shape)

    desired_shape = ttnn.Shape(
        [bsz, config.encoder_attention_heads, tgt_len, head_size],
        [bsz, config.encoder_attention_heads, tgt_len_padded, head_size],
    )
    value_states = ttnn.reshape(value_states, shape=desired_shape)

    return key_states, value_states


def calculate_query_key_values(config, hidden_states, *, parameters, whisper_memory_config):
    fused_qkv = ttnn.linear(
        hidden_states,
        parameters.weight,
        bias=parameters.bias,
    )

    return ttnn.transformer.split_query_key_value_and_split_heads(
        fused_qkv, memory_config=whisper_memory_config, num_heads=config.num_attention_heads
    )


def whisper_attention(
    config, device, hidden_states, attention_mask, key_value_states=None, *, parameters, whisper_memory_config
):
    head_size = config.d_model // config.encoder_attention_heads
    scaling = head_size**-0.5
    bsz, *_, tgt_len, _ = hidden_states.shape

    is_cross_attention = key_value_states is not None
    if is_cross_attention:
        query_states = ttnn.linear(
            hidden_states,
            parameters.q_proj.weight,
            bias=parameters.q_proj.bias,
            memory_config=whisper_memory_config,
        )
        query_states = ttnn.to_layout(query_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        query_states = ttnn.from_device(query_states)
        query_states = ttnn.reshape(query_states, (bsz, tgt_len, config.encoder_attention_heads, head_size))
        query_states = ttnn.to_layout(query_states, layout=ttnn.TILE_LAYOUT)
        query_states = ttnn.to_device(query_states, device=device)
        query_states = ttnn.permute(query_states, (0, 2, 1, 3))
        key_states, value_states = calculate_key_values(
            config, key_value_states, parameters=parameters.key_value, whisper_memory_config=whisper_memory_config
        )
    else:
        query_states, key_states, value_states = calculate_query_key_values(
            config, hidden_states, parameters=parameters.query_key_value, whisper_memory_config=whisper_memory_config
        )

    query_states *= scaling
    attn_weights = ttnn.matmul(query_states, key_states)

    if attention_mask is not None:
        attn_weights = ttnn.add(attn_weights, attention_mask)

    # differences in ttnn.softmax vs torch.softmax cause the attn_weights to be slightly different
    attn_weights = ttnn.softmax(attn_weights, dim=-1)

    attn_probs = dropout(attn_weights, p=0, training=False)
    attn_output = ttnn.matmul(attn_probs, value_states, memory_config=whisper_memory_config)

    ttnn.deallocate(attn_probs)
    ttnn.deallocate(attn_weights)
    ttnn.deallocate(query_states)

    attn_output = ttnn.transformer.concatenate_heads(attn_output)

    attn_output = ttnn.linear(
        attn_output,
        parameters.out_proj.weight,
        bias=parameters.out_proj.bias,
        memory_config=whisper_memory_config,
    )

    return attn_output


def encoder_layer(config, device, hidden_states, *, parameters, whisper_memory_config):
    residual = hidden_states

    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.self_attn_layer_norm.weight,
        bias=parameters.self_attn_layer_norm.bias,
        memory_config=whisper_memory_config,
    )

    hidden_states = whisper_attention(
        config,
        device,
        hidden_states,
        attention_mask=None,
        parameters=parameters.self_attn,
        whisper_memory_config=whisper_memory_config,
    )
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = ttnn.add(residual, hidden_states)

    residual = hidden_states

    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.final_layer_norm.weight,
        bias=parameters.final_layer_norm.bias,
        memory_config=whisper_memory_config,
    )

    hidden_states = ttnn.linear(
        hidden_states,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
    )

    hidden_states = ttnn.gelu(hidden_states, memory_config=whisper_memory_config)
    hidden_states = dropout(hidden_states, p=0, training=False)

    hidden_states = ttnn.linear(
        hidden_states,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=whisper_memory_config,
    )

    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = ttnn.add(residual, hidden_states)

    # if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
    #     clamp_value = torch.finfo(hidden_states.dtype).max - 1000
    #     hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    return hidden_states


def encoder(config, device, inputs_embeds, *, parameters, whisper_memory_config):
    hidden_states = ttnn.add(inputs_embeds, parameters.embed_positions.weight)
    hidden_states = dropout(hidden_states, p=0, training=False)

    for encoder_layer_parameter in parameters.layers:
        hidden_states = encoder_layer(
            config,
            device,
            hidden_states,
            parameters=encoder_layer_parameter,
            whisper_memory_config=whisper_memory_config,
        )

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


def decoder_layer(
    config, device, hidden_states, attention_mask, encoder_hidden_states, *, parameters, whisper_memory_config
):
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.self_attn_layer_norm.weight,
        bias=parameters.self_attn_layer_norm.bias,
        memory_config=whisper_memory_config,
    )

    hidden_states = whisper_attention(
        config,
        device,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        parameters=parameters.self_attn,
        whisper_memory_config=whisper_memory_config,
    )
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = ttnn.add(residual, hidden_states)

    # Cross-Attention Block
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.encoder_attn_layer_norm.weight,
        bias=parameters.encoder_attn_layer_norm.bias,
    )

    hidden_states = whisper_attention(
        config,
        device,
        hidden_states,
        attention_mask=None,
        key_value_states=encoder_hidden_states,
        parameters=parameters.encoder_attn,
        whisper_memory_config=whisper_memory_config,
    )

    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = ttnn.add(residual, hidden_states)

    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.final_layer_norm.weight,
        bias=parameters.final_layer_norm.bias,
    )

    hidden_states = ttnn.linear(
        hidden_states, parameters.fc1.weight, bias=parameters.fc1.bias, memory_config=whisper_memory_config
    )
    hidden_states = ttnn.gelu(hidden_states, memory_config=whisper_memory_config)
    hidden_states = dropout(hidden_states, p=0, training=False)

    hidden_states = ttnn.linear(
        hidden_states, parameters.fc2.weight, bias=parameters.fc2.bias, memory_config=whisper_memory_config
    )
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = ttnn.add(residual, hidden_states)
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


def decoder(
    config, device, hidden_states, decoder_attention_mask, encoder_hidden_states, *, parameters, whisper_memory_config
):
    hidden_states = dropout(hidden_states, p=0, training=False)

    for decoder_layer_parameter in parameters.layers:
        hidden_states = decoder_layer(
            config,
            device,
            hidden_states,
            decoder_attention_mask,
            encoder_hidden_states,
            parameters=decoder_layer_parameter,
            whisper_memory_config=whisper_memory_config,
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


def preprocess_encoder_inputs(input_features, *, parameters, device, whisper_memory_config):
    def conv(input, weight, bias, stride=1, padding=1, dilation=1, groups=1):
        return F.conv1d(input, weight, bias, stride, padding, dilation, groups)

    def ttnn_conv1d(
        device,
        tt_input_tensor,
        weights,
        conv_params,
        bias,
        *,
        output_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
        deallocate_activation=True,
        act_block_h=32,
        height_sharding=True,
        use_shallow_conv_variant=False,
        fp32_accum=False,
        packer_l1_acc=False,
        debug=False,
        groups=1,
        math_approx=False,
        activation="",
        reallocate_halo=False,
        reshard=False,
        whisper_memory_config=ttnn.L1_MEMORY_CONFIG,
    ):
        weights = ttnn.from_torch(weights, dtype=ttnn.float32)
        bias = ttnn.from_torch(bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.float32)

        conv_config = ttnn.Conv1dConfig(
            dtype=output_dtype,
            weights_dtype=weights_dtype,
            math_approx_mode_enabled=math_approx,
            fp32_dest_acc_enabled=fp32_accum,
            packer_l1_accum_enabled=packer_l1_acc,
            activation=activation,
            input_channels_alignment=(16 if use_shallow_conv_variant else 32),
            deallocate_activation=deallocate_activation,
            reallocate_halo_output=reallocate_halo,
            act_block_h_override=act_block_h,
            reshard_if_not_optimal=reshard,
            shard_layout=(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            ),
            math_fidelity=math_fidelity,
        )

        [tt_output_tensor_on_device, out_length, weights_device, bias_device] = ttnn.Conv1d(
            input_tensor=tt_input_tensor,
            weight_tensor=weights,
            in_channels=tt_input_tensor.shape[-1],
            out_channels=weights.shape[0],
            device=device,
            bias_tensor=bias,
            kernel_size=3,
            stride=conv_params[0],
            padding=conv_params[1],
            batch_size=tt_input_tensor.shape[0],
            input_length=tt_input_tensor.shape[1],
            conv_config=conv_config,
            conv_op_cache={},
            debug=debug,
            groups=groups,
        )
        tt_output_tensor_on_device = ttnn.squeeze(tt_output_tensor_on_device, 0)
        tt_output_tensor_on_device = ttnn.to_layout(tt_output_tensor_on_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_output_tensor_on_device = ttnn.reshape(
            tt_output_tensor_on_device, (tt_input_tensor.shape[0], out_length, tt_output_tensor_on_device.shape[-1])
        )
        tt_output_tensor = ttnn.from_device(tt_output_tensor_on_device)

        return tt_output_tensor

    if parameters.conv1.weight.shape[0] == 512:
        input_features = ttnn.from_torch(input_features, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        input_features = ttnn.permute(input_features, (0, 2, 1))
        conv1 = ttnn_conv1d(
            device,
            input_features,
            parameters.conv1.weight,
            [1, 1],
            parameters.conv1.bias,
        )
        conv1 = ttnn.to_layout(conv1, ttnn.TILE_LAYOUT)
        conv1 = ttnn.to_device(conv1, device)
        conv1 = ttnn.permute(conv1, (0, 2, 1))

    else:
        conv1 = conv(
            input_features.float(),
            weight=parameters.conv1.weight,
            bias=parameters.conv1.bias,
            stride=1,
            padding=1,
        )
        conv1 = ttnn.from_torch(conv1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    input_embeds = ttnn.gelu(conv1, memory_config=whisper_memory_config)
    input_embeds = ttnn.to_layout(input_embeds, layout=ttnn.ROW_MAJOR_LAYOUT)

    # input_embeds = ttnn.permute(input_embeds, (0, 2, 1))
    input_embeds = ttnn.to_torch(input_embeds)

    # #13529 ttnn.conv1d throws OOM here.
    # conv2 = ttnn_conv1d(
    #     device,
    #     input_embeds,
    #     parameters.conv2.weight,
    #     [2, 1],
    #     parameters.conv2.bias,
    # )
    # conv2 = ttnn.to_layout(conv2, ttnn.TILE_LAYOUT)
    # conv2 = ttnn.to_device(conv2, device)
    # conv2 = ttnn.permute(conv2, (0, 2, 1))
    # input_embeds = ttnn.gelu(conv2, memory_config=whisper_memory_config)

    conv = conv(
        input_embeds.float(),
        weight=parameters.conv2.weight,
        bias=parameters.conv2.bias,
        stride=2,
        padding=1,
    )
    conv = ttnn.from_torch(conv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    input_embeds = ttnn.gelu(conv, memory_config=whisper_memory_config)
    input_embeds = ttnn.permute(input_embeds, (0, 2, 1))

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


def preprocess_inputs(*, config, input_features, input_ids, attention_mask, parameters, device, whisper_memory_config):
    input_embeds = preprocess_encoder_inputs(
        input_features, parameters=parameters.encoder, device=device, whisper_memory_config=whisper_memory_config
    )
    (decoder_hidden_states, attention_mask) = preprocess_decoder_inputs(
        config, input_ids, attention_mask, parameters=parameters.decoder, device=device
    )
    return input_embeds, decoder_hidden_states, attention_mask


def whisper(
    config,
    device,
    encoder_hidden_states,
    decoder_hidden_states,
    decoder_attention_mask,
    *,
    parameters,
    whisper_memory_config,
):
    encoder_hidden_states = encoder(
        config,
        device,
        encoder_hidden_states,
        parameters=parameters.encoder,
        whisper_memory_config=whisper_memory_config,
    )

    last_hidden_state = decoder(
        config,
        device,
        decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        parameters=parameters.decoder,
        whisper_memory_config=whisper_memory_config,
    )

    return last_hidden_state


def whisper_for_audio_classification(config, inputs_embeds, *, parameters, device, batch_size, whisper_memory_config):
    encoder_outputs = encoder(
        config=config,
        device=device,
        inputs_embeds=inputs_embeds,
        parameters=parameters.encoder,
        whisper_memory_config=whisper_memory_config,
    )
    hidden_states = ttnn.linear(
        encoder_outputs,
        parameters.projector.weight,
        bias=parameters.projector.bias,
        memory_config=whisper_memory_config,
    )
    pooled_output = ttnn.mean(hidden_states, dim=-2, keepdim=True)

    logits = ttnn.linear(
        pooled_output,
        parameters.classifier.weight,
        bias=parameters.classifier.bias,
        memory_config=whisper_memory_config,
    )
    return logits


def whisper_for_conditional_generation(
    config,
    input_embeds,
    decoder_hidden_states,
    decoder_attention_mask,
    *,
    parameters,
    device,
    ttnn_linear_weight,
    whisper_memory_config,
):
    output = whisper(
        config=config,
        device=device,
        encoder_hidden_states=input_embeds,
        decoder_hidden_states=decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        parameters=parameters,
        whisper_memory_config=whisper_memory_config,
    )

    ttnn_output = ttnn.matmul(
        output,
        ttnn_linear_weight,
        dtype=ttnn.bfloat16,
    )
    return ttnn_output


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


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
        embeddings = torch_model.weight.unsqueeze(0).expand(8, -1, -1)
        embeddings = ttnn.from_torch(embeddings, dtype=ttnn.bfloat16)
        embeddings = ttnn.to_layout(embeddings, ttnn.TILE_LAYOUT)
        parameters["weight"] = embeddings

    return parameters
