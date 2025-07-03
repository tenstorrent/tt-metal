# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import transformers
from loguru import logger
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn
from models.utility_functions import nearest_32

WHISPER_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG

WHISPER_L1_SMALL_SIZE = 1024


def gelu(tensor):
    return ttnn.gelu(tensor, memory_config=WHISPER_MEMORY_CONFIG)


def dropout(hidden_states, p, training):
    # ignored for inference
    return hidden_states


def init_kv_cache(config, device, max_batch_size, max_seq_len, n_layers=None):
    """
    Generates empty KV cache and sends to device
    """

    logger.info(f"Initializing KV cache with max batch size: {max_batch_size} and max sequence length: {max_seq_len}")

    kv_cache = []
    if n_layers is None:
        n_layers = config.decoder_layers
    for i in range(n_layers):
        kv_cache_layer = []
        for j in range(2):
            cache_k_or_v = torch.zeros(
                (
                    max_batch_size,
                    config.decoder_attention_heads,
                    max_seq_len,
                    config.d_model // config.decoder_attention_heads,
                )
            )
            cache_k_or_v = ttnn.as_tensor(
                cache_k_or_v,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=None,
                cache_file_name=None,
            )
            kv_cache_layer.append(cache_k_or_v)
        kv_cache.append(kv_cache_layer)

    return kv_cache


def calculate_key_values(config, key_value_states, *, parameters):
    bsz, tgt_len, hidden_size = key_value_states.shape
    head_size = hidden_size // config.encoder_attention_heads

    fused_kv = key_value_states @ parameters.key_value.weight + parameters.key_value.bias
    fused_kv = ttnn.unsqueeze_to_4D(fused_kv)  # 1, 1, S, 2xHxd

    key_states = fused_kv[:, :, :, :hidden_size]
    key_states = ttnn.transpose(key_states, 2, 3)  # 1, 1, Hxd, S
    key_states = ttnn.reshape(key_states, (bsz, config.encoder_attention_heads, head_size, tgt_len))  # 1, H, d, S

    value_states = fused_kv[:, :, :, hidden_size:]
    value_states = ttnn.transpose(value_states, 1, 2)  # 1, S, 1, Hxd
    value_states = ttnn.reshape(value_states, (bsz, tgt_len, config.encoder_attention_heads, head_size))
    value_states = ttnn.transpose(value_states, 1, 2)  # 1, H, S, d

    return key_states, value_states


def get_decode_sdpa_configs(config, device):
    head_size = config.d_model // config.decoder_attention_heads
    padded_num_heads = nearest_32(config.decoder_attention_heads)

    # Q, K, V are batch sharded across cores (currently only supporting batch 1)
    sdpa_batch_sharded_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            grid=ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        # Volume must match batch size
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(0, 0),
                    ),
                }
            ),
            shard_shape=[
                padded_num_heads,
                head_size,
            ],
            shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    sdpa_decode_progcfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        exp_approx_mode=False,
        q_chunk_size=256,
        k_chunk_size=256,
    )

    sdpa_decode_compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    return sdpa_batch_sharded_memcfg, sdpa_decode_progcfg, sdpa_decode_compute_kernel_config


def functional_sdpa(query_states, key_states, value_states, scaling, attention_mask):
    query_states *= scaling

    attn_weights = query_states @ key_states

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = ttnn.softmax(attn_weights, dim=-1, memory_config=WHISPER_MEMORY_CONFIG)

    attn_probs = dropout(attn_weights, p=0, training=False)
    attn_output = attn_probs @ value_states
    return attn_output


def whisper_attention(
    config,
    hidden_states,
    attention_mask,
    is_decode,
    encoder_hidden_states=None,
    kv_cache=None,
    current_decode_pos=None,
    *,
    parameters,
):
    head_size = config.d_model // config.encoder_attention_heads
    scaling = head_size**-0.5
    bsz, *_, tgt_len, _ = hidden_states.shape

    is_cross_attention = encoder_hidden_states is not None
    sdpa_with_kv_cache = not is_cross_attention and is_decode and kv_cache is not None

    if is_cross_attention:
        query_states = hidden_states @ parameters.q_proj.weight + parameters.q_proj.bias
        query_states = ttnn.unsqueeze_to_4D(query_states)
        query_states = ttnn.transpose(query_states, 1, 2)  # 1, 32, 1, Hxd
        query_states = ttnn.reshape(query_states, (bsz, tgt_len, config.encoder_attention_heads, head_size))
        query_states = ttnn.transpose(query_states, 1, 2)  # 1, H, 32, d
        key_states, value_states = calculate_key_values(config, encoder_hidden_states, parameters=parameters)
        attn_output = functional_sdpa(query_states, key_states, value_states, scaling, attention_mask)
    else:
        fused_qkv = hidden_states @ parameters.query_key_value.weight + parameters.query_key_value.bias  # 1, S, 3xHxd
        fused_qkv = ttnn.unsqueeze_to_4D(fused_qkv)
        (
            query_states,  # 1, H, S, d
            key_states,  # 1, H, d, S
            value_states,  # 1, H, S, d
        ) = ttnn.experimental.nlp_create_qkv_heads(
            fused_qkv,
            num_heads=config.decoder_attention_heads,
            num_kv_heads=config.decoder_attention_heads,
            transpose_k_heads=(not sdpa_with_kv_cache),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if sdpa_with_kv_cache:
            k_cache = kv_cache[0]  # 1, H, MaxS, d
            v_cache = kv_cache[1]  # 1, H, MaxS, d

            # Reshape qkv to 1, S, H, D
            query_states = ttnn.transpose(query_states, 1, 2)
            key_states = ttnn.transpose(key_states, 1, 2)
            value_states = ttnn.transpose(value_states, 1, 2)

            # Unpad batch
            unpadded_batch_size = current_decode_pos.shape[0]
            query_states = query_states[:, :unpadded_batch_size, :, :]
            key_states = key_states[:, :unpadded_batch_size, :, :]
            value_states = value_states[:, :unpadded_batch_size, :, :]

            sdpa_batch_sharded_memcfg, sdpa_decode_progcfg, sdpa_decode_compute_kernel_config = get_decode_sdpa_configs(
                config, hidden_states.device()
            )

            # Convert to sharded (required by paged_update_cache and sdpa ops)
            query_states = ttnn.interleaved_to_sharded(query_states, sdpa_batch_sharded_memcfg)
            key_states = ttnn.interleaved_to_sharded(key_states, sdpa_batch_sharded_memcfg)
            value_states = ttnn.interleaved_to_sharded(value_states, sdpa_batch_sharded_memcfg)

            # Update KV cache
            ttnn.experimental.paged_update_cache(
                k_cache, key_states, update_idxs_tensor=current_decode_pos, page_table=None
            )
            ttnn.experimental.paged_update_cache(
                v_cache, value_states, update_idxs_tensor=current_decode_pos, page_table=None
            )

            attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
                query_states,
                k_cache,
                v_cache,
                cur_pos_tensor=current_decode_pos,
                scale=scaling,
                program_config=sdpa_decode_progcfg,
                compute_kernel_config=sdpa_decode_compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # 1, 1, H, D

            attn_output = ttnn.transpose(attn_output, 1, 2)
        else:
            attn_output = functional_sdpa(query_states, key_states, value_states, scaling, attention_mask)

    attn_output = ttnn.experimental.nlp_concat_heads(attn_output)
    attn_output = ttnn.squeeze(attn_output, 0)
    attn_output = attn_output[:, :tgt_len, :]

    attn_output = attn_output @ parameters.out_proj.weight + parameters.out_proj.bias
    return attn_output


def encoder_layer(config, hidden_states, *, parameters):
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.self_attn_layer_norm.weight,
        bias=parameters.self_attn_layer_norm.bias,
        memory_config=WHISPER_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    hidden_states = whisper_attention(
        config, hidden_states, attention_mask=None, is_decode=False, parameters=parameters.self_attn
    )
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.final_layer_norm.weight,
        bias=parameters.final_layer_norm.bias,
        memory_config=WHISPER_MEMORY_CONFIG,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )
    hidden_states = hidden_states @ parameters.fc1.weight + parameters.fc1.bias
    hidden_states = gelu(hidden_states)
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = hidden_states @ parameters.fc2.weight + parameters.fc2.bias
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = residual + hidden_states

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
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
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
    config, hidden_states, attention_mask, encoder_hidden_states, kv_cache=None, current_decode_pos=None, *, parameters
):
    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.self_attn_layer_norm.weight,
        bias=parameters.self_attn_layer_norm.bias,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    hidden_states = whisper_attention(
        config,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        is_decode=True,
        kv_cache=kv_cache,
        current_decode_pos=current_decode_pos,
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
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    hidden_states = whisper_attention(
        config,
        hidden_states,
        attention_mask=None,
        is_decode=True,
        encoder_hidden_states=encoder_hidden_states,
        kv_cache=kv_cache,
        current_decode_pos=current_decode_pos,
        parameters=parameters.encoder_attn,
    )

    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.final_layer_norm.weight,
        bias=parameters.final_layer_norm.bias,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )
    hidden_states = hidden_states @ parameters.fc1.weight + parameters.fc1.bias
    hidden_states = gelu(hidden_states)
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = hidden_states @ parameters.fc2.weight + parameters.fc2.bias
    hidden_states = dropout(hidden_states, p=0, training=False)
    hidden_states = residual + hidden_states

    return hidden_states


def prepare_decoder_attention_mask(attention_mask, input_shape, dtype=torch.bfloat16):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None

    if input_shape[-1] > 1:
        combined_attention_mask = make_causal_mask(input_shape, dtype)

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = expand_mask(attention_mask, dtype, tgt_len=input_shape[-1])
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


def decoder(
    config,
    hidden_states,
    decoder_attention_mask,
    encoder_hidden_states,
    kv_cache=None,
    current_decode_pos=None,
    *,
    parameters,
):
    hidden_states = dropout(hidden_states, p=0, training=False)

    if kv_cache is not None:
        assert current_decode_pos is not None, "current_decode_pos must be provided when using kv_cache"

    for i, decoder_layer_parameter in enumerate(parameters.layers):
        hidden_states = decoder_layer(
            config,
            hidden_states,
            decoder_attention_mask,
            encoder_hidden_states,
            kv_cache=kv_cache[i] if kv_cache is not None else None,
            current_decode_pos=current_decode_pos,
            parameters=decoder_layer_parameter,
        )

    hidden_states = ttnn.layer_norm(
        hidden_states,
        weight=parameters.layer_norm.weight,
        bias=parameters.layer_norm.bias,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4),
    )

    return hidden_states


def convert_to_ttnn(model, name):
    return name not in [
        "encoder.conv1",
        "encoder.conv2",
    ]


def get_conv_configs(device):
    conv1d_config = ttnn.Conv1dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    conv1d_compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    return conv1d_config, conv1d_compute_config


def prepare_conv_weights(config, parameters):
    conv2_out_channel_splits = 4
    conv2_out_channels = config.d_model // conv2_out_channel_splits
    if isinstance(parameters.conv1.weight, torch.Tensor):
        parameters.conv1.weight = ttnn.from_torch(
            parameters.conv1.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
    if isinstance(parameters.conv2.weight, torch.Tensor):
        parameters.conv2.weight = ttnn.from_torch(
            parameters.conv2.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        # Split output channels to avoid running out of L1 memory
        weight_splits = []
        for i in range(conv2_out_channel_splits):
            weight_splits.append(parameters.conv2.weight[i * conv2_out_channels : (i + 1) * conv2_out_channels, :, :])
        parameters.conv2.weight = weight_splits
    return conv2_out_channel_splits, conv2_out_channels


def preprocess_encoder_inputs(config, input_features, *, parameters, device):
    input_length = input_features.shape[-1]
    input_features = ttnn.from_torch(input_features, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_features = ttnn.transpose(input_features, 1, 2)

    conv1d_config, conv1d_compute_config = get_conv_configs(device)

    # First time convs are runs, weights are on host (convs will return weights on device)
    conv2_out_channel_splits, conv2_out_channels = prepare_conv_weights(config, parameters)

    input_embeds, [weights_device, _] = ttnn.conv1d(
        input_tensor=input_features,
        weight_tensor=parameters.conv1.weight,
        device=device,
        in_channels=config.num_mel_bins,
        out_channels=config.d_model,
        batch_size=1,
        input_length=input_length,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        conv_config=conv1d_config,
        compute_config=conv1d_compute_config,
        return_weights_and_bias=True,
    )
    parameters.conv1.weight = weights_device
    input_embeds = ttnn.gelu(input_embeds)
    input_embeds = ttnn.sharded_to_interleaved(input_embeds)

    out_tensor_splits = []
    for i in range(conv2_out_channel_splits):
        out_split, [weights_device, _] = ttnn.conv1d(
            input_tensor=input_embeds,
            weight_tensor=parameters.conv2.weight[i],
            device=device,
            in_channels=config.d_model,
            out_channels=conv2_out_channels,
            batch_size=1,
            input_length=input_length,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            groups=1,
            conv_config=conv1d_config,
            compute_config=conv1d_compute_config,
            return_weights_and_bias=True,
        )
        parameters.conv2.weight[i] = weights_device
        out_split = ttnn.sharded_to_interleaved(out_split)
        out_tensor_splits.append(out_split)

    input_embeds = ttnn.concat(out_tensor_splits, dim=3)
    input_embeds = ttnn.gelu(input_embeds)
    input_embeds = ttnn.squeeze(input_embeds, 0)

    return input_embeds


def preprocess_decoder_inputs(
    config, input_ids, attention_mask, *, parameters, device, decode_pos=None, create_attention_mask=True
):
    input_shape = input_ids.size()
    input_ids = torch.reshape(input_ids, (-1, input_shape[-1]))
    tt_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    inputs_embeds = ttnn.embedding(
        tt_input_ids, parameters.embed_tokens.weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    if create_attention_mask:
        attention_mask = prepare_decoder_attention_mask(attention_mask, input_shape)
    if attention_mask is not None:
        # ttnn cannot broadcast when adding on the batch or channel dimensions so this is a workaround
        attention_mask = attention_mask.expand(-1, config.decoder_attention_heads, -1, -1)
        attention_mask = ttnn.from_torch(attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    if decode_pos is None:
        positions = parameters.embed_positions.weight[0 : input_ids.shape[-1]]
    else:
        positions = parameters.embed_positions.weight[decode_pos : decode_pos + 1]

    positions = ttnn.to_layout(positions, ttnn.TILE_LAYOUT)
    decoder_hidden_states = inputs_embeds + positions

    return decoder_hidden_states, attention_mask


def preprocess_inputs(
    *,
    config,
    input_features,
    input_ids,
    attention_mask,
    parameters,
    device,
    create_attention_mask=True,
):
    input_embeds = preprocess_encoder_inputs(config, input_features, parameters=parameters.encoder, device=device)
    (decoder_hidden_states, attention_mask) = preprocess_decoder_inputs(
        config,
        input_ids,
        attention_mask,
        parameters=parameters.decoder,
        device=device,
        create_attention_mask=create_attention_mask,
    )
    return input_embeds, decoder_hidden_states, attention_mask


def whisper(
    config,
    encoder_hidden_states,
    decoder_hidden_states,
    decoder_attention_mask,
    kv_cache=None,
    current_decode_pos=None,
    *,
    parameters,
):
    encoder_hidden_states = encoder(config, encoder_hidden_states, parameters=parameters.encoder)
    last_hidden_state = decoder(
        config,
        decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        kv_cache=kv_cache,
        current_decode_pos=current_decode_pos,
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
