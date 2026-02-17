# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Optional

import torch
import transformers
from loguru import logger
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn
from models.common.utility_functions import nearest_32

WHISPER_MEMORY_CONFIG = ttnn.DRAM_MEMORY_CONFIG

WHISPER_BATCH_SIZE = 2
WHISPER_L1_SMALL_SIZE = 1024
WHISPER_TRACE_REGION_SIZE = 100000000


def gelu(tensor):
    return ttnn.gelu(tensor, memory_config=WHISPER_MEMORY_CONFIG)


def dropout(hidden_states, p, training):
    # ignored for inference
    return hidden_states


def unsqueeze_to_4D_at_dim_1(tensor):
    rank = len(tensor.shape)
    if rank == 4:
        return tensor
    elif rank == 3:
        return ttnn.unsqueeze(tensor, 1)
    elif rank == 2:
        return ttnn.unsqueeze(ttnn.unsqueeze(tensor, 1), 1)
    else:
        raise ValueError(f"Unsupported shape: {tensor.shape}")


def init_kv_cache(config, device, max_seq_len, weights_mesh_mapper, n_layers=None):
    """
    Generates empty KV cache for self-attention and cross-attention, and sends to device.
    Returns:
        tuple: (kv_cache, cross_attn_cache)
            - kv_cache: List of [K, V] tensors per layer for self-attention
            - cross_attn_cache: List of [K, V] tensors per layer for cross-attention (pre-allocated)
    """

    logger.info(f"Initializing KV cache for both batch size per device 1 and 2 and max sequence length: {max_seq_len}")

    kv_cache_per_batch_size = defaultdict(lambda: None)
    cross_attn_cache_per_batch_size = defaultdict(lambda: None)
    for batch_size in [1, WHISPER_BATCH_SIZE]:
        kv_cache = []
        cross_attn_cache = []
        if n_layers is None:
            n_layers = config.decoder_layers

        # Cross-attention cache dimensions
        # encoder_seq_len = 1500 for Whisper (30s max audio / 20ms per frame)
        encoder_seq_len = 1500
        num_heads = config.encoder_attention_heads
        head_dim = config.d_model // config.encoder_attention_heads

        for i in range(n_layers):
            # Self-attention cache
            kv_cache_layer = []
            for j in range(2):
                cache_k_or_v = torch.zeros(
                    (
                        batch_size,
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
                    mesh_mapper=weights_mesh_mapper,
                    cache_file_name=None,
                )
                kv_cache_layer.append(cache_k_or_v)
            kv_cache.append(kv_cache_layer)

            # Pre-allocate cross-attention cache for tracing
            # bfloat16 to match calculate_key_values output dtype
            cross_k = torch.zeros((batch_size, num_heads, head_dim, encoder_seq_len))
            cross_k = ttnn.as_tensor(
                cross_k,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=weights_mesh_mapper,
                cache_file_name=None,
            )

            cross_v = torch.zeros((batch_size, num_heads, encoder_seq_len, head_dim))
            cross_v = ttnn.as_tensor(
                cross_v,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=weights_mesh_mapper,
                cache_file_name=None,
            )
            cross_attn_cache.append([cross_k, cross_v])

        kv_cache_per_batch_size[batch_size] = kv_cache
        cross_attn_cache_per_batch_size[batch_size] = cross_attn_cache

    return kv_cache_per_batch_size, cross_attn_cache_per_batch_size


def calculate_key_values(config, key_value_states, *, parameters):
    hidden_size = key_value_states.shape[-1]
    head_size = hidden_size // config.encoder_attention_heads

    compute_grid_size = key_value_states.device().compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=compute_grid_size.y, x=compute_grid_size.x)
    key_value_states = ttnn.to_memory_config(key_value_states, ttnn.L1_MEMORY_CONFIG)
    fused_kv = ttnn.linear(
        key_value_states,
        parameters.key_value.weight,
        bias=parameters.key_value.bias,
        core_grid=core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn.deallocate(key_value_states)
    fused_kv = ttnn.to_memory_config(fused_kv, WHISPER_MEMORY_CONFIG)
    fused_kv = unsqueeze_to_4D_at_dim_1(fused_kv)  # B, 1, S, 2xHxd
    key_states = fused_kv[:, :, :, :hidden_size]
    value_states = fused_kv[:, :, :, hidden_size:]

    # num_kv_heads=0 is required here because we're calling nlp_create_qkv_heads
    # on just the K or V tensor (not combined QKV)
    key_states = ttnn.experimental.nlp_create_qkv_heads(
        key_states,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        num_heads=config.encoder_attention_heads,
        num_kv_heads=0,
    )[0]
    key_states = ttnn.permute(key_states, [0, 1, 3, 2])

    value_states = ttnn.experimental.nlp_create_qkv_heads(
        value_states,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        num_heads=config.encoder_attention_heads,
        num_kv_heads=0,
    )[0]

    return key_states, value_states


def get_decode_sdpa_configs(config, bsz, device):
    head_size = config.d_model // config.decoder_attention_heads
    padded_num_heads = nearest_32(config.decoder_attention_heads)

    # Q, K, V are batch sharded across cores
    grid_size = device.compute_with_storage_grid_size()
    batch_grid = ttnn.num_cores_to_corerangeset(bsz, grid_size, row_wise=True)

    sdpa_batch_sharded_memcfg = ttnn.create_sharded_memory_config(
        shape=(padded_num_heads, head_size),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    compute_grid_size = device.compute_with_storage_grid_size()
    sdpa_decode_progcfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(compute_grid_size.x, compute_grid_size.y),
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


def functional_sdpa(
    query_states, key_states, value_states, scaling, attention_mask, is_cross_attention=False, is_decode=False
):
    if is_cross_attention:
        head_num = query_states.shape[1]
        if head_num == 20:  # openai/whisper-large-v3 & distil-whisper/distil-large-v3 models
            core_grid = ttnn.CoreGrid(y=4, x=5)
        elif head_num == 16:  # openai/whisper-medium
            core_grid = ttnn.CoreGrid(y=4, x=4)
        elif head_num == 12:  # openai/whisper-small
            core_grid = ttnn.CoreGrid(y=3, x=4)
        elif head_num == 8:  # openai/whisper-base
            core_grid = ttnn.CoreGrid(y=2, x=4)
        elif head_num == 6:  # openai/whisper-tiny
            core_grid = ttnn.CoreGrid(y=2, x=3)
        else:
            raise ValueError(f"Unsupported head number: {head_num}")

        height_sharded_config_query_states = ttnn.create_sharded_memory_config(
            query_states.padded_shape,
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        query_states = ttnn.to_memory_config(query_states, height_sharded_config_query_states)

        height_sharded_config_key_states = ttnn.create_sharded_memory_config(
            key_states.padded_shape,
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        key_states = ttnn.to_memory_config(key_states, height_sharded_config_key_states)

        height_sharded_config_value_states = ttnn.create_sharded_memory_config(
            value_states.padded_shape,
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        value_states = ttnn.to_memory_config(value_states, height_sharded_config_value_states)

        query_states *= scaling

        attn_weights = ttnn.matmul(query_states, key_states, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
        ttnn.deallocate(query_states)
        ttnn.deallocate(key_states)

        attn_weights = ttnn.softmax_in_place(attn_weights, dim=-1)

        attn_probs = dropout(attn_weights, p=0, training=False)

        attn_output = ttnn.matmul(
            attn_probs,
            value_states,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_probs)
        ttnn.deallocate(value_states)
        attn_output = ttnn.to_memory_config(attn_output, WHISPER_MEMORY_CONFIG)

    elif (not is_decode) and (query_states.shape[-2] == value_states.shape[-2]):
        ## Encoder-self-attention
        q_chunk_size = 256
        k_chunk_size = 256
        compute_grid_size = query_states.device().compute_with_storage_grid_size()
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(compute_grid_size.x, compute_grid_size.y),
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=True,  # NOTE: False is more correct
        )

        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            query_states.device().arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=scaling,
            attn_mask=None,
            is_causal=False,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    else:
        # original implementation for decoder-attention-no-KV-cache
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
    cross_attn_cache=None,
    cross_attn_cache_valid=False,
    current_decode_pos=None,
    *,
    parameters,
):
    hidden_states = unsqueeze_to_4D_at_dim_1(hidden_states)

    head_size = config.d_model // config.encoder_attention_heads
    scaling = head_size**-0.5
    bsz, *_, tgt_len, _ = hidden_states.shape

    is_cross_attention = encoder_hidden_states is not None
    sdpa_with_kv_cache = not is_cross_attention and is_decode and kv_cache is not None
    # Enabling encoder SDPA, to disable the K-transpose in ttnn.experimental.nlp_create_qkv_heads
    encoder_sdpa_attention = not is_decode
    if encoder_sdpa_attention:
        transpose_k_heads = False
    elif sdpa_with_kv_cache:
        transpose_k_heads = False
    else:
        transpose_k_heads = True

    if is_cross_attention:
        query_states = hidden_states @ parameters.q_proj.weight + parameters.q_proj.bias
        query_states = ttnn.transpose(query_states, 1, 2)  # B, S, 1, Hxd
        query_states = ttnn.reshape(
            query_states, (bsz, tgt_len, config.encoder_attention_heads, head_size)
        )  # B, S, H, d
        query_states = ttnn.transpose(query_states, 1, 2)  # B, H, S, d
        # Use cached cross-attention K/V if cache is valid, otherwise compute and copy to pre-allocated cache
        if cross_attn_cache is not None and cross_attn_cache_valid:
            key_states, value_states = cross_attn_cache[0], cross_attn_cache[1]
        else:
            # Compute K/V and copy to pre-allocated cache
            key_states, value_states = calculate_key_values(config, encoder_hidden_states, parameters=parameters)
            if cross_attn_cache is not None:
                # Copy to pre-allocated tensors
                ttnn.copy(key_states, cross_attn_cache[0])
                ttnn.copy(value_states, cross_attn_cache[1])
                # Use cache references for attention
                key_states, value_states = cross_attn_cache[0], cross_attn_cache[1]
        attn_output = functional_sdpa(
            query_states,
            key_states,
            value_states,
            scaling,
            attention_mask,
            is_cross_attention=is_cross_attention,
            is_decode=is_decode,
        )
    else:
        if not is_decode:
            fused_qkv_dtype = ttnn.bfloat8_b
        else:
            fused_qkv_dtype = ttnn.bfloat16
        compute_grid_size = hidden_states.device().compute_with_storage_grid_size()
        core_grid = ttnn.CoreGrid(y=compute_grid_size.y, x=compute_grid_size.x)
        fused_qkv = ttnn.linear(
            hidden_states,
            parameters.query_key_value.weight,
            bias=parameters.query_key_value.bias,
            core_grid=core_grid,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=fused_qkv_dtype,
        )
        ttnn.deallocate(hidden_states)
        fused_qkv = ttnn.to_memory_config(fused_qkv, WHISPER_MEMORY_CONFIG)

        fused_qkv = unsqueeze_to_4D_at_dim_1(fused_qkv)
        (
            query_states,
            key_states,
            value_states,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            fused_qkv,
            num_heads=config.decoder_attention_heads,
            num_kv_heads=config.decoder_attention_heads,
            transpose_k_heads=transpose_k_heads,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(fused_qkv)
        if sdpa_with_kv_cache:
            k_cache = kv_cache[0]  # B, H, MaxS, d
            v_cache = kv_cache[1]  # B, H, MaxS, d

            sdpa_batch_sharded_memcfg, sdpa_decode_progcfg, sdpa_decode_compute_kernel_config = get_decode_sdpa_configs(
                config, bsz, hidden_states.device()
            )

            # Transpose from [B, H, S, d] to [S, B, H, d] for SDPA decode operations
            query_states = ttnn.transpose(query_states, 0, 2)  # [B, H, S, d] -> [S, H, B, d]
            query_states = ttnn.transpose(query_states, 1, 2)  # [S, H, B, d] -> [S, B, H, d]
            key_states = ttnn.transpose(key_states, 0, 2)
            key_states = ttnn.transpose(key_states, 1, 2)
            value_states = ttnn.transpose(value_states, 0, 2)
            value_states = ttnn.transpose(value_states, 1, 2)

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
            )  # [1, B, H, d]

            # Transpose back to [B, H, S, d] format for nlp_concat_heads
            attn_output = ttnn.transpose(attn_output, 1, 2)  # [1, B, H, d] -> [1, H, B, d]
            attn_output = ttnn.transpose(attn_output, 0, 2)  # [1, H, B, d] -> [B, H, 1, d]
        else:
            attn_output = functional_sdpa(
                query_states,
                key_states,
                value_states,
                scaling,
                attention_mask,
                is_cross_attention=is_cross_attention,
                is_decode=is_decode,
            )
            ttnn.deallocate(query_states)
            ttnn.deallocate(key_states)
            ttnn.deallocate(value_states)

    attn_output = ttnn.experimental.nlp_concat_heads(attn_output)
    attn_output = attn_output @ parameters.out_proj.weight + parameters.out_proj.bias
    return attn_output


def encoder_layer(config, hidden_states, *, parameters):
    hidden_states = unsqueeze_to_4D_at_dim_1(hidden_states)

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
    config,
    hidden_states,
    attention_mask,
    encoder_hidden_states,
    kv_cache=None,
    current_decode_pos=None,
    cross_attn_cache=None,
    cross_attn_cache_valid=False,
    *,
    parameters,
):
    hidden_states = unsqueeze_to_4D_at_dim_1(hidden_states)
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
        cross_attn_cache=cross_attn_cache,
        cross_attn_cache_valid=cross_attn_cache_valid,
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
    cross_attn_cache=None,
    cross_attn_cache_valid=False,
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
            cross_attn_cache=cross_attn_cache[i] if cross_attn_cache is not None else None,
            cross_attn_cache_valid=cross_attn_cache_valid,
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


def prepare_conv_weights(config, parameters, weights_mesh_mapper):
    conv2_out_channel_splits = 4
    conv2_out_channels = config.d_model // conv2_out_channel_splits
    if isinstance(parameters.conv1.weight, torch.Tensor):
        parameters.conv1.weight = ttnn.from_torch(
            parameters.conv1.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=weights_mesh_mapper
        )
    if isinstance(parameters.conv2.weight, torch.Tensor):
        parameters.conv2.weight = ttnn.from_torch(
            parameters.conv2.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=weights_mesh_mapper
        )
        # Split output channels to avoid running out of L1 memory
        weight_splits = []
        for i in range(conv2_out_channel_splits):
            weight_splits.append(parameters.conv2.weight[i * conv2_out_channels : (i + 1) * conv2_out_channels, :, :])
        parameters.conv2.weight = weight_splits
    return conv2_out_channel_splits, conv2_out_channels


def preprocess_encoder_inputs(config, input_features, *, parameters, device, input_mesh_mapper, weights_mesh_mapper):
    input_length = input_features.shape[-1]

    input_features = ttnn.from_torch(
        input_features, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=input_mesh_mapper, device=device
    )
    input_features = ttnn.transpose(input_features, -1, -2)
    batch_size = input_features.shape[0]

    conv1_config, conv1_compute_config = get_conv_configs(device)

    # First time convs are runs, weights are on host (convs will return weights on device)
    conv2_out_channel_splits, conv2_out_channels = prepare_conv_weights(
        config, parameters, weights_mesh_mapper=weights_mesh_mapper
    )

    # TODO: This is a hack to support batch size > 1, till the Conv OOM issue is resolved
    input_embeds_all_batches_splits = []
    for j in range(batch_size):
        input_features_slice = input_features[j : j + 1, :, :, :]
        input_embeds, [weights_device, _] = ttnn.conv1d(
            input_tensor=input_features_slice,
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
            dtype=ttnn.bfloat16,
            conv_config=conv1_config,
            compute_config=conv1_compute_config,
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
                dtype=ttnn.bfloat16,
                compute_config=conv1_compute_config,
                return_weights_and_bias=True,
            )
            parameters.conv2.weight[i] = weights_device
            out_split = ttnn.sharded_to_interleaved(out_split)
            out_tensor_splits.append(out_split)

        input_embeds = ttnn.concat(out_tensor_splits, dim=3)
        input_embeds = ttnn.gelu(input_embeds)
        input_embeds = ttnn.squeeze(input_embeds, 0)
        input_embeds_all_batches_splits.append(input_embeds)

    input_embeds_all_batches = ttnn.concat(input_embeds_all_batches_splits, dim=0)
    return input_embeds_all_batches


def preprocess_decoder_inputs(
    config,
    input_ids,
    attention_mask,
    *,
    parameters,
    device,
    input_mesh_mapper,
    decode_pos=None,
    create_attention_mask=True,
):
    input_shape = input_ids.size()
    input_ids = torch.reshape(input_ids, (-1, input_shape[-1]))
    tt_input_ids = ttnn.from_torch(
        input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=input_mesh_mapper
    )
    inputs_embeds = ttnn.embedding(
        tt_input_ids, parameters.embed_tokens.weight, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    if create_attention_mask:
        attention_mask = prepare_decoder_attention_mask(attention_mask, input_shape)
    if attention_mask is not None:
        # ttnn cannot broadcast when adding on the batch or channel dimensions so this is a workaround
        attention_mask = attention_mask.expand(-1, config.decoder_attention_heads, -1, -1)
        attention_mask = ttnn.from_torch(
            attention_mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=input_mesh_mapper
        )

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
    input_mesh_mapper,
    weights_mesh_mapper,
):
    input_embeds = preprocess_encoder_inputs(
        config,
        input_features,
        parameters=parameters.encoder,
        device=device,
        input_mesh_mapper=input_mesh_mapper,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    (decoder_hidden_states, attention_mask) = preprocess_decoder_inputs(
        config,
        input_ids,
        attention_mask,
        parameters=parameters.decoder,
        device=device,
        create_attention_mask=create_attention_mask,
        input_mesh_mapper=input_mesh_mapper,
    )
    return input_embeds, decoder_hidden_states, attention_mask


def whisper(
    config,
    encoder_hidden_states,
    decoder_hidden_states,
    decoder_attention_mask,
    kv_cache=None,
    cross_attn_cache=None,
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
        cross_attn_cache=cross_attn_cache,
        current_decode_pos=current_decode_pos,
        parameters=parameters.decoder,
    )
    return last_hidden_state


def create_custom_mesh_preprocessor(weights_mesh_mapper):
    def custom_mesh_preprocessor(model, name):
        return custom_preprocessor(model, name, weights_mesh_mapper)

    return custom_mesh_preprocessor


def custom_preprocessor(torch_model, name, weights_mesh_mapper):
    parameters = {}
    if isinstance(torch_model, transformers.models.whisper.modeling_whisper.WhisperAttention):
        height, width = torch_model.k_proj.weight.shape

        if "encoder_attn" in name:
            parameters = {"key_value": {}, "q_proj": {}, "out_proj": {}}
            preprocessed_weight = torch.cat([torch_model.k_proj.weight, torch_model.v_proj.weight], dim=0)
            preprocessed_bias = torch.cat([torch.zeros(height), torch_model.v_proj.bias], dim=0)
            parameters["key_value"]["weight"] = preprocess_linear_weight(
                preprocessed_weight, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
            )
            parameters["key_value"]["bias"] = preprocess_linear_bias(
                preprocessed_bias, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
            )
            parameters["q_proj"]["weight"] = preprocess_linear_weight(
                torch_model.q_proj.weight, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
            )
            parameters["q_proj"]["bias"] = preprocess_linear_bias(
                torch_model.q_proj.bias, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
            )
        else:
            parameters = {"query_key_value": {}, "out_proj": {}}
            preprocessed_weight = torch.cat(
                [torch_model.q_proj.weight, torch_model.k_proj.weight, torch_model.v_proj.weight], dim=0
            )
            preprocessed_bias = torch.cat(
                [torch_model.q_proj.bias, torch.zeros(height), torch_model.v_proj.bias], dim=0
            )
            parameters["query_key_value"]["weight"] = preprocess_linear_weight(
                preprocessed_weight, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
            )
            parameters["query_key_value"]["bias"] = preprocess_linear_bias(
                preprocessed_bias, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
            )

        parameters["out_proj"]["weight"] = preprocess_linear_weight(
            torch_model.out_proj.weight, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
        )
        parameters["out_proj"]["bias"] = preprocess_linear_bias(
            torch_model.out_proj.bias, dtype=ttnn.bfloat16, weights_mesh_mapper=weights_mesh_mapper
        )
    elif name == "encoder.embed_positions" and isinstance(torch_model, torch.nn.Embedding):
        embeddings = ttnn.from_torch(torch_model.weight, dtype=ttnn.bfloat16, mesh_mapper=weights_mesh_mapper)
        embeddings = ttnn.to_layout(embeddings, ttnn.TILE_LAYOUT)
        parameters["weight"] = embeddings
    return parameters
