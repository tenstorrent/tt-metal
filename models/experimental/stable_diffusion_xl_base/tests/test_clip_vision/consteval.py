# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Const-eval functions for CLIP Resampler model."""

import ttnn


def _full_1_16_1280_ones(device):
    return ttnn.full(
        shape=ttnn.Shape([1, 16, 1280]),
        fill_value=1.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _single_weight_reshape_repeat_5120(input, device):
    x = ttnn.to_device(input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.reshape(x, [1, 1, 5120], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.repeat(x, ttnn.Shape([1, 257, 1]))


def _single_weight_reshape_repeat_1280(input, device):
    x = ttnn.to_device(input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.reshape(x, [1, 1, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.repeat(x, ttnn.Shape([1, 257, 1]))


def _single_weight_reshape_repeat_2048(input, device):
    x = ttnn.to_device(input, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.reshape(x, [1, 1, 2048], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.repeat(x, ttnn.Shape([1, 16, 1]))


def _three_weight_reshape_repeat_concat_dim2(input, device):
    """Concatenates Q, K, V biases along dim 2. Input order: [q_proj, k_proj, v_proj]."""
    mem = ttnn.DRAM_MEMORY_CONFIG

    def process(tensor):
        x = ttnn.to_device(tensor, device=device, memory_config=mem)
        x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=mem)
        x = ttnn.reshape(x, [1, 1, 1280], memory_config=mem)
        return ttnn.repeat(x, ttnn.Shape([1, 257, 1]))

    q = process(input[0])
    k = process(input[1])
    v = process(input[2])
    return ttnn.concat([q, k, v], 2, memory_config=mem)


def _three_weight_concat_dim0(input, device):
    """Concatenates Q, K, V weights along dim 0. Input order: [q_proj, k_proj, v_proj]."""
    mem = ttnn.DRAM_MEMORY_CONFIG

    def process(tensor):
        x = ttnn.to_device(tensor, device=device, memory_config=mem)
        return ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=mem)

    q = process(input[0])
    k = process(input[1])
    v = process(input[2])
    return ttnn.concat([q, k, v], 0, memory_config=mem)


def _resampler_attention_query(input, device):
    """Precompute resampler attention query from latents.

    Input: [latents, ln1_bias, ln1_weight, to_q_weight]
    Returns: (ln1_latents_reshaped, precomputed_q)
    """
    mem = ttnn.DRAM_MEMORY_CONFIG

    # Load weights to device
    to_q_weight = ttnn.to_device(input[3], device=device, memory_config=mem)
    to_q_weight = ttnn.to_layout(to_q_weight, ttnn.Layout.TILE, None, memory_config=mem)
    ln1_weight = ttnn.to_device(input[2], device=device, memory_config=mem)
    ln1_weight = ttnn.to_layout(ln1_weight, ttnn.Layout.TILE, None, memory_config=mem)
    ln1_bias = ttnn.to_device(input[1], device=device, memory_config=mem)
    ln1_bias = ttnn.to_layout(ln1_bias, ttnn.Layout.TILE, None, memory_config=mem)

    # Scale factor for attention (1/sqrt(8))
    scale = ttnn.full(
        shape=ttnn.Shape([1, 20, 16, 64]),
        fill_value=0.35355338454246521,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=mem,
    )

    # Layer norm on latents
    ln_out = ttnn.layer_norm(
        input[0],
        epsilon=9.9999997473787516e-06,
        weight=ln1_weight,
        bias=ln1_bias,
        residual_input_tensor=None,
        memory_config=mem,
        program_config=None,
    )

    # Q projection: [16, 1280] @ [1280, 1280]^T -> [16, 1280]
    ln_reshaped = ttnn.reshape(ln_out, [16, 1280], memory_config=mem)
    q = ttnn.matmul(
        ln_reshaped,
        to_q_weight,
        transpose_a=False,
        transpose_b=True,
        memory_config=mem,
        dtype=None,
        program_config=None,
        activation=None,
    )

    # Reshape and permute for multi-head attention
    q = ttnn.reshape(q, [1, 16, 20, 64], memory_config=mem)
    q = ttnn.permute(q, [0, 2, 1, 3], memory_config=mem, pad_value=0.0)

    # Apply scale
    q = ttnn.typecast(q, ttnn.DataType.FLOAT32, memory_config=mem)
    q = ttnn.multiply(q, scale, dtype=ttnn.DataType.FLOAT32, memory_config=mem)
    q = ttnn.typecast(q, ttnn.DataType.BFLOAT16, memory_config=mem)

    # Return reshaped ln output and precomputed q
    ln_reshaped_out = ttnn.reshape(ln_out, [16, 1280], memory_config=mem)
    return ln_reshaped_out, q


def _prepare_conv_weights(input, device):
    return ttnn.prepare_conv_weights(
        weight_tensor=input,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=3,
        out_channels=1280,
        batch_size=1,
        input_height=224,
        input_width=224,
        kernel_size=[14, 14],
        stride=[14, 14],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=False,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )


def _position_embedding_lookup(input, device):
    """Lookup position embeddings. Input: [position_ids, embedding_weight]."""
    mem = ttnn.DRAM_MEMORY_CONFIG

    # Prepare position IDs
    pos_ids = ttnn.to_device(input[0], device=device, memory_config=mem)
    pos_ids = ttnn.to_layout(pos_ids, ttnn.Layout.TILE, None, memory_config=mem)
    pos_ids = ttnn.typecast(pos_ids, ttnn.DataType.UINT32, memory_config=mem)
    pos_ids = ttnn.to_layout(pos_ids, ttnn.Layout.ROW_MAJOR, None, memory_config=mem)

    # Load embedding weight
    embed_weight = ttnn.to_device(input[1], device=device, memory_config=mem)

    # Lookup and permute
    embed = ttnn.embedding(pos_ids, embed_weight, layout=ttnn.Layout.TILE)
    return ttnn.permute(embed, [0, 2, 1], memory_config=mem, pad_value=0.0)


def _reshape_permute_1280(input, device):
    """Reshape to [1, 1, 1280] and permute to [1, 1280, 1]."""
    mem = ttnn.DRAM_MEMORY_CONFIG
    x = ttnn.to_device(input, device=device, memory_config=mem)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=mem)
    x = ttnn.reshape(x, [1, 1, 1280], memory_config=mem)
    return ttnn.permute(x, [0, 2, 1], memory_config=mem, pad_value=0.0)


def run_const_evals(weights, device):
    # fmt: off
    weights["__ONES_1_16_1280__"] = _full_1_16_1280_ones(device)
    weights["__POSITION_EMBEDDING__"] = _position_embedding_lookup([weights["__POSITION_IDS__"], weights["image_encoder.vision_model.embeddings.position_embedding.weight"]], device)
    weights["image_encoder.vision_model.embeddings.class_embedding"] = _reshape_permute_1280(weights["image_encoder.vision_model.embeddings.class_embedding"], device)
    weights["image_encoder.vision_model.embeddings.patch_embedding.weight"] = _prepare_conv_weights(weights["image_encoder.vision_model.embeddings.patch_embedding.weight"], device)
    weights["image_encoder.vision_model.encoder.layers.0.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.0.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.0.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.0.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.0.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.0.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.0.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.0.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.0.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.0.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.0.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.0.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.0.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.0.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.1.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.1.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.1.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.1.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.1.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.1.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.1.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.1.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.1.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.1.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.1.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.1.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.1.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.1.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.2.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.2.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.2.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.2.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.2.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.2.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.2.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.2.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.2.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.2.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.2.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.2.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.2.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.2.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.3.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.3.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.3.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.3.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.3.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.3.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.3.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.3.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.3.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.3.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.3.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.3.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.3.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.3.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.4.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.4.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.4.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.4.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.4.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.4.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.4.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.4.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.4.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.4.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.4.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.4.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.4.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.4.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.5.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.5.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.5.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.5.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.5.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.5.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.5.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.5.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.5.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.5.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.5.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.5.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.5.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.5.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.6.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.6.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.6.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.6.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.6.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.6.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.6.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.6.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.6.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.6.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.6.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.6.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.6.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.6.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.7.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.7.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.7.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.7.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.7.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.7.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.7.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.7.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.7.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.7.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.7.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.7.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.7.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.7.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.8.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.8.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.8.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.8.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.8.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.8.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.8.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.8.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.8.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.8.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.8.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.8.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.8.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.8.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.9.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.9.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.9.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.9.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.9.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.9.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.9.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.9.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.9.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.9.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.9.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.9.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.9.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.9.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.10.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.10.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.10.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.10.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.10.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.10.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.10.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.10.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.10.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.10.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.10.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.10.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.10.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.10.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.11.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.11.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.11.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.11.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.11.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.11.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.11.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.11.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.11.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.11.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.11.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.11.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.11.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.11.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.12.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.12.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.12.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.12.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.12.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.12.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.12.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.12.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.12.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.12.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.12.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.12.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.12.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.12.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.13.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.13.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.13.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.13.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.13.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.13.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.13.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.13.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.13.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.13.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.13.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.13.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.13.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.13.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.14.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.14.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.14.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.14.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.14.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.14.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.14.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.14.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.14.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.14.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.14.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.14.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.14.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.14.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.15.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.15.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.15.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.15.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.15.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.15.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.15.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.15.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.15.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.15.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.15.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.15.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.15.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.15.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.16.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.16.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.16.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.16.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.16.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.16.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.16.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.16.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.16.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.16.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.16.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.16.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.16.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.16.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.17.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.17.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.17.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.17.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.17.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.17.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.17.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.17.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.17.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.17.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.17.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.17.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.17.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.17.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.18.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.18.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.18.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.18.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.18.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.18.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.18.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.18.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.18.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.18.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.18.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.18.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.18.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.18.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.19.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.19.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.19.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.19.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.19.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.19.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.19.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.19.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.19.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.19.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.19.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.19.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.19.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.19.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.20.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.20.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.20.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.20.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.20.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.20.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.20.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.20.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.20.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.20.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.20.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.20.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.20.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.20.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.21.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.21.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.21.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.21.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.21.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.21.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.21.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.21.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.21.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.21.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.21.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.21.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.21.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.21.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.22.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.22.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.22.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.22.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.22.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.22.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.22.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.22.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.22.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.22.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.22.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.22.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.22.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.22.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.23.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.23.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.23.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.23.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.23.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.23.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.23.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.23.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.23.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.23.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.23.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.23.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.23.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.23.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.24.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.24.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.24.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.24.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.24.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.24.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.24.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.24.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.24.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.24.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.24.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.24.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.24.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.24.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.25.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.25.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.25.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.25.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.25.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.25.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.25.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.25.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.25.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.25.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.25.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.25.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.25.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.25.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.26.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.26.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.26.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.26.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.26.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.26.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.26.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.26.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.26.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.26.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.26.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.26.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.26.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.26.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.27.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.27.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.27.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.27.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.27.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.27.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.27.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.27.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.27.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.27.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.27.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.27.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.27.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.27.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.28.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.28.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.28.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.28.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.28.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.28.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.28.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.28.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.28.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.28.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.28.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.28.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.28.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.28.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.29.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.29.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.29.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.29.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.29.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.29.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.29.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.29.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.29.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.29.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.29.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.29.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.29.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.29.mlp.fc2.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.30.self_attn.qkv_weight"] = _three_weight_concat_dim0([weights["image_encoder.vision_model.encoder.layers.30.self_attn.q_proj.weight"], weights["image_encoder.vision_model.encoder.layers.30.self_attn.k_proj.weight"], weights["image_encoder.vision_model.encoder.layers.30.self_attn.v_proj.weight"]], device)
    weights["image_encoder.vision_model.encoder.layers.30.self_attn.qkv_bias"] = _three_weight_reshape_repeat_concat_dim2([weights["image_encoder.vision_model.encoder.layers.30.self_attn.q_proj.bias"], weights["image_encoder.vision_model.encoder.layers.30.self_attn.k_proj.bias"], weights["image_encoder.vision_model.encoder.layers.30.self_attn.v_proj.bias"]], device)
    weights["image_encoder.vision_model.encoder.layers.30.self_attn.out_proj.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.30.self_attn.out_proj.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.30.mlp.fc1.bias"] = _single_weight_reshape_repeat_5120(weights["image_encoder.vision_model.encoder.layers.30.mlp.fc1.bias"], device)
    weights["image_encoder.vision_model.encoder.layers.30.mlp.fc2.bias"] = _single_weight_reshape_repeat_1280(weights["image_encoder.vision_model.encoder.layers.30.mlp.fc2.bias"], device)
    weights["resampler.proj_in.bias"] = _single_weight_reshape_repeat_1280(weights["resampler.proj_in.bias"], device)
    weights["resampler.proj_out.bias"] = _single_weight_reshape_repeat_2048(weights["resampler.proj_out.bias"], device)
    _tmp_30 = _resampler_attention_query([weights["resampler.latents"], weights["resampler.layers.0.ln1.bias"], weights["resampler.layers.0.ln1.weight"], weights["resampler.layers.0.attn.to_q.weight"]], device)
    weights["resampler.layers.0.ln1_latents_reshaped"] = _tmp_30[0]
    weights["resampler.layers.0.attn.precomputed_q"] = _tmp_30[1]
    # fmt: on

    return weights
