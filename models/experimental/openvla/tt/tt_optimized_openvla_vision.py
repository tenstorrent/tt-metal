# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Optimized DINOv2 and SigLIP encoders for OpenVLA.
Copied from models/demos/vit/tt/ttnn_optimized_vit_highres_gs.py for self-contained openvla module.
"""

import numpy as np
import torch
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn

core_grid = ttnn.CoreGrid(y=8, x=12)


def vit_patch_embeddings_weight_vars(
    config,
    pixel_values,
    proj_weight,
    proj_bias,
    patch_size=16,
):
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_count = img_h // patch_size
    patch_count_all = int(patch_count * patch_count)
    stride_h = patch_size
    stride_w = 1

    pixel_values = ttnn.reshape(pixel_values, (batch_size, img_h, img_w // patch_size, 4 * patch_size))
    pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    patch_embedding_output = ttnn.linear(
        pixel_values,
        proj_weight,
        bias=proj_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    ttnn.deallocate(pixel_values)

    patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embedding_output = ttnn.reshape(patch_embedding_output, (batch_size, patch_count_all, -1))

    return patch_embedding_output


def siglip_patch_embeddings(
    pixel_values,
    *,
    parameters,
):
    batch_size, img_h, img_w, img_c = pixel_values.shape  # permuted input NHWC
    patch_size = 14
    patch_count = img_h // patch_size  # 16
    patch_count_all = int(patch_count * patch_count)  # 256
    stride_h = patch_size
    stride_w = 1

    pixel_values = ttnn.reshape(pixel_values, (batch_size, img_h, img_w // patch_size, 4 * patch_size))
    pixel_values = ttnn.fold(pixel_values, stride_h, stride_w)
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    patch_embedding_output = ttnn.linear(
        pixel_values,
        parameters.projection.weight,
        bias=parameters.projection.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    ttnn.deallocate(pixel_values)

    patch_embedding_output = ttnn.to_layout(patch_embedding_output, layout=ttnn.ROW_MAJOR_LAYOUT)
    patch_embedding_output = ttnn.reshape(patch_embedding_output, (batch_size, patch_count_all, -1))

    return patch_embedding_output


def siglip_attention(
    hidden_states,
    attention_mask,
    parameters,
):
    num_heads = 16
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads

    query_key_value = ttnn.linear(
        hidden_states,
        parameters.query_key_value.weight,
        bias=parameters.query_key_value.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    ttnn.reallocate(hidden_states)
    (
        query,
        key,
        value,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value)

    # Scale query by 1/sqrt(head_size) - same as DINOv2 approach
    scale = 1.0 / (head_size**0.5)
    query = ttnn.mul_(query, scale)
    value = ttnn.reallocate(value)

    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    # Use softmax_in_place instead of attention_softmax_ (doesn't require mask)
    attention_probs = ttnn.softmax_in_place(attention_scores, numeric_stable=True)

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)

    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    self_output = ttnn.linear(
        context_layer,
        parameters.proj.weight,
        bias=parameters.proj.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    ttnn.deallocate(context_layer)

    return self_output


def siglip_intermediate(
    hidden_states,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
        activation="gelu",
    )
    return output


def siglip_output(
    hidden_states,
    residual,
    *,
    parameters,
):
    output = ttnn.linear(
        hidden_states,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    ttnn.deallocate(hidden_states)
    output = ttnn.add(output, residual, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    return output


def siglip_feedforward(
    hidden_states,
    attention_output,
    *,
    parameters,
):
    intermediate = siglip_intermediate(hidden_states, parameters=parameters.mlp)
    hidden_states = siglip_output(intermediate, attention_output, parameters=parameters.mlp)
    return hidden_states


def siglip_layer(
    hidden_states,
    attention_mask,
    parameters,
):
    layernorm_before_output = ttnn.layer_norm(
        hidden_states,
        weight=parameters.norm1.weight,
        bias=parameters.norm1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    multi_head_attention_output = siglip_attention(
        layernorm_before_output,
        attention_mask=attention_mask,
        parameters=parameters.attn,
    )

    multi_head_attention_output = ttnn.add(
        multi_head_attention_output, hidden_states, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )

    layernorm_after_output = ttnn.layer_norm(
        multi_head_attention_output,
        weight=parameters.norm2.weight,
        bias=parameters.norm2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    feedforward_output = siglip_feedforward(
        layernorm_after_output,
        multi_head_attention_output,
        parameters=parameters,
    )

    return feedforward_output


def siglip_encoder(
    embeddings,
    head_masks,
    parameters,
    layer_end_index=None,
):
    encoder_input = embeddings
    if layer_end_index is None:
        layer_end_index = len(parameters)
    params = [parameters[index] for index in parameters]
    encoder_output = None
    for index, param in enumerate(params[:layer_end_index]):
        encoder_output = siglip_layer(
            encoder_input,
            head_masks[index],
            param,
        )
        encoder_input = encoder_output

    return encoder_output


def upchannel_attn_weight_bias(qkv_weight, qkv_bias, proj_weight, proj_bias, num_heads):
    qkv = 3
    is_padding_required = (qkv_weight.shape[0] // (num_heads * qkv)) % 32 != 0
    if is_padding_required:
        padded_val = int(np.ceil(qkv_weight.shape[0] / (num_heads * qkv * 32)) * (num_heads * qkv * 32))
        new_qkv_weight = torch.zeros((padded_val, qkv_weight.shape[1]), dtype=qkv_weight.dtype).reshape(
            qkv, num_heads, -1, qkv_weight.shape[1]
        )
        reshaped_qkv_weight = qkv_weight.reshape(qkv, num_heads, -1, qkv_weight.shape[1])
        new_qkv_weight[:, :, : reshaped_qkv_weight.shape[2], :] = reshaped_qkv_weight
        new_qkv_weight = new_qkv_weight.reshape(padded_val, qkv_weight.shape[1])
        new_qkv_bias = torch.zeros((padded_val), dtype=qkv_bias.dtype).reshape(qkv, num_heads, -1)
        reshaped_qkv_bias = qkv_bias.reshape(qkv, num_heads, -1)
        new_qkv_bias[:, :, : reshaped_qkv_bias.shape[2]] = reshaped_qkv_bias
        new_qkv_bias = new_qkv_bias.reshape((-1,))
        new_proj_weight = torch.zeros((proj_weight.shape[0], padded_val // qkv), dtype=proj_weight.dtype).reshape(
            proj_weight.shape[0], num_heads, -1
        )
        reshaped_proj_head = proj_weight.reshape(proj_weight.shape[0], num_heads, -1)
        new_proj_weight[:, :, : reshaped_proj_head.shape[2]] = reshaped_proj_head
        new_proj_weight = new_proj_weight.reshape((proj_weight.shape[0], padded_val // qkv))
        qkv_weight, qkv_bias, proj_weight = new_qkv_weight, new_qkv_bias, new_proj_weight
    return qkv_weight, qkv_bias, proj_weight, proj_bias


def prepare_dinov2_embedding_constants(tensors, device):
    assert len(tensors) == 2
    proj_weight = tensors[0]
    proj_bias = tensors[1]
    three_times_hidden_size, c, _, _ = proj_weight.shape
    pad_value = 4 - c
    preprocessed_weight = torch.nn.functional.pad(proj_weight, (0, 0, 0, 0, 0, pad_value))
    preprocessed_weight = torch.permute(preprocessed_weight, (2, 3, 1, 0))
    preprocessed_weight = torch.reshape(preprocessed_weight, (-1, three_times_hidden_size))

    tensors[0] = ttnn.from_torch(preprocessed_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[1] = ttnn.from_torch(proj_bias.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return [tensors[0], tensors[1]]


def dinov2_embedding(var0, *args):
    var2 = vit_patch_embeddings_weight_vars(None, var0, args[0], args[1], patch_size=14)
    var5 = ttnn.add(var2, args[2])
    var6 = ttnn.concat([args[3], args[4], var5], dim=1)
    return var6


def prepare_dinov2_attention_constants(tensors, device):
    assert len(tensors) == 7
    tensors[0] = ttnn.from_torch(tensors[0].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[1] = ttnn.from_torch(tensors[1].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[2] = ttnn.from_torch(tensors[2].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[3] = ttnn.from_torch(tensors[3].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[4] = ttnn.from_torch(tensors[4].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[5] = ttnn.from_torch(tensors[5].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[6] = ttnn.from_torch(tensors[6].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tensors


def dinov2_attention(var0, *args, num_heads=16):
    hidden_states = ttnn.layer_norm(
        var0,
        weight=args[0],
        bias=args[1],
        epsilon=1e-06,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    *_, hidden_size = hidden_states.shape
    head_size = hidden_size // num_heads
    query_key_value = ttnn.linear(
        hidden_states,
        args[3],
        bias=args[2],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    ttnn.reallocate(hidden_states)
    (
        query,
        key,
        value,
    ) = ttnn.transformer.split_query_key_value_and_split_heads(
        query_key_value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=num_heads,
    )
    ttnn.deallocate(query_key_value)
    scale = 1.0 / (head_size**0.5)
    query = ttnn.mul_(
        query,
        scale,
    )
    value = ttnn.reallocate(value)
    attention_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )

    ttnn.deallocate(query)
    ttnn.deallocate(key)
    attention_probs = ttnn.softmax_in_place(attention_scores, numeric_stable=True)

    context_layer = ttnn.matmul(
        attention_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    ttnn.deallocate(attention_probs)
    ttnn.deallocate(value)
    context_layer = ttnn.transformer.concatenate_heads(
        context_layer,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    self_output = ttnn.linear(
        context_layer,
        args[5],
        bias=args[4],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=8, x=12),
    )
    ttnn.deallocate(context_layer)
    var19 = ttnn.mul(self_output, args[6])
    var20 = ttnn.add(var0, var19)
    return var20


def prepare_dinov2_feedforward_constants(tensors, device):
    assert len(tensors) == 7
    tensors[0] = ttnn.from_torch(tensors[0].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[1] = ttnn.from_torch(tensors[1].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[2] = ttnn.from_torch(tensors[2].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[3] = ttnn.from_torch(tensors[3].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[4] = ttnn.from_torch(tensors[4].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[5] = ttnn.from_torch(tensors[5].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[6] = ttnn.from_torch(tensors[6].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tensors


def dinov2_feedforward(var0, *args):
    hidden_states = ttnn.layer_norm(
        var0,
        weight=args[0],
        bias=args[1],
        epsilon=1e-06,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    hidden_states = ttnn.linear(
        hidden_states,
        args[3],
        bias=args[2],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
        activation="gelu",
    )
    hidden_states = ttnn.linear(
        hidden_states,
        args[5],
        bias=args[4],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=ttnn.CoreGrid(y=10, x=12),
    )
    hidden_states = ttnn.mul(hidden_states, args[6])
    var11 = ttnn.add(var0, hidden_states)
    return var11


def prepare_dinov2_head_constants(tensors, device):
    assert len(tensors) == 2
    tensors[0] = ttnn.from_torch(tensors[0].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensors[1] = ttnn.from_torch(tensors[1].unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    return tensors


def dinov2_head(var0, *args):
    var1 = ttnn.layer_norm(
        var0,
        weight=args[0],
        bias=args[1],
        epsilon=1e-06,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return var1[:, 0, :]


def get_dinov2_params(torch_model):
    state_dict = torch_model.state_dict()
    return {
        "embeddings": [
            state_dict["patch_embed.proj.weight"],
            state_dict["patch_embed.proj.bias"],
            state_dict["pos_embed"],
            state_dict["cls_token"],
            state_dict["reg_token"],
        ],
        "encoder": {
            f"layer{i}": {
                "attention": [
                    state_dict[f"blocks.{i}.norm1.weight"],
                    state_dict[f"blocks.{i}.norm1.bias"],
                    state_dict[f"blocks.{i}.attn.qkv.bias"],
                    state_dict[f"blocks.{i}.attn.qkv.weight"].T,
                    state_dict[f"blocks.{i}.attn.proj.bias"],
                    state_dict[f"blocks.{i}.attn.proj.weight"].T,
                    state_dict[f"blocks.{i}.ls1.scale_factor"],
                ],
                "feed_forward": [
                    state_dict[f"blocks.{i}.norm2.weight"],
                    state_dict[f"blocks.{i}.norm2.bias"],
                    state_dict[f"blocks.{i}.mlp.fc1.bias"],
                    state_dict[f"blocks.{i}.mlp.fc1.weight"].T,
                    state_dict[f"blocks.{i}.mlp.fc2.bias"],
                    state_dict[f"blocks.{i}.mlp.fc2.weight"].T,
                    state_dict[f"blocks.{i}.ls2.scale_factor"],
                ],
            }
            for i in range(len(torch_model.blocks))
        },
    }


def dinov2_encoder(torch_model, ttnn_device, num_output_layers=None):
    """
    Create DINOv2 encoder that returns output from SECOND-TO-LAST layer (no final norm).
    This matches HuggingFace OpenVLA's get_intermediate_layers behavior.

    Args:
        torch_model: PyTorch DINOv2 model
        ttnn_device: TTNN device
        num_output_layers: Number of layers to process (default: total_layers - 1 to get 2nd-to-last)
    """
    parameters = get_dinov2_params(torch_model)
    total_layers = len(torch_model.blocks)

    if num_output_layers is None:
        num_output_layers = total_layers - 1

    embedding_params = prepare_dinov2_embedding_constants(parameters["embeddings"][:2], ttnn_device)
    parameters["embeddings"] = [
        ttnn.from_torch(t, dtype=ttnn.bfloat16, device=ttnn_device) if isinstance(t, torch.Tensor) else t
        for t in embedding_params + parameters["embeddings"][2:]
    ]

    state_dict = torch_model.state_dict()
    final_norm_weight = ttnn.from_torch(
        state_dict["norm.weight"].unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
    )
    final_norm_bias = ttnn.from_torch(
        state_dict["norm.bias"].unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
    )

    def get_layer_num(key):
        if isinstance(key, int):
            return key
        if isinstance(key, str) and key.startswith("layer"):
            return int(key.replace("layer", ""))
        return int(key)

    encoder_layers = sorted(parameters["encoder"].keys(), key=get_layer_num)

    for idx, layer in enumerate(encoder_layers):
        if idx >= num_output_layers:
            continue
        attention_params = prepare_dinov2_attention_constants(
            parameters["encoder"][layer]["attention"][:7], ttnn_device
        )
        parameters["encoder"][layer]["attention"] = [
            ttnn.from_torch(t, dtype=ttnn.bfloat16, device=ttnn_device) if isinstance(t, torch.Tensor) else t
            for t in attention_params + parameters["encoder"][layer]["attention"][7:]
        ]
        feedforward_params = prepare_dinov2_feedforward_constants(
            parameters["encoder"][layer]["feed_forward"][:7], ttnn_device
        )
        parameters["encoder"][layer]["feed_forward"] = [
            ttnn.from_torch(t, dtype=ttnn.bfloat16, device=ttnn_device) if isinstance(t, torch.Tensor) else t
            for t in feedforward_params + parameters["encoder"][layer]["feed_forward"][7:]
        ]

    def model_forward(pixel_values):
        embeddings_output = dinov2_embedding(pixel_values, *parameters["embeddings"])
        embeddings_output = ttnn.to_layout(embeddings_output, layout=ttnn.TILE_LAYOUT)
        for idx, layer in enumerate(encoder_layers):
            if idx >= num_output_layers:
                break
            embeddings_output = dinov2_attention(embeddings_output, *parameters["encoder"][layer]["attention"])
            embeddings_output = dinov2_feedforward(embeddings_output, *parameters["encoder"][layer]["feed_forward"])
        return embeddings_output

    return model_forward


def custom_preprocessor_siglip(torch_model, name):
    import timm

    attention_class = None
    try:
        attention_class = timm.layers.attention.Attention
    except:
        try:
            attention_class = timm.models.vision_transformer.Attention
        except:
            attention_class = None
    assert (
        attention_class is not None
    ), f"Could not find Attention Class in timm library. Please check version timm={timm.__version__}."
    parameters = {}
    if isinstance(torch_model, timm.layers.patch_embed.PatchEmbed):
        weight = torch_model.proj.weight
        bias = torch_model.proj.bias

        three_times_hidden_size, c, _, _ = weight.shape
        pad_value = 4 - c
        preprocessed_weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, pad_value))
        preprocessed_weight = torch.permute(preprocessed_weight, (2, 3, 1, 0))
        preprocessed_weight = torch.reshape(preprocessed_weight, (-1, three_times_hidden_size))

        parameters = {"patch_embeddings": {}}
        parameters["patch_embeddings"] = {"projection": {}}
        parameters["patch_embeddings"]["projection"]["weight"] = ttnn.from_torch(
            preprocessed_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        parameters["patch_embeddings"]["projection"]["bias"] = ttnn.from_torch(
            bias.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
    elif isinstance(torch_model, attention_class):
        num_heads = 16
        qkv_weight, qkv_bias, proj_weight, proj_bias = (
            torch_model.qkv.weight,
            torch_model.qkv.bias,
            torch_model.proj.weight,
            torch_model.proj.bias,
        )
        qkv_weight, qkv_bias, proj_weight, proj_bias = upchannel_attn_weight_bias(
            qkv_weight, qkv_bias, proj_weight, proj_bias, num_heads
        )
        parameters = {"query_key_value": {}, "proj": {}}
        parameters["query_key_value"]["weight"] = preprocess_linear_weight(qkv_weight, dtype=ttnn.bfloat16)
        parameters["query_key_value"]["bias"] = preprocess_linear_bias(qkv_bias, dtype=ttnn.bfloat16)
        parameters["proj"]["weight"] = preprocess_linear_weight(proj_weight, dtype=ttnn.bfloat16)
        parameters["proj"]["bias"] = preprocess_linear_bias(proj_bias, dtype=ttnn.bfloat16)
    elif isinstance(torch_model, torch.nn.Linear):
        parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat16)

    return parameters


def ttnn_featurizer(embedding, encoder, pixel):
    """Helper function to run embedding + encoder."""
    embd = embedding(pixel)
    # Encoder's layer_norm requires TILE layout
    embd = ttnn.to_layout(embd, layout=ttnn.TILE_LAYOUT)
    return encoder(embd)
