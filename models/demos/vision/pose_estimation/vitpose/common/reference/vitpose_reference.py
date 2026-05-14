# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation for ViTPose-B (usyd-community/vitpose-base-simple).
Each function mirrors a TTNN implementation for layer-by-layer PCC testing.

Architecture:
  ViT-Base backbone (12 layers, 768 hidden, 12 heads, no CLS token)
  → SimpleDecoder (ReLU → Bilinear Upsample 4x → Conv2d 768→17)
  → 17 keypoint heatmaps (64×48)
"""

import torch
import torch.nn.functional as F


def vitpose_patch_embeddings(pixel_values, *, parameters):
    """
    Conv2d patch embedding for rectangular input (256×192).

    Args:
        pixel_values: (batch, 3, 256, 192) NCHW
        parameters: dict with projection.weight [768,3,16,16] and projection.bias [768]

    Returns:
        (batch, 192, 768) patch embeddings
    """
    dtype = pixel_values.dtype
    output = F.conv2d(
        pixel_values,
        parameters["projection.weight"].to(dtype),
        bias=parameters["projection.bias"].to(dtype),
        stride=(16, 16),
        padding=(2, 2),
    )
    batch_size = output.shape[0]
    output = output.flatten(2).transpose(1, 2)
    return output


def vitpose_embeddings(pixel_values, *, parameters):
    """
    Patch embedding + position encoding.

    Args:
        pixel_values: (batch, 3, 256, 192) NCHW
        parameters: dict with projection weights and position_embeddings [1,193,768]

    Returns:
        (batch, 192, 768) with position encoding added
    """
    patch_emb = vitpose_patch_embeddings(pixel_values, parameters=parameters)
    dtype = patch_emb.dtype
    pos_embeddings = parameters["position_embeddings"].to(dtype)
    embeddings = patch_emb + pos_embeddings[:, 1:] + pos_embeddings[:, :1]
    return embeddings


def vitpose_attention(hidden_states, *, parameters, num_heads=12):
    """
    Multi-head self-attention.

    Args:
        hidden_states: (batch, 192, 768)
        parameters: dict with query/key/value/output dense weights and biases

    Returns:
        (batch, 192, 768)
    """
    batch_size, seq_len, hidden_size = hidden_states.shape
    head_dim = hidden_size // num_heads

    dtype = hidden_states.dtype
    query = F.linear(hidden_states, parameters["attention.query.weight"].to(dtype), parameters["attention.query.bias"].to(dtype))
    key = F.linear(hidden_states, parameters["attention.key.weight"].to(dtype), parameters["attention.key.bias"].to(dtype))
    value = F.linear(hidden_states, parameters["attention.value.weight"].to(dtype), parameters["attention.value.bias"].to(dtype))

    query = query.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    key = key.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 3, 1)
    value = value.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

    attention_scores = (query @ key) * (1.0 / (head_dim**0.5))
    attention_probs = F.softmax(attention_scores, dim=-1)
    context = attention_probs @ value

    context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
    output = F.linear(context, parameters["output.dense.weight"].to(dtype), parameters["output.dense.bias"].to(dtype))
    return output


def vitpose_mlp(hidden_states, *, parameters):
    """
    MLP: fc1 → GELU → fc2.

    Args:
        hidden_states: (batch, 192, 768)
        parameters: dict with fc1/fc2 weights and biases

    Returns:
        (batch, 192, 768)
    """
    dtype = hidden_states.dtype
    output = F.linear(hidden_states, parameters["fc1.weight"].to(dtype), parameters["fc1.bias"].to(dtype))
    output = F.gelu(output)
    output = F.linear(output, parameters["fc2.weight"].to(dtype), parameters["fc2.bias"].to(dtype))
    return output


def vitpose_layer(hidden_states, *, parameters, num_heads=12, layer_norm_eps=1e-6):
    """
    Single transformer block: LN → Attention → residual → LN → MLP → residual.

    Args:
        hidden_states: (batch, 192, 768)
        parameters: dict with all layer parameters

    Returns:
        (batch, 192, 768)
    """
    hidden_size = hidden_states.shape[-1]

    dtype = hidden_states.dtype
    ln1 = F.layer_norm(
        hidden_states,
        (hidden_size,),
        weight=parameters["layernorm_before.weight"].to(dtype),
        bias=parameters["layernorm_before.bias"].to(dtype),
        eps=layer_norm_eps,
    )
    attn_out = vitpose_attention(ln1, parameters=parameters, num_heads=num_heads)
    hidden_states = hidden_states + attn_out

    ln2 = F.layer_norm(
        hidden_states,
        (hidden_size,),
        weight=parameters["layernorm_after.weight"].to(dtype),
        bias=parameters["layernorm_after.bias"].to(dtype),
        eps=layer_norm_eps,
    )
    mlp_out = vitpose_mlp(ln2, parameters=parameters)
    hidden_states = hidden_states + mlp_out

    return hidden_states


def vitpose_encoder(hidden_states, *, parameters, num_layers=12, num_heads=12, layer_norm_eps=1e-6):
    """
    Stack of transformer layers.

    Args:
        hidden_states: (batch, 192, 768)
        parameters: list of layer parameter dicts

    Returns:
        (batch, 192, 768)
    """
    for i in range(num_layers):
        hidden_states = vitpose_layer(
            hidden_states,
            parameters=parameters[i],
            num_heads=num_heads,
            layer_norm_eps=layer_norm_eps,
        )
    return hidden_states


def vitpose_backbone(pixel_values, *, parameters, num_layers=12, num_heads=12, layer_norm_eps=1e-6):
    """
    Full backbone: embeddings → encoder → final layernorm.

    Args:
        pixel_values: (batch, 3, 256, 192) NCHW
        parameters: dict with embedding, encoder, and layernorm parameters

    Returns:
        (batch, 192, 768)
    """
    embeddings = vitpose_embeddings(pixel_values, parameters=parameters["embeddings"])
    output = vitpose_encoder(
        embeddings,
        parameters=parameters["encoder"],
        num_layers=num_layers,
        num_heads=num_heads,
        layer_norm_eps=layer_norm_eps,
    )
    dtype = output.dtype
    output = F.layer_norm(
        output,
        (output.shape[-1],),
        weight=parameters["layernorm.weight"].to(dtype),
        bias=parameters["layernorm.bias"].to(dtype),
        eps=1e-12,
    )
    return output


def vitpose_simple_decoder(hidden_states, *, parameters, patch_height=16, patch_width=12, scale_factor=4):
    """
    SimpleDecoder: ReLU → Bilinear Upsample 4x → Conv2d(768→17, k=3, p=1).

    Args:
        hidden_states: (batch, 192, 768) from backbone
        parameters: dict with conv.weight [17,768,3,3] and conv.bias [17]

    Returns:
        (batch, 17, 64, 48) heatmaps
    """
    batch_size = hidden_states.shape[0]
    hidden_size = hidden_states.shape[-1]

    spatial = hidden_states.reshape(batch_size, patch_height, patch_width, hidden_size)
    spatial = spatial.permute(0, 3, 1, 2)

    dtype = spatial.dtype
    spatial = F.relu(spatial)
    spatial = F.interpolate(spatial.float(), scale_factor=scale_factor, mode="bilinear", align_corners=False).to(dtype)
    heatmaps = F.conv2d(spatial, parameters["conv.weight"].to(dtype), parameters["conv.bias"].to(dtype), stride=1, padding=1)
    return heatmaps


def vitpose_forward(pixel_values, *, parameters, num_layers=12, num_heads=12, layer_norm_eps=1e-6):
    """
    Full ViTPose-B forward pass.

    Args:
        pixel_values: (batch, 3, 256, 192) NCHW
        parameters: dict with all model parameters

    Returns:
        (batch, 17, 64, 48) heatmaps
    """
    backbone_output = vitpose_backbone(
        pixel_values,
        parameters=parameters["backbone"],
        num_layers=num_layers,
        num_heads=num_heads,
        layer_norm_eps=layer_norm_eps,
    )
    heatmaps = vitpose_simple_decoder(backbone_output, parameters=parameters["head"])
    return heatmaps


def extract_reference_parameters(model):
    """
    Extract parameters from HuggingFace VitPoseForPoseEstimation into a flat dict
    structure suitable for the reference functions above.

    Args:
        model: VitPoseForPoseEstimation instance

    Returns:
        Nested dict matching reference function expectations
    """
    sd = model.state_dict()
    params = {"backbone": {"embeddings": {}, "encoder": []}, "head": {}}

    params["backbone"]["embeddings"]["projection.weight"] = sd["backbone.embeddings.patch_embeddings.projection.weight"]
    params["backbone"]["embeddings"]["projection.bias"] = sd["backbone.embeddings.patch_embeddings.projection.bias"]
    params["backbone"]["embeddings"]["position_embeddings"] = sd["backbone.embeddings.position_embeddings"]

    num_layers = model.config.backbone_config.num_hidden_layers
    for i in range(num_layers):
        prefix = f"backbone.encoder.layer.{i}"
        layer_params = {
            "layernorm_before.weight": sd[f"{prefix}.layernorm_before.weight"],
            "layernorm_before.bias": sd[f"{prefix}.layernorm_before.bias"],
            "attention.query.weight": sd[f"{prefix}.attention.attention.query.weight"],
            "attention.query.bias": sd[f"{prefix}.attention.attention.query.bias"],
            "attention.key.weight": sd[f"{prefix}.attention.attention.key.weight"],
            "attention.key.bias": sd[f"{prefix}.attention.attention.key.bias"],
            "attention.value.weight": sd[f"{prefix}.attention.attention.value.weight"],
            "attention.value.bias": sd[f"{prefix}.attention.attention.value.bias"],
            "output.dense.weight": sd[f"{prefix}.attention.output.dense.weight"],
            "output.dense.bias": sd[f"{prefix}.attention.output.dense.bias"],
            "layernorm_after.weight": sd[f"{prefix}.layernorm_after.weight"],
            "layernorm_after.bias": sd[f"{prefix}.layernorm_after.bias"],
            "fc1.weight": sd[f"{prefix}.mlp.fc1.weight"],
            "fc1.bias": sd[f"{prefix}.mlp.fc1.bias"],
            "fc2.weight": sd[f"{prefix}.mlp.fc2.weight"],
            "fc2.bias": sd[f"{prefix}.mlp.fc2.bias"],
        }
        params["backbone"]["encoder"].append(layer_params)

    params["backbone"]["layernorm.weight"] = sd["backbone.layernorm.weight"]
    params["backbone"]["layernorm.bias"] = sd["backbone.layernorm.bias"]

    params["head"]["conv.weight"] = sd["head.conv.weight"]
    params["head"]["conv.bias"] = sd["head.conv.bias"]

    return params
