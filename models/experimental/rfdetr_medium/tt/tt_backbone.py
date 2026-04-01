# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN DINOv2-ViT-S backbone with windowed attention for RF-DETR Medium.
Pure TTNN — all computation on device.

Architecture:
  - 12 transformer blocks (384-dim, 6 heads, GELU MLP)
  - No register tokens (dinov2_windowed_small)
  - patch_size=16, resolution=576 → 36×36 = 1296 patches
  - 2×2 = 4 windows, each with 1 CLS + 324 patches = 325 tokens
  - Tensor shape between layers: [B*4, 325, 384] (windowed form)
  - Full attention at layers {3,6,9}: ttnn.reshape to [B, 1300, 384]
  - Windowed layers {0,1,2,4,5,7,8,10,11}: regular SDPA on [B*4, 325, 384]
  - Features at stages [3,6,9,12] = after layers [2,5,8,11]
  - Feature extraction: strip CLS → un-window → [B, 384, 36, 36]

Windowed attention pattern from: models/demos/qwen25_vl/tt/vision_attention.py
DINOv2 patterns from: models/experimental/openvla/tt/tt_optimized_openvla_vision.py
"""

import math
import ttnn

from models.experimental.rfdetr_medium.common import (
    VIT_HIDDEN_SIZE,
    VIT_NUM_HEADS,
    VIT_NUM_LAYERS,
    VIT_HEAD_SIZE,
    NUM_PATCHES,
    NUM_PATCHES_PER_SIDE,
    NUM_WINDOWS,
    NUM_WINDOWS_SQUARED,
    PATCHES_PER_WINDOW_SIDE,
    PATCHES_PER_WINDOW,
    TOKENS_PER_WINDOW,
    FULL_ATTN_SEQ_LEN,
    PATCH_SIZE,
    OUT_FEATURE_INDEXES,
    FULL_ATTN_LAYER_INDEXES,
)

try:
    _CoreGrid = ttnn.CoreGrid
except AttributeError:
    _CoreGrid = ttnn.types.CoreGrid
CORE_GRID = _CoreGrid(y=8, x=12)

# HiFi4 + fp32 accumulation for maximum precision in backbone matmuls/softmax
COMPUTE_CONFIG = None  # lazily initialized per-device


def _get_compute_config(device):
    global COMPUTE_CONFIG
    if COMPUTE_CONFIG is None:
        COMPUTE_CONFIG = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )
    return COMPUTE_CONFIG


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------


def dinov2_patch_embeddings(pixel_values, proj_weight, proj_bias, device):
    """
    Patch embedding via fold + linear (on device).
    Conv2d(3, 384, 16, 16) decomposed as fold then matmul.

    Input:  [B, H, W, 4] NHWC padded on device
    Output: [B, 1296, 384] on device
    """
    batch_size = pixel_values.shape[0]
    img_h = pixel_values.shape[1]
    patch_count = img_h // PATCH_SIZE

    pixel_values = ttnn.reshape(pixel_values, (batch_size, img_h, img_h // PATCH_SIZE, 4 * PATCH_SIZE))
    pixel_values = ttnn.fold(pixel_values, PATCH_SIZE, 1)
    pixel_values = ttnn.to_layout(pixel_values, layout=ttnn.TILE_LAYOUT)

    patches = ttnn.linear(
        pixel_values,
        proj_weight,
        bias=proj_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
        compute_kernel_config=_get_compute_config(device),
    )
    ttnn.deallocate(pixel_values)

    patches = ttnn.to_layout(patches, layout=ttnn.ROW_MAJOR_LAYOUT)
    patches = ttnn.reshape(patches, (batch_size, NUM_PATCHES, VIT_HIDDEN_SIZE))
    return patches


# ---------------------------------------------------------------------------
# Window Partition / Unpartition (device-side reshape+permute)
# ---------------------------------------------------------------------------


def window_partition(patches, cls_token, pos_embed, batch_size, device):
    """
    Embedding + window partition, all on device.

    Steps:
      1. Prepend CLS → [B, 1297, 384]
      2. Add positional encoding
      3. Separate CLS and patches
      4. Reshape patches into 2×2 windows: [B*4, 324, 384]
      5. Replicate CLS per window: [B*4, 1, 384]
      6. Concat: [B*4, 325, 384]

    Input:  patches [B, 1296, 384], cls_token [1, 1, 384], pos_embed [1, 1297, 384]
    Output: [B*4, 325, 384] on device
    """
    patches = ttnn.to_layout(patches, layout=ttnn.TILE_LAYOUT)

    # Prepend CLS: [B, 1, 384] + [B, 1296, 384] → [B, 1297, 384]
    cls_expanded = ttnn.to_layout(cls_token, layout=ttnn.TILE_LAYOUT)
    if batch_size > 1:
        cls_expanded = ttnn.repeat(cls_expanded, ttnn.Shape([batch_size, 1, 1]))
    embeddings = ttnn.concat([cls_expanded, patches], dim=1)
    ttnn.deallocate(patches)

    # Add positional encoding
    embeddings = ttnn.add(embeddings, pos_embed)

    # Separate CLS [B, 1, 384] and patches [B, 1296, 384]
    embeddings = ttnn.to_layout(embeddings, layout=ttnn.ROW_MAJOR_LAYOUT)
    cls_tokens = embeddings[:, :1, :]
    patch_tokens = embeddings[:, 1:, :]
    ttnn.deallocate(embeddings)

    # Reshape patches into spatial grid: [B, 1296, 384] → [B, 36, 36, 384]
    patch_tokens = ttnn.reshape(patch_tokens, (batch_size, NUM_PATCHES_PER_SIDE, NUM_PATCHES_PER_SIDE, VIT_HIDDEN_SIZE))

    # Split into 2×2 windows using reshape+permute
    # [B, 36, 36, 384] → [B*2, 18, 36, 384] → [B*2, 18, 2, 18, 384] → permute → [B*4, 18, 18, 384]
    patch_tokens = ttnn.reshape(
        patch_tokens, (batch_size * NUM_WINDOWS, PATCHES_PER_WINDOW_SIDE, NUM_PATCHES_PER_SIDE, VIT_HIDDEN_SIZE)
    )
    patch_tokens = ttnn.reshape(
        patch_tokens,
        (batch_size * NUM_WINDOWS, PATCHES_PER_WINDOW_SIDE, NUM_WINDOWS, PATCHES_PER_WINDOW_SIDE, VIT_HIDDEN_SIZE),
    )
    patch_tokens = ttnn.permute(patch_tokens, (0, 2, 1, 3, 4))
    patch_tokens = ttnn.reshape(patch_tokens, (batch_size * NUM_WINDOWS_SQUARED, PATCHES_PER_WINDOW, VIT_HIDDEN_SIZE))

    # Replicate CLS for each window: [B, 1, 384] → [B*4, 1, 384]
    cls_tokens = ttnn.to_layout(cls_tokens, layout=ttnn.TILE_LAYOUT)
    cls_replicated = ttnn.concat([cls_tokens] * NUM_WINDOWS_SQUARED, dim=0)

    # Concat CLS + patches per window: [B*4, 325, 384]
    patch_tokens = ttnn.to_layout(patch_tokens, layout=ttnn.TILE_LAYOUT)
    windowed = ttnn.concat([cls_replicated, patch_tokens], dim=1)

    return windowed


def extract_feature_map(hidden_states, batch_size, device, ln_weight=None, ln_bias=None):
    """
    Extract spatial feature map from windowed hidden states (on device).

    Input:  [B*4, 325, 384] windowed
    Output: [B, 384, 36, 36] NCHW (on device, for projector)

    Steps:
      0. Apply layernorm (matching reference WindowedDinov2WithRegistersBackbone)
      1. Strip CLS → [B*4, 324, 384]
      2. Reshape to spatial windows: [B*4, 18, 18, 384]
      3. Un-partition windows: [B, 36, 36, 384]
      4. Permute to NCHW: [B, 384, 36, 36]
    """
    hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)

    if ln_weight is not None:
        hidden_states = ttnn.layer_norm(
            hidden_states,
            weight=ln_weight,
            bias=ln_bias,
            epsilon=1e-06,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=_get_compute_config(device),
        )

    hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Strip CLS token
    patches = hidden_states[:, 1:, :]  # [B*4, 324, 384]

    # Reshape to spatial windows: [B*4, 18, 18, 384]
    patches = ttnn.reshape(
        patches, (batch_size * NUM_WINDOWS_SQUARED, PATCHES_PER_WINDOW_SIDE, PATCHES_PER_WINDOW_SIDE, VIT_HIDDEN_SIZE)
    )

    # Un-partition: reverse of the partitioning
    # [B*4, 18, 18, 384] → [B*2, 2, 18, 18, 384] → permute → [B*2, 18, 2, 18, 384] → [B, 36, 36, 384]
    patches = ttnn.reshape(
        patches,
        (batch_size * NUM_WINDOWS, NUM_WINDOWS, PATCHES_PER_WINDOW_SIDE, PATCHES_PER_WINDOW_SIDE, VIT_HIDDEN_SIZE),
    )
    patches = ttnn.permute(patches, (0, 2, 1, 3, 4))
    patches = ttnn.reshape(patches, (batch_size, NUM_PATCHES_PER_SIDE, NUM_PATCHES_PER_SIDE, VIT_HIDDEN_SIZE))

    # NHWC → NCHW: [B, 36, 36, 384] → [B, 384, 36, 36]
    patches = ttnn.permute(patches, (0, 3, 1, 2))

    return patches


# ---------------------------------------------------------------------------
# Attention + FFN blocks
# ---------------------------------------------------------------------------


def dinov2_attention(hidden_states, params, device):
    """
    DINOv2 self-attention (pre-norm, with layer scale).
    Runs on whatever shape is passed — windowed [B*4, 325, 384] or full [B, 1300, 384].

    All on device.
    """
    compute_config = _get_compute_config(device)

    normed = ttnn.layer_norm(
        hidden_states,
        weight=params["norm1_weight"],
        bias=params["norm1_bias"],
        epsilon=1e-06,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=compute_config,
    )

    qkv = ttnn.linear(
        normed,
        params["qkv_weight"],
        bias=params["qkv_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(normed)

    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        qkv,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        num_heads=VIT_NUM_HEADS,
    )
    ttnn.deallocate(qkv)

    scale = 1.0 / math.sqrt(VIT_HEAD_SIZE)
    query = ttnn.mul_(query, scale)
    value = ttnn.reallocate(value)

    attn_scores = ttnn.matmul(
        query,
        key,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(query)
    ttnn.deallocate(key)

    attn_probs = ttnn.softmax_in_place(attn_scores, numeric_stable=True, compute_kernel_config=compute_config)

    context = ttnn.matmul(
        attn_probs,
        value,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(attn_probs)
    ttnn.deallocate(value)

    context = ttnn.transformer.concatenate_heads(context, memory_config=ttnn.L1_MEMORY_CONFIG)

    attn_output = ttnn.linear(
        context,
        params["proj_weight"],
        bias=params["proj_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
        compute_kernel_config=compute_config,
    )
    ttnn.deallocate(context)

    # Layer scale + residual
    attn_output = ttnn.mul(attn_output, params["ls1_scale"])
    output = ttnn.add(hidden_states, attn_output)
    return output


def dinov2_feedforward(hidden_states, params, device):
    """
    DINOv2 FFN (pre-norm): LN → Linear(384→1536) → GELU → Linear(1536→384).
    With layer scale + residual. All on device.
    """
    compute_config = _get_compute_config(device)

    normed = ttnn.layer_norm(
        hidden_states,
        weight=params["norm2_weight"],
        bias=params["norm2_bias"],
        epsilon=1e-06,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=compute_config,
    )

    ff = ttnn.linear(
        normed,
        params["fc1_weight"],
        bias=params["fc1_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
        activation="gelu",
        compute_kernel_config=compute_config,
    )

    ff = ttnn.linear(
        ff,
        params["fc2_weight"],
        bias=params["fc2_bias"],
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        core_grid=CORE_GRID,
        compute_kernel_config=compute_config,
    )

    ff = ttnn.mul(ff, params["ls2_scale"])
    return ttnn.add(hidden_states, ff)


# ---------------------------------------------------------------------------
# Single layer forward (windowed / full attention)
# ---------------------------------------------------------------------------


def dinov2_layer(hidden_states, params, run_full_attention, batch_size, device):
    """
    Single DINOv2 layer. Tensor is always [B*4, 325, 384] between layers.
    Full attention layers reshape to [B, 1300, 384] for global attention.
    All on device.
    """
    if run_full_attention:
        # Merge windows: [B*4, 325, 384] → [B, 1300, 384]
        hidden_states = ttnn.reshape(hidden_states, (batch_size, FULL_ATTN_SEQ_LEN, VIT_HIDDEN_SIZE))
        hidden_states = dinov2_attention(hidden_states, params, device)
        # Split back: [B, 1300, 384] → [B*4, 325, 384]
        hidden_states = ttnn.reshape(
            hidden_states, (batch_size * NUM_WINDOWS_SQUARED, TOKENS_PER_WINDOW, VIT_HIDDEN_SIZE)
        )
    else:
        hidden_states = dinov2_attention(hidden_states, params, device)

    hidden_states = dinov2_feedforward(hidden_states, params, device)
    return hidden_states


# ---------------------------------------------------------------------------
# Full backbone
# ---------------------------------------------------------------------------


def dinov2_backbone(pixel_values, backbone_params, batch_size=1):
    """
    Full DINOv2-ViT-S backbone. Pure TTNN.

    Input:  [B, H, W, 4] NHWC padded on device
    Output: list of 4 TTNN tensors [B, 384, 36, 36] (NCHW, on device)
    """
    device = pixel_values.device()

    # Patch embedding → [B, 1296, 384]
    patches = dinov2_patch_embeddings(
        pixel_values,
        backbone_params["proj_weight"],
        backbone_params["proj_bias"],
        device,
    )

    # Embedding + window partition → [B*4, 325, 384]
    hidden_states = window_partition(
        patches,
        backbone_params["cls_token"],
        backbone_params["pos_embed"],
        batch_size,
        device,
    )

    feature_maps = []

    for layer_idx in range(VIT_NUM_LAYERS):
        run_full = layer_idx in FULL_ATTN_LAYER_INDEXES

        hidden_states = dinov2_layer(
            hidden_states,
            backbone_params["layers"][layer_idx],
            run_full,
            batch_size,
            device,
        )

        # Features at stages [3,6,9,12] = after layers [2,5,8,11]
        stage_num = layer_idx + 1
        if stage_num in OUT_FEATURE_INDEXES:
            feat = extract_feature_map(
                hidden_states,
                batch_size,
                device,
                ln_weight=backbone_params["layernorm_weight"],
                ln_bias=backbone_params["layernorm_bias"],
            )
            feature_maps.append(feat)
            hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)

    return feature_maps
