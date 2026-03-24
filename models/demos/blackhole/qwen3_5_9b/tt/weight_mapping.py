# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Remap HuggingFace Qwen3.5-9B state dict to internal format.

Handles:
- Stripping 'model.language_model.' prefix
- Filtering out vision encoder and MTP weights
- Splitting combined in_proj_qkv into separate Q, K, V projections (DeltaNet layers)
- Splitting combined conv1d.weight into separate Q, K, V conv weights (DeltaNet layers)
- Renaming lm_head.weight → output.weight
- Renaming embed_tokens → tok_embeddings
"""
from typing import Dict

import torch

# Layer indices that use full (softmax) attention
FULL_ATTENTION_LAYERS = {3, 7, 11, 15, 19, 23, 27, 31}

# DeltaNet QKV split dimensions
# Q: num_key_heads(16) × key_head_dim(128) = 2048
# K: num_key_heads(16) × key_head_dim(128) = 2048
# V: num_value_heads(32) × value_head_dim(128) = 4096
LINEAR_Q_DIM = 2048
LINEAR_K_DIM = 2048
LINEAR_V_DIM = 4096


def remap_qwen35_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remap HF Qwen3.5-9B state dict to internal format.

    Args:
        state_dict: Raw HuggingFace state dict loaded from safetensors.

    Returns:
        Remapped state dict with internal naming convention.
    """
    remapped = {}

    for key, tensor in state_dict.items():
        # Filter out vision encoder weights (check original key — no prefix stripping yet)
        if "visual" in key or key.startswith("model.visual"):
            continue
        # Filter out MTP (multi-token prediction) weights (original key)
        if key.startswith("mtp"):
            continue

        # Strip model.language_model. prefix
        # Note: after this point, use `new_key` for language model weights,
        # but `key` for top-level weights like lm_head.weight that have no prefix.
        new_key = key
        if new_key.startswith("model.language_model."):
            new_key = new_key[len("model.language_model.") :]

        # Rename top-level weights
        if new_key == "embed_tokens.weight":
            remapped["tok_embeddings.weight"] = tensor
            continue
        if key == "lm_head.weight":
            remapped["output.weight"] = tensor
            continue
        # Final norm (model.language_model.norm.weight)
        if new_key == "norm.weight":
            remapped["norm.weight"] = tensor
            continue

        # Handle per-layer weights
        if new_key.startswith("layers."):
            parts = new_key.split(".")
            layer_idx = int(parts[1])
            layer_prefix = f"layers.{layer_idx}"
            sub_key = ".".join(parts[2:])

            # DeltaNet layers: keep combined QKV AND split for backward compat
            if sub_key == "linear_attn.in_proj_qkv.weight":
                qkv = tensor  # [8192, 4096]
                # Keep combined weight for fused QKV projection
                remapped[f"{layer_prefix}.linear_attn.qkv_proj.weight"] = qkv
                # Also split for any code that still uses separate weights
                q = qkv[:LINEAR_Q_DIM, :]
                k = qkv[LINEAR_Q_DIM : LINEAR_Q_DIM + LINEAR_K_DIM, :]
                v = qkv[LINEAR_Q_DIM + LINEAR_K_DIM :, :]
                remapped[f"{layer_prefix}.linear_attn.q_proj.weight"] = q
                remapped[f"{layer_prefix}.linear_attn.k_proj.weight"] = k
                remapped[f"{layer_prefix}.linear_attn.v_proj.weight"] = v
                continue

            if sub_key == "linear_attn.conv1d.weight":
                conv = tensor  # [8192, 1, 4]
                q_conv = conv[:LINEAR_Q_DIM, :, :]
                k_conv = conv[LINEAR_Q_DIM : LINEAR_Q_DIM + LINEAR_K_DIM, :, :]
                v_conv = conv[LINEAR_Q_DIM + LINEAR_K_DIM :, :, :]
                remapped[f"{layer_prefix}.linear_attn.q_conv.weight"] = q_conv
                remapped[f"{layer_prefix}.linear_attn.k_conv.weight"] = k_conv
                remapped[f"{layer_prefix}.linear_attn.v_conv.weight"] = v_conv
                continue

            # All other keys pass through unchanged
            remapped[new_key] = tensor
            continue

        # Any remaining keys pass through
        remapped[new_key] = tensor

    return remapped
