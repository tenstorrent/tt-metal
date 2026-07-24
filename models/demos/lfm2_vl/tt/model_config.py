# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Configuration for LiquidAI/LFM2.5-VL-1.6B.

Values match the public HuggingFace checkpoint config for
``LiquidAI/LFM2.5-VL-1.6B`` (transformers ``lfm2_vl`` / ``lfm2``).
"""

from __future__ import annotations

from typing import Any, Dict


# Public HF model id for this bring-up.
HF_MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B"

# Layer schedule from text_config.layer_types (16 hybrid layers).
LFM25_VL_LAYER_TYPES = [
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
]


def _adjusted_ff_dim(
    intermediate_size: int,
    *,
    auto_adjust: bool = True,
    ffn_dim_multiplier: float = 1.0,
    multiple_of: int = 256,
) -> int:
    """Mirror Lfm2MLP intermediate-size adjustment (block_auto_adjust_ff_dim)."""
    if not auto_adjust:
        return intermediate_size
    size = int(2 * intermediate_size / 3)
    size = int(ffn_dim_multiplier * size)
    size = multiple_of * ((size + multiple_of - 1) // multiple_of)
    return size


def create_model_config(batch_size: int = 1, seq_len: int = 128) -> Dict[str, Any]:
    """Build the LFM2.5-VL-1.6B config used by the ttnn model and weight loader."""
    text_intermediate = _adjusted_ff_dim(12288)
    return {
        "hf_model_id": HF_MODEL_ID,
        "model_type": "lfm2_vl",
        "batch_size": batch_size,
        "seq_len": seq_len,
        # Multimodal
        "image_token_id": 396,
        "downsample_factor": 2,
        "projector_hidden_act": "gelu",
        "projector_hidden_size": 2048,
        "projector_bias": True,
        "projector_use_layernorm": False,
        "min_image_tokens": 64,
        "max_image_tokens": 256,
        "tile_size": 512,
        "encoder_patch_size": 16,
        "do_image_splitting": True,
        "use_thumbnail": True,
        "use_image_special_tokens": True,
        # Language backbone (LFM2.5-1.2B)
        "hidden_size": 2048,
        "num_heads": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "intermediate_size": text_intermediate,
        "num_hidden_layers": 16,
        "layer_types": list(LFM25_VL_LAYER_TYPES),
        "vocab_size": 65536,
        "norm_eps": 1e-5,
        "rope_theta": 1_000_000.0,
        "max_position_embeddings": 128_000,
        "conv_L_cache": 3,
        "conv_bias": False,
        # Vision tower (SigLIP2 NaFlex 400M)
        "vision_config": {
            "model_type": "siglip2_vision_model",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_channels": 3,
            "patch_size": 16,
            "num_patches": 256,
            "layer_norm_eps": 1e-6,
            "hidden_act": "gelu_pytorch_tanh",
        },
    }
