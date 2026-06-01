# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Weight loading for Qwen3.6-27B.

Downloads from HuggingFace, concatenates safetensors shards into a single
state_dict. Weights are loaded as torch tensors first, then converted to
BFP4_B on the device during model construction.
"""

import os
from pathlib import Path

import torch
from models.demos.qwen36_27b.tt.model_config import Qwen36ModelConfig


def load_state_dict(
    config: Qwen36ModelConfig,
    max_layers: int | None = None,
    model_path: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Load Qwen3.6-27B state dict from HuggingFace cache or download.

    Args:
        config: Model configuration
        max_layers: If set, only load weights for the first N layers (for testing)
        model_path: If set, load from this directory instead of downloading

    Returns:
        state_dict: dict of parameter name → torch.Tensor
    """
    if model_path is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError("huggingface_hub required: pip install huggingface_hub")

        model_path = snapshot_download(config.model_name, allow_patterns=["*.safetensors", "*.json"])

    return _load_safetensors(model_path, max_layers)


def _normalize_key(key: str) -> str:
    """Normalize weight keys to canonical form (strip 'language_model.' prefix)."""
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    return key


def _load_safetensors(model_path: str, max_layers: int | None = None) -> dict[str, torch.Tensor]:
    """Load all safetensors shards into a single dict."""
    try:
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("safetensors required: pip install safetensors")

    model_path = Path(model_path)
    shard_files = sorted(model_path.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    state_dict = {}
    for shard_file in shard_files:
        shard = load_file(str(shard_file))
        for key, tensor in shard.items():
            norm_key = _normalize_key(key)
            if max_layers is not None and _is_layer_weight(norm_key):
                layer_idx = _extract_layer_idx(norm_key)
                if layer_idx is not None and layer_idx >= max_layers:
                    continue
            state_dict[norm_key] = tensor

    return state_dict


def _is_layer_weight(key: str) -> bool:
    return "model.layers." in key


def _extract_layer_idx(key: str) -> int | None:
    parts = key.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None


def create_dummy_state_dict(config: Qwen36ModelConfig, num_layers: int = 4) -> dict[str, torch.Tensor]:
    """
    Create a dummy state dict for testing without downloading the real model.
    All weights are random with correct shapes.
    """
    sd = {}
    H = config.hidden_size
    I = config.intermediate_size
    V = config.vocab_size

    sd["model.embed_tokens.weight"] = torch.randn(V, H, dtype=torch.bfloat16)

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        layer_type = config.layer_types[i]

        sd[f"{prefix}.input_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
        sd[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(H, dtype=torch.bfloat16)

        # MLP
        sd[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16)
        sd[f"{prefix}.mlp.up_proj.weight"] = torch.randn(I, H, dtype=torch.bfloat16)
        sd[f"{prefix}.mlp.down_proj.weight"] = torch.randn(H, I, dtype=torch.bfloat16)

        if layer_type == "linear_attention":
            nk = config.linear_num_key_heads
            nv = config.linear_num_value_heads
            dk = config.linear_key_head_dim
            dv = config.linear_value_head_dim
            conv_dim = nk * dk * 2 + nv * dv

            sd[f"{prefix}.linear_attn.in_proj_qkv.weight"] = torch.randn(conv_dim, H, dtype=torch.bfloat16)
            sd[f"{prefix}.linear_attn.in_proj_z.weight"] = torch.randn(nv * dv, H, dtype=torch.bfloat16)
            sd[f"{prefix}.linear_attn.in_proj_b.weight"] = torch.randn(nv, H, dtype=torch.bfloat16)
            sd[f"{prefix}.linear_attn.in_proj_a.weight"] = torch.randn(nv, H, dtype=torch.bfloat16)
            sd[f"{prefix}.linear_attn.out_proj.weight"] = torch.randn(H, nv * dv, dtype=torch.bfloat16)
            sd[f"{prefix}.linear_attn.norm.weight"] = torch.ones(dv, dtype=torch.bfloat16)
            sd[f"{prefix}.linear_attn.A_log"] = torch.log(torch.rand(nv) * 16 + 0.1)
            sd[f"{prefix}.linear_attn.dt_bias"] = torch.ones(nv)
            sd[f"{prefix}.linear_attn.conv1d.weight"] = torch.randn(conv_dim, 1, config.linear_conv_kernel_dim, dtype=torch.bfloat16)

        else:  # full_attention
            nh = config.num_attention_heads
            nkv = config.num_key_value_heads
            dh = config.head_dim

            sd[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(nh * dh * 2, H, dtype=torch.bfloat16)
            sd[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(nkv * dh, H, dtype=torch.bfloat16)
            sd[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(nkv * dh, H, dtype=torch.bfloat16)
            sd[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(H, nh * dh, dtype=torch.bfloat16)
            sd[f"{prefix}.self_attn.q_norm.weight"] = torch.ones(dh, dtype=torch.bfloat16)
            sd[f"{prefix}.self_attn.k_norm.weight"] = torch.ones(dh, dtype=torch.bfloat16)

    sd["model.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
    sd["lm_head.weight"] = torch.randn(V, H, dtype=torch.bfloat16)

    return sd
