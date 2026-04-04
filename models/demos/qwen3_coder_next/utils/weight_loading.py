# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading and mapping utilities for Qwen3-Coder-Next.

Handles conversion from HuggingFace safetensors format to TT implementation format,
including expert weight sharding across devices.

Expert Sharding Strategy:
    - 512 total experts across 8 devices = 64 experts per device
    - Experts are sharded contiguously: device 0 gets experts 0-63, device 1 gets 64-127, etc.
    - Shared expert is replicated across all devices
    - Router weights are replicated across all devices
    - Non-expert weights (embedding, attention, norms, lm_head) are tensor-parallel sharded
"""

from typing import Optional

import torch

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


# HF state dict key prefixes for Qwen3-Coder-Next
# Based on HuggingFace Qwen3NextForCausalLM model structure
HF_KEY_MAP = {
    # Embeddings
    "model.embed_tokens.weight": "embedding.weight",
    # Final norm
    "model.norm.weight": "norm.weight",
    # LM head
    "lm_head.weight": "lm_head.weight",
}

# Per-layer key patterns shared by ALL layer types (replace {layer} with layer index)
# VERIFIED against actual Qwen/Qwen3-Coder-Next checkpoint (74391 keys total)
HF_LAYER_KEY_PATTERNS_COMMON = {
    # Norms
    "model.layers.{layer}.input_layernorm.weight": "layers.{layer}.input_norm.weight",
    "model.layers.{layer}.post_attention_layernorm.weight": "layers.{layer}.post_attn_norm.weight",
    # MoE routing gate
    "model.layers.{layer}.mlp.gate.weight": "layers.{layer}.moe.gate.weight",
    # Shared expert MLP
    "model.layers.{layer}.mlp.shared_expert.gate_proj.weight": "layers.{layer}.moe.shared_expert.gate_proj.weight",
    "model.layers.{layer}.mlp.shared_expert.up_proj.weight": "layers.{layer}.moe.shared_expert.up_proj.weight",
    "model.layers.{layer}.mlp.shared_expert.down_proj.weight": "layers.{layer}.moe.shared_expert.down_proj.weight",
    # Shared expert output gate (scales shared expert contribution)
    "model.layers.{layer}.mlp.shared_expert_gate.weight": "layers.{layer}.moe.shared_expert_gate.weight",
    # Per-expert weights handled separately via EXPERT_KEY_PATTERN
}

# GQA attention keys (layers 3, 7, 11, ..., 47 — every 4th layer)
# Uses standard separate Q/K/V/O projections + QK-norm
HF_LAYER_KEY_PATTERNS_GQA = {
    "model.layers.{layer}.self_attn.q_proj.weight": "layers.{layer}.attn.q_proj.weight",
    "model.layers.{layer}.self_attn.k_proj.weight": "layers.{layer}.attn.k_proj.weight",
    "model.layers.{layer}.self_attn.v_proj.weight": "layers.{layer}.attn.v_proj.weight",
    "model.layers.{layer}.self_attn.o_proj.weight": "layers.{layer}.attn.o_proj.weight",
    "model.layers.{layer}.self_attn.q_norm.weight": "layers.{layer}.attn.q_norm.weight",
    "model.layers.{layer}.self_attn.k_norm.weight": "layers.{layer}.attn.k_norm.weight",
}

# DeltaNet attention keys (layers 0, 1, 2, 4, 5, 6, ... — 3 of every 4 layers)
# Uses FUSED projections: in_proj_qkvz (Q+K+V+Z fused), in_proj_ba (beta+alpha fused)
# Single conv1d (not separate q_conv/k_conv), A_log (decay), dt_bias, norm (RMSNormGated)
HF_LAYER_KEY_PATTERNS_DELTANET = {
    "model.layers.{layer}.linear_attn.in_proj_qkvz.weight": "layers.{layer}.linear_attn.in_proj_qkvz.weight",
    "model.layers.{layer}.linear_attn.in_proj_ba.weight": "layers.{layer}.linear_attn.in_proj_ba.weight",
    "model.layers.{layer}.linear_attn.conv1d.weight": "layers.{layer}.linear_attn.conv1d.weight",
    "model.layers.{layer}.linear_attn.A_log": "layers.{layer}.linear_attn.A_log",
    "model.layers.{layer}.linear_attn.dt_bias": "layers.{layer}.linear_attn.dt_bias",
    "model.layers.{layer}.linear_attn.norm.weight": "layers.{layer}.linear_attn.norm.weight",
    "model.layers.{layer}.linear_attn.out_proj.weight": "layers.{layer}.linear_attn.out_proj.weight",
}


def get_layer_key_patterns(layer_idx: int, full_attention_interval: int = 4) -> dict:
    """Get the correct key mapping patterns for a given layer type.

    Args:
        layer_idx: Layer index (0-47).
        full_attention_interval: GQA layer interval (default 4).

    Returns:
        Dict of HF key pattern -> TT key pattern for this layer type.
    """
    is_gqa = layer_idx % full_attention_interval == (full_attention_interval - 1)
    attn_patterns = HF_LAYER_KEY_PATTERNS_GQA if is_gqa else HF_LAYER_KEY_PATTERNS_DELTANET
    return {**HF_LAYER_KEY_PATTERNS_COMMON, **attn_patterns}


def preprocess_deltanet_state_dict(state_dict, layer_idx, config=None):
    """Split fused DeltaNet projections into separate weights for GatedDeltaNet.

    Qwen3-Coder-Next checkpoint has fused projections:
        in_proj_qkvz (12288, 2048) = [Q, K, V, Z] fused
        in_proj_ba (64, 2048) = [beta, alpha] fused

    GatedDeltaNet (changh95) expects separate:
        in_proj_qkv (8192, 2048) = [Q, K, V]
        in_proj_z (4096, 2048) = Z
        in_proj_b (32, 2048) = beta
        in_proj_a (32, 2048) = alpha

    Args:
        state_dict: HF state dict (or layer subset)
        layer_idx: Layer index
        config: Model config (for dimensions). If None, uses defaults.

    Returns:
        Modified state_dict with split projections.
    """
    prefix = f"model.layers.{layer_idx}.linear_attn"
    # Also support prefix without "model." (for stripped dicts)
    alt_prefix = f"layers.{layer_idx}.linear_attn"

    for pfx in [prefix, alt_prefix]:
        qkvz_key = f"{pfx}.in_proj_qkvz.weight"
        ba_key = f"{pfx}.in_proj_ba.weight"

        if qkvz_key in state_dict:
            qkvz = state_dict.pop(qkvz_key)
            # Split: key_dim=2048, value_dim=4096 (16*128 and 32*128)
            key_dim = (config.linear_num_key_heads if config else 16) * (config.linear_key_head_dim if config else 128)
            value_dim = (config.linear_num_value_heads if config else 32) * (
                config.linear_value_head_dim if config else 128
            )
            q, k, v, z = qkvz.split([key_dim, key_dim, value_dim, value_dim], dim=0)
            state_dict[f"{pfx}.in_proj_qkv.weight"] = torch.cat([q, k, v], dim=0)
            state_dict[f"{pfx}.in_proj_z.weight"] = z

        if ba_key in state_dict:
            ba = state_dict.pop(ba_key)
            num_v_heads = config.linear_num_value_heads if config else 32
            b, a = ba.split([num_v_heads, num_v_heads], dim=0)
            state_dict[f"{pfx}.in_proj_b.weight"] = b
            state_dict[f"{pfx}.in_proj_a.weight"] = a

    return state_dict


# Expert key pattern template
EXPERT_KEY_PATTERN = "model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
EXPERT_PROJS = ["gate_proj", "up_proj", "down_proj"]


def get_expert_device_mapping(num_experts: int = 512, num_devices: int = 8) -> dict:
    """Compute which experts are assigned to which device.

    Expert sharding: contiguous assignment.
    Device 0: experts 0..63
    Device 1: experts 64..127
    ...
    Device 7: experts 448..511

    Args:
        num_experts: Total number of experts.
        num_devices: Number of devices to shard across.

    Returns:
        Dict mapping device_id -> list of expert indices.
    """
    experts_per_device = num_experts // num_devices
    assert (
        num_experts % num_devices == 0
    ), f"num_experts ({num_experts}) must be evenly divisible by num_devices ({num_devices})"

    mapping = {}
    for device_id in range(num_devices):
        start = device_id * experts_per_device
        end = start + experts_per_device
        mapping[device_id] = list(range(start, end))

    return mapping


def get_expert_for_device(expert_idx: int, num_experts: int = 512, num_devices: int = 8) -> int:
    """Get which device an expert is assigned to.

    Args:
        expert_idx: Global expert index (0-511).
        num_experts: Total number of experts.
        num_devices: Number of devices.

    Returns:
        Device ID (0-7).
    """
    experts_per_device = num_experts // num_devices
    return expert_idx // experts_per_device


def get_local_expert_idx(global_expert_idx: int, num_experts: int = 512, num_devices: int = 8) -> int:
    """Convert global expert index to device-local expert index.

    Args:
        global_expert_idx: Global expert index (0-511).
        num_experts: Total number of experts.
        num_devices: Number of devices.

    Returns:
        Local expert index within device (0-63).
    """
    experts_per_device = num_experts // num_devices
    return global_expert_idx % experts_per_device


def extract_layer_state_dict(
    full_state_dict: dict,
    layer_idx: int,
    num_experts: int = 512,
) -> dict:
    """Extract state dict for a single decoder layer.

    Args:
        full_state_dict: Full model state dict from HF.
        layer_idx: Layer index (0-47).
        num_experts: Total number of experts.

    Returns:
        Dict with layer-specific weights (HF key names preserved).
    """
    prefix = f"model.layers.{layer_idx}."
    layer_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith(prefix):
            # Strip the layer prefix for cleaner access
            local_key = key[len(prefix) :]
            layer_dict[local_key] = value
    return layer_dict


def extract_expert_weights_for_device(
    layer_state_dict: dict,
    device_id: int,
    num_experts: int = 512,
    num_devices: int = 8,
) -> dict:
    """Extract expert weights assigned to a specific device.

    Args:
        layer_state_dict: State dict for a single layer (from extract_layer_state_dict).
        device_id: Target device ID (0-7).
        num_experts: Total number of experts.
        num_devices: Number of devices.

    Returns:
        Dict mapping local_expert_idx -> {proj_name: weight_tensor}.
    """
    expert_mapping = get_expert_device_mapping(num_experts, num_devices)
    device_experts = expert_mapping[device_id]

    expert_weights = {}
    for global_idx in device_experts:
        local_idx = get_local_expert_idx(global_idx, num_experts, num_devices)
        expert_weights[local_idx] = {}
        for proj in EXPERT_PROJS:
            key = f"mlp.experts.{global_idx}.{proj}.weight"
            if key in layer_state_dict:
                expert_weights[local_idx][proj] = layer_state_dict[key]

    return expert_weights


def load_and_shard_weights(
    model_name_or_path: str,
    num_devices: int = 8,
    cache_dir: Optional[str] = None,
) -> dict:
    """Load weights and prepare sharding metadata.

    This is a high-level utility that loads the full state dict and prepares
    metadata for device-specific weight extraction.

    Args:
        model_name_or_path: HF model name or local path.
        num_devices: Number of devices to shard across.
        cache_dir: Optional cache directory for downloaded weights.

    Returns:
        Dict with 'state_dict', 'config', 'expert_mapping', 'num_layers'.
    """
    from models.demos.qwen3_coder_next.reference.model import load_reference_state_dict

    state_dict, config = load_reference_state_dict(model_name_or_path)

    expert_mapping = get_expert_device_mapping(
        num_experts=config.num_experts if hasattr(config, "num_experts") else 512,
        num_devices=num_devices,
    )

    return {
        "state_dict": state_dict,
        "config": config,
        "expert_mapping": expert_mapping,
        "num_layers": config.num_hidden_layers,
        "num_devices": num_devices,
        "experts_per_device": len(next(iter(expert_mapping.values()))),
    }
