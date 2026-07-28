# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Indexed safetensor loading for one Kimi Delta Attention layer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open

from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.reference import validate_reference_weights

KDA_COMMON_WEIGHT_NAMES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "A_log",
    "f_a_proj.weight",
    "f_b_proj.weight",
    "dt_bias",
    "b_proj.weight",
    "o_norm.weight",
    "o_proj.weight",
)
KDA_LOW_RANK_GATE_WEIGHT_NAMES = ("g_a_proj.weight", "g_b_proj.weight")
KDA_FULL_RANK_GATE_WEIGHT_NAMES = ("g_proj.weight",)


def required_kda_weight_names(config: KDAConfig) -> tuple[str, ...]:
    """Return the canonical layer-local checkpoint keys for ``config``."""
    gate_names = KDA_FULL_RANK_GATE_WEIGHT_NAMES if config.use_full_rank_gate else KDA_LOW_RANK_GATE_WEIGHT_NAMES
    return KDA_COMMON_WEIGHT_NAMES + gate_names


def kda_layer_prefix(layer_idx: int) -> str:
    """Return Kimi-K3's Hugging Face prefix for one KDA layer."""
    if layer_idx < 0:
        raise ValueError(f"layer_idx must be nonnegative, got {layer_idx}")
    return f"language_model.model.layers.{layer_idx}.self_attn."


def resolve_kda_layer_shards(checkpoint_dir: Path, layer_idx: int, config: KDAConfig) -> tuple[Path, ...]:
    """Resolve and validate the complete local shards containing one KDA layer."""
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensor index: {index_path}")
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file).get("weight_map", {})

    prefix = kda_layer_prefix(layer_idx)
    full_names = {name: f"{prefix}{name}" for name in required_kda_weight_names(config)}
    missing = [name for name, full_name in full_names.items() if full_name not in weight_map]
    if missing:
        raise ValueError(f"layer {layer_idx} checkpoint index is missing KDA weights: {missing}")

    shards = tuple(sorted({checkpoint_dir / weight_map[full_name] for full_name in full_names.values()}))
    missing_shards = [path for path in shards if not path.is_file()]
    if missing_shards:
        raise FileNotFoundError(f"missing complete KDA checkpoint shard(s): {missing_shards}")
    return shards


def _normalize_a_log(a_log: torch.Tensor, config: KDAConfig) -> torch.Tensor:
    """Normalize checkpoint head padding into canonical ``[1,1,H,1]`` form."""
    if a_log.numel() == config.num_heads:
        return a_log.reshape(1, 1, config.num_heads, 1)
    if config.num_heads == 96 and a_log.numel() == 128:
        return a_log.reshape(-1)[: config.num_heads].reshape(1, 1, config.num_heads, 1)
    raise ValueError(f"A_log has {a_log.numel()} entries; expected {config.num_heads} logical heads")


def load_kda_layer_state_dict(checkpoint_dir: Path, layer_idx: int, config: KDAConfig) -> dict[str, torch.Tensor]:
    """Load and canonicalize one KDA layer from complete indexed safetensor shards."""
    checkpoint_dir = Path(checkpoint_dir)
    shards = resolve_kda_layer_shards(checkpoint_dir, layer_idx, config)
    prefix = kda_layer_prefix(layer_idx)
    required = required_kda_weight_names(config)
    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as shard_file:
            shard_keys = set(shard_file.keys())
            for name in required:
                full_name = f"{prefix}{name}"
                if full_name in shard_keys:
                    state_dict[name] = shard_file.get_tensor(full_name)

    missing = [name for name in required if name not in state_dict]
    if missing:
        raise ValueError(f"layer {layer_idx} checkpoint shard(s) are missing KDA weights: {missing}")
    state_dict["A_log"] = _normalize_a_log(state_dict["A_log"], config)
    validate_reference_weights(state_dict, config)
    return state_dict


def normalize_kda_state_dict(state_dict: Mapping[str, torch.Tensor], config: KDAConfig) -> dict[str, torch.Tensor]:
    """Canonicalize a caller-provided layer-local mapping and validate all weights."""
    normalized = dict(state_dict)
    normalized["A_log"] = _normalize_a_log(normalized["A_log"], config)
    validate_reference_weights(normalized, config)
    return normalized
