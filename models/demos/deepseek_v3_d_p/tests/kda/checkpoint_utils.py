# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Indexed safetensor loading for one Kimi Delta Attention layer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.weights import required_kda_weight_names, validate_kda_weights
from models.demos.deepseek_v3_d_p.tt.kda.weight_schema import normalize_kda_a_log

KIMI_K3_FIRST_KDA_LAYER = 1
KIMI_K3_HF_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
KIMI_K3_LAYER_1_SHA256 = "10b99878599e02d002f2566b04c9cc7433da6d267351991978f6e348478f3097"


def kda_state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Return a canonical content identity for the normalized KDA layer weights."""
    fingerprint = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":"))
        fingerprint.update(metadata.encode())
        fingerprint.update(memoryview(tensor.view(torch.uint8).numpy()))
    return fingerprint.hexdigest()


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
    state_dict["A_log"] = normalize_kda_a_log(state_dict["A_log"], config)
    validate_kda_weights(state_dict, config)
    return state_dict
