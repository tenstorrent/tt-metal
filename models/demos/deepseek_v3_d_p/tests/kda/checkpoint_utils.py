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
from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict, required_kda_weight_names

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


# Kimi-K3 ships two checkpoints and they root their keys differently: the published MXFP4 one is a
# multimodal wrapper and spells everything `language_model.model.…`, while the dequantized export
# drops the wrapper. KDA weights are bf16 in both (`quantization_config.ignore` covers
# `re:.*self_attn.*`), so either is loadable and only the root differs. Hardcoding one meant the
# dequantized export -- the only one that loads end to end -- could not be read at all (#54837).
KIMI_K3_WRAPPED_ROOT = "language_model.model."
KIMI_K3_BARE_ROOT = "model."


def kda_layer_prefix(layer_idx: int, model_root: str = KIMI_K3_WRAPPED_ROOT) -> str:
    """Return Kimi-K3's Hugging Face prefix for one KDA layer."""
    if layer_idx < 0:
        raise ValueError(f"layer_idx must be nonnegative, got {layer_idx}")
    return f"{model_root}layers.{layer_idx}.self_attn."


def resolve_model_root(checkpoint_dir: Path) -> str:
    """Which of the two key roots this checkpoint uses, read off its index rather than guessed.

    Checked against `layers.` rather than `embed_tokens.weight` so a partial index holding only the
    layers a test needs still resolves. `language_model.model.` is tried first because `model.` is a
    suffix of it and would otherwise match the wrapped keys too.
    """
    index_path = Path(checkpoint_dir) / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensor index: {index_path}")
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file).get("weight_map", {})
    for root in (KIMI_K3_WRAPPED_ROOT, KIMI_K3_BARE_ROOT):
        if any(name.startswith(f"{root}layers.") for name in weight_map):
            return root
    return KIMI_K3_WRAPPED_ROOT


def resolve_kda_layer_shards(
    checkpoint_dir: Path, layer_idx: int, config: KDAConfig, model_root: str | None = None
) -> tuple[Path, ...]:
    """Resolve and validate the complete local shards containing one KDA layer."""
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensor index: {index_path}")
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file).get("weight_map", {})

    prefix = kda_layer_prefix(layer_idx, model_root or resolve_model_root(checkpoint_dir))
    full_names = {name: f"{prefix}{name}" for name in required_kda_weight_names(config)}
    missing = [name for name, full_name in full_names.items() if full_name not in weight_map]
    if missing:
        raise ValueError(f"layer {layer_idx} checkpoint index is missing KDA weights: {missing}")

    shards = tuple(sorted({checkpoint_dir / weight_map[full_name] for full_name in full_names.values()}))
    missing_shards = [path for path in shards if not path.is_file()]
    if missing_shards:
        raise FileNotFoundError(f"missing complete KDA checkpoint shard(s): {missing_shards}")
    return shards


def load_kda_layer_state_dict(
    checkpoint_dir: Path, layer_idx: int, config: KDAConfig, model_root: str | None = None
) -> dict[str, torch.Tensor]:
    """Load and canonicalize one KDA layer from complete indexed safetensor shards."""
    checkpoint_dir = Path(checkpoint_dir)
    model_root = model_root or resolve_model_root(checkpoint_dir)
    shards = resolve_kda_layer_shards(checkpoint_dir, layer_idx, config, model_root)
    prefix = kda_layer_prefix(layer_idx, model_root)
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
    return normalize_kda_state_dict(state_dict, config)
