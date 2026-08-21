# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real Kimi-K3 layer weights in the layout ``TtPrefillBlock`` reads."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import load_kda_layer_state_dict

LAYER_PREFIX = "language_model.model.layers"


def load_hf_tensors(checkpoint_dir: Path, full_names: dict[str, str]) -> dict[str, torch.Tensor]:
    """Read ``{alias: checkpoint key}`` from the shards the index puts each key in."""
    index_path = Path(checkpoint_dir) / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]

    missing = sorted(name for name in full_names.values() if name not in weight_map)
    if missing:
        raise ValueError(f"{checkpoint_dir} index is missing: {missing}")

    by_shard: dict[str, list[tuple[str, str]]] = {}
    for alias, name in full_names.items():
        by_shard.setdefault(weight_map[name], []).append((alias, name))

    tensors: dict[str, torch.Tensor] = {}
    for shard, entries in by_shard.items():
        with safe_open(Path(checkpoint_dir) / shard, framework="pt", device="cpu") as shard_file:
            for alias, name in entries:
                tensors[alias] = shard_file.get_tensor(name)
    return tensors


def load_dense_block_state_dict(checkpoint_dir: Path, layer_idx: int) -> dict:
    """Assemble one DENSE Kimi-K3 layer for ``TtPrefillBlock``.

    Dense only (layer 0 under ``first_k_dense_replace=1``): every tensor a dense layer touches is
    stored unquantized, so this needs no MXFP4 dequantizer. A MoE layer's 896 routed experts ship
    as packed FP4 nibbles plus per-32 e8m0 scales and cannot be read this way.
    """
    if layer_idx >= KimiK3Config.NUM_DENSE_LAYERS:
        raise ValueError(f"layer {layer_idx} is a MoE layer; only layers < {KimiK3Config.NUM_DENSE_LAYERS} are dense")

    prefix = f"{LAYER_PREFIX}.{layer_idx}"
    tensors = load_hf_tensors(
        checkpoint_dir,
        {
            "attn_norm_weight": f"{prefix}.input_layernorm.weight",
            "ffn_norm_weight": f"{prefix}.post_attention_layernorm.weight",
            "gate_proj": f"{prefix}.mlp.gate_proj.weight",
            "up_proj": f"{prefix}.mlp.up_proj.weight",
            "down_proj": f"{prefix}.mlp.down_proj.weight",
        },
    )
    return {
        "attn_norm_weight": tensors["attn_norm_weight"],
        "ffn_norm_weight": tensors["ffn_norm_weight"],
        "kda_weights": load_kda_layer_state_dict(checkpoint_dir, layer_idx, kimi_k3_kda_config()),
        "ffn_weights": {name: tensors[name] for name in ("gate_proj", "up_proj", "down_proj")},
    }
