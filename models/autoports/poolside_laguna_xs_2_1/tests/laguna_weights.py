# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real / synthetic weight loading for poolside/Laguna-XS-2.1 decoder layers.

Provides:
  * ``load_layer_tensors``  – pull one decoder layer's tensors from the sharded
    safetensors checkpoint (only the shards holding that layer are opened).
  * ``to_hf_layer_state_dict`` – convert the raw per-expert checkpoint layout into
    the fused format that HF ``LagunaDecoderLayer.load_state_dict`` expects
    (``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj`` and the router bias
    remapped to ``mlp.gate.e_score_correction_bias``).
  * ``collect_layer_stats`` – per-tensor {shape,dtype,mean,std} for synthetic gen.
  * ``synth_layer_tensors`` – deterministic synthetic tensors matching real stats.

The raw per-expert tensors (``experts_gate/up/down`` stacked ``[E, ...]``) are what
the TTNN implementation consumes directly, so both the reference and the device
module are fed from the same source of truth.
"""
from __future__ import annotations

import json
import os

import torch

MODEL_ID = "poolside/Laguna-XS-2.1"


def _snapshot_dir():
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.py"])


def _index():
    d = _snapshot_dir()
    with open(os.path.join(d, "model.safetensors.index.json")) as f:
        return d, json.load(f)["weight_map"]


def _resolve_shard(snap_dir, shard_name):
    """Return an on-disk path to a shard, downloading it if absent."""
    p = os.path.join(snap_dir, shard_name)
    if os.path.exists(p):
        return p
    from huggingface_hub import hf_hub_download

    return hf_hub_download(MODEL_ID, shard_name)


def load_layer_tensors(layer_idx: int) -> dict:
    """Load all raw checkpoint tensors for ``model.layers.{layer_idx}`` as fp32 torch
    tensors, keyed with the ``model.layers.{idx}.`` prefix stripped."""
    from safetensors import safe_open

    snap_dir, wm = _index()
    prefix = f"model.layers.{layer_idx}."
    keys = [k for k in wm if k.startswith(prefix)]
    shards = sorted({wm[k] for k in keys})
    out = {}
    for shard in shards:
        path = _resolve_shard(snap_dir, shard)
        with safe_open(path, "pt") as f:
            present = set(f.keys())
            for k in keys:
                if k in present:
                    out[k[len(prefix) :]] = f.get_tensor(k).to(torch.float32)
    return out


def _stack_experts(raw: dict, config):
    """Build fused expert tensors from per-expert weights.

    HF ``LagunaExperts`` uses ``gate_up_proj[E, 2*I, H]`` (F.linear(x, W) -> chunk(2)
    => gate = concat[:I], up = concat[I:]) and ``down_proj[E, H, I]``."""
    E = config.num_experts
    gate = torch.stack([raw[f"mlp.experts.{i}.gate_proj.weight"] for i in range(E)])  # [E, I, H]
    up = torch.stack([raw[f"mlp.experts.{i}.up_proj.weight"] for i in range(E)])  # [E, I, H]
    down = torch.stack([raw[f"mlp.experts.{i}.down_proj.weight"] for i in range(E)])  # [E, H, I]
    gate_up = torch.cat([gate, up], dim=1)  # [E, 2I, H]
    return gate_up, down


def to_hf_layer_state_dict(raw: dict, config, layer_idx: int) -> dict:
    """Convert raw checkpoint tensors into the fused layout HF expects for one layer."""
    is_moe = (layer_idx not in config.mlp_only_layers) and (
        config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
    )
    sd = {}
    for k, v in raw.items():
        if k.startswith("mlp.experts.") and k.endswith(("gate_proj.weight", "up_proj.weight", "down_proj.weight")):
            continue  # folded below
        if k == "mlp.experts.e_score_correction_bias":
            sd["mlp.gate.e_score_correction_bias"] = v
            continue
        sd[k] = v
    if is_moe:
        gate_up, down = _stack_experts(raw, config)
        sd["mlp.experts.gate_up_proj"] = gate_up
        sd["mlp.experts.down_proj"] = down
    return sd


def collect_layer_stats(layer_idx: int) -> dict:
    raw = load_layer_tensors(layer_idx)
    stats = {}
    for k, v in raw.items():
        stats[k] = {
            "shape": list(v.shape),
            "dtype": "bfloat16",
            "mean": float(v.mean()),
            "std": float(v.std()),
        }
    return stats


def synth_layer_tensors(layer_idx: int, stats: dict, *, seed: int = 0) -> dict:
    """Deterministic synthetic tensors with the same shapes as the real checkpoint and
    per-tensor mean/std drawn from ``stats``."""
    g = torch.Generator().manual_seed(seed + layer_idx)
    out = {}
    for k, meta in stats.items():
        shape = meta["shape"]
        mean, std = meta["mean"], max(meta["std"], 1e-6)
        t = torch.empty(shape, dtype=torch.float32)
        t.normal_(0.0, 1.0, generator=g)
        out[k] = t * std + mean
    return out
