# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Indexed safetensor loading for Kimi K3's attention-residual query weights."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import NUM_LAYERS
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.weights import (
    CHECKPOINT_PREFIX,
    query_weight_names,
    validate_query_weights,
)
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3 import KimiK3Adapter


def resolve_attn_res_shards(
    checkpoint_dir: Path,
    num_layers: int = NUM_LAYERS,
    prefix: str = CHECKPOINT_PREFIX,
) -> tuple[Path, ...]:
    """Resolve the local shards holding every AttnRes weight, or say which are missing."""
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensor index: {index_path}")
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file).get("weight_map", {})

    names = query_weight_names(num_layers, prefix)
    missing = [name for name in names if name not in weight_map]
    if missing:
        raise ValueError(f"checkpoint index is missing {len(missing)} AttnRes weights, e.g. {missing[:3]}")

    shards = tuple(sorted({checkpoint_dir / weight_map[name] for name in names}))
    missing_shards = [path for path in shards if not path.is_file()]
    if missing_shards:
        raise FileNotFoundError(f"missing AttnRes checkpoint shard(s): {missing_shards}")
    return shards


def load_attn_res_state_dict(
    checkpoint_dir: Path,
    num_layers: int = NUM_LAYERS,
    prefix: str = CHECKPOINT_PREFIX,
) -> dict[str, torch.Tensor]:
    """Load every AttnRes weight and nothing else from an indexed checkpoint."""
    shards = resolve_attn_res_shards(checkpoint_dir, num_layers, prefix)
    wanted = set(query_weight_names(num_layers, prefix))
    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as shard_file:
            for name in wanted.intersection(shard_file.keys()):
                state_dict[name] = shard_file.get_tensor(name)
    validate_query_weights(state_dict, num_layers, prefix=prefix)
    return state_dict


def attn_res_tensor_cache_path(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int = 1,
    checkpoint_dir: Path | None = None,
) -> Path | None:
    """The cache root for one mesh placement, or `None` when nothing names one.

    `TT_KIMI_K3_PREFILL_TTNN_CACHE` is the whole model's cache root and wins outright, so
    tensorbins published with the model load with no checkpoint anywhere on the box. A
    checkpoint without that variable caches beside itself, which is what a first fetch gets.

    Placement is the directory rather than the stem because a query sharded four ways is
    a different tensorbin from the same query sharded eight ways, under the same name.
    """
    root = os.getenv(KimiK3Adapter.ttnn_cache_env)
    if root is None and checkpoint_dir is None:
        return None

    mesh_shape = tuple(mesh_device.shape)
    sequence_parallel_axis = 1 - tensor_parallel_axis
    layout = f"sp{mesh_shape[sequence_parallel_axis]}_tp{mesh_shape[tensor_parallel_axis]}"
    return (Path(root) if root else Path(checkpoint_dir) / "ttnn_cache") / layout
