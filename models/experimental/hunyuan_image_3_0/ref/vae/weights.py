# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Safetensors weight loading helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor

MODEL_DIR = Path(
    os.environ.get(
        "HUNYUAN_MODEL_DIR",
        "/home/iguser/ign-sakthi/HunyuanImage-3.0/HunyuanImage-3",
    )
)


def load_tensors(model_dir: Path, keys: list[str]) -> dict[str, Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    shard_to_keys: dict[str, list[str]] = {}
    for key in keys:
        if key not in weight_map:
            raise KeyError(f"{key} not found in {index_path}")
        shard_to_keys.setdefault(weight_map[key], []).append(key)

    tensors: dict[str, Tensor] = {}
    for shard_file, shard_keys in shard_to_keys.items():
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing weight shard: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in shard_keys:
                tensors[key] = f.get_tensor(key)
    return tensors


def load_prefixed_state_dict(model_dir: Path, prefix: str, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    keys = [k for k in weight_map if k.startswith(prefix)]
    if not keys:
        raise RuntimeError(f"No keys with prefix {prefix!r} in {index_path}")

    tensors = load_tensors(model_dir, keys)
    strip = len(prefix)
    return {k[strip:]: v.to(dtype) for k, v in tensors.items()}
