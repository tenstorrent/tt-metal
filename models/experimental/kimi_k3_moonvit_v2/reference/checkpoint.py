# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Vision weights from a Kimi-K3 HF checkpoint folder.

The published checkpoint stores the vision tower under `vision_tower.` and the projector under
`mm_projector.`, in the last two of its 96 shards. `load_vision_state_dict` reads
`model.safetensors.index.json` and opens only the shards that hold those keys, so a folder with
the index, `config.json` and those two shards is enough. The upstream model class is not used: it
imports the K3 text model and `transformers`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

CHECKPOINT_ENV = "KIMI_K3_HF_MODEL"
CHECKPOINT_PREFIXES = ("vision_tower.", "mm_projector.")
BLOCKS_PREFIX = "vision_tower.encoder.blocks."


def checkpoint_path() -> Path | None:
    """`$KIMI_K3_HF_MODEL` if it names a folder with a safetensors index."""
    value = os.environ.get(CHECKPOINT_ENV)
    if value and (Path(value) / "model.safetensors.index.json").is_file():
        return Path(value)
    return None


def drop_layers(state: dict[str, torch.Tensor], num_layers: int) -> dict[str, torch.Tensor]:
    """Keep blocks below `num_layers`, so a shallow config loads from a full checkpoint."""
    return {
        k: v
        for k, v in state.items()
        if not (k.startswith(BLOCKS_PREFIX) and int(k.removeprefix(BLOCKS_PREFIX).split(".")[0]) >= num_layers)
    }


def load_vision_state_dict(path: str | Path, num_layers: int) -> dict[str, torch.Tensor]:
    """State dict for `KimiK3VisionModel` (checkpoint keys as they are), bf16, blocks below `num_layers`."""
    from safetensors import safe_open

    path = Path(path)
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(CHECKPOINT_PREFIXES):
            by_shard.setdefault(shard, []).append(key)
    if not by_shard:
        raise KeyError(f"{path} has no {' or '.join(CHECKPOINT_PREFIXES)} weights")

    state = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(path / shard, framework="pt") as f:
            for key in keys:
                state[key] = f.get_tensor(key).to(torch.bfloat16)
    return drop_layers(state, num_layers)
