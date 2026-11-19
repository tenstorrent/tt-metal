# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import torch
from safetensors import safe_open


def load_weights(model_dir: Path, prefix: str) -> dict[str, torch.Tensor]:
    """Return every checkpoint tensor under `prefix`, keyed with the prefix stripped."""
    model_dir = Path(model_dir)
    with open(model_dir / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    by_shard: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        if name.startswith(prefix):
            by_shard.setdefault(shard, []).append(name)
    if not by_shard:
        raise KeyError(f"no checkpoint tensors under prefix {prefix!r}")

    out = {}
    for shard, names in by_shard.items():
        with safe_open(model_dir / shard, framework="pt") as f:
            for name in names:
                out[name[len(prefix) :]] = f.get_tensor(name)
    return out
