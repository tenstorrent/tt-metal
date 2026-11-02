# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Weight loader for the MiniMax-M3 vision tower and projector.

Reads only the vision tensors (`vision_tower.*`, `multi_modal_projector.*`, `patch_merge_mlp.*`) with
`safetensors`, from the shards that the checkpoint's model.safetensors.index.json lists for them; no
transformers needed. The checkpoint was saved with transformers 4.52.4 names, which `_remap` turns into
the transformers 5.12 MiniMaxM3VLModel names. `build_reference` puts the weights into the Torch
reference (reference/vision.py), the module tree that the TT `from_torch` builders read, built with
the vision config of the checkpoint's config.json.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
from safetensors import safe_open

from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionConfig, MiniMaxM3VisionEncoder

_WANT = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")


def _remap(key: str) -> str:
    """Checkpoint (transformers 4.52.4) key -> v5.12 model key layout."""
    if key.startswith("patch_merge_mlp."):
        return "multi_modal_projector.merge_" + key[len("patch_merge_mlp.") :]
    if key.startswith("vision_tower.vision_model."):
        rest = key[len("vision_tower.vision_model.") :]
        rest = rest.replace("embeddings.patch_embedding", "embeddings.proj")
        rest = rest.replace("encoder.layers.", "layers.")
        return "vision_tower." + rest
    return key  # multi_modal_projector.linear_* unchanged


def _checkpoint_file(snapshot_dir: str, name: str) -> str:
    """snapshot_dir/name. An empty snapshot_dir raises instead of naming a file in the current directory."""
    if not snapshot_dir:
        raise ValueError(f"snapshot_dir must be a MiniMax-M3 checkpoint directory, got {snapshot_dir!r}")
    return os.path.join(snapshot_dir, name)


def vision_shards(snapshot_dir: str) -> List[str]:
    """The shard files that hold vision tensors, from the checkpoint's model.safetensors.index.json."""
    index_path = _checkpoint_file(snapshot_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted({shard for key, shard in weight_map.items() if key.startswith(_WANT)})
    if not shards:
        raise ValueError(f"{index_path} lists no tensor starting with {', '.join(_WANT)}")
    return shards


def load_vision_state_dict(snapshot_dir: str, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    """Return {v5.12 key: tensor} for the vision tower and projector, remapped from the checkpoint."""
    sd: Dict[str, torch.Tensor] = {}
    for shard in vision_shards(snapshot_dir):
        with safe_open(os.path.join(snapshot_dir, shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith(_WANT):
                    sd[_remap(k)] = f.get_tensor(k).to(dtype)
    return sd


def build_reference(snapshot_dir: str) -> MiniMaxM3VisionEncoder:
    """Torch reference with the vision config (MiniMaxM3VisionConfig.from_hf_config of its config.json) and the vision
    weights of the checkpoint in snapshot_dir. config.json is read first, so a missing or malformed one fails before the
    weights are read."""
    with open(_checkpoint_file(snapshot_dir, "config.json")) as f:
        config = MiniMaxM3VisionConfig.from_hf_config(json.load(f))
    return MiniMaxM3VisionEncoder(config).load_hf_state_dict(load_vision_state_dict(snapshot_dir)).eval()
