# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Weight loader for the MiniMax-M3 vision tower and projector.

Reads only the vision tensors (`vision_tower.*`, `multi_modal_projector.*`, `patch_merge_mlp.*`), through the text
reference's SafetensorsWeights: it opens only the shards that the checkpoint's model.safetensors.index.json lists for
them and dequantizes a quantized weight as the text model's loader does; no transformers needed. The checkpoint was
saved with transformers 4.52.4 names, which `_remap` turns into the transformers 5.12 MiniMaxM3VLModel names.
`build_reference` puts the weights into the Torch reference (reference/vision.py), the module tree that the TT
`from_torch` builders read, built with the vision config of the checkpoint's config.json.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import torch

from models.demos.minimax_m3.reference.model import SafetensorsWeights
from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionConfig, MiniMaxM3VisionEncoder
from models.demos.minimax_m3.utils.checkpoint import is_aux_key, plain_key

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


def _checkpoint_dir(snapshot_dir: str) -> str:
    """snapshot_dir, or ValueError for an empty one, which would name files in the current directory."""
    if not snapshot_dir:
        raise ValueError(f"snapshot_dir must be a MiniMax-M3 checkpoint directory, got {snapshot_dir!r}")
    return snapshot_dir


def load_vision_state_dict(snapshot_dir: str) -> Dict[str, torch.Tensor]:
    """Return {v5.12 key: fp32 tensor} for the vision tower and projector, remapped from the checkpoint. Raises
    ValueError if the checkpoint's index lists no vision tensor."""
    weights = SafetensorsWeights(_checkpoint_dir(snapshot_dir), prefix="")
    keys = [plain_key(key) for key in weights.weight_map if key.startswith(_WANT) and not is_aux_key(key)]
    if not keys:
        index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
        raise ValueError(f"{index_path} lists no tensor starting with {', '.join(_WANT)}")
    return {_remap(key): weights.get(key) for key in keys}


def build_reference(snapshot_dir: str) -> MiniMaxM3VisionEncoder:
    """Torch reference with the vision config (MiniMaxM3VisionConfig.from_hf_config of its config.json) and the vision
    weights of the checkpoint in snapshot_dir. config.json is read first, so a missing or malformed one fails before the
    weights are read."""
    with open(os.path.join(_checkpoint_dir(snapshot_dir), "config.json")) as f:
        config = MiniMaxM3VisionConfig.from_hf_config(json.load(f))
    return MiniMaxM3VisionEncoder(config).load_hf_state_dict(load_vision_state_dict(snapshot_dir)).eval()
