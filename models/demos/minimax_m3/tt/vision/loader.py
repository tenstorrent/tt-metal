# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Weight loader for the MiniMax-M3 vision tower and projector.

Reads only the vision tensors (`vision_tower.*`, `multi_modal_projector.*`, `patch_merge_mlp.*`), through the text
reference's SafetensorsWeights: it opens only the shards that the checkpoint's model.safetensors.index.json lists for
them and dequantizes a quantized weight as the text model's loader does; no transformers needed. The checkpoint was
saved with transformers 4.52.4 names, which `_remap` turns into the transformers 5.12 MiniMaxM3VLModel names that
M3VLVisionModel.from_state_dict takes. tests/test_vision_accuracy.py checks on the CPU that they are the names that
transformers' own rules give the checkpoint's tensors (tests/golden_hf_vision.py's vision_weights), by which its
goldens were made.
"""
from __future__ import annotations

import os
from typing import Dict

import torch

from models.demos.minimax_m3.reference.model import SafetensorsWeights
from models.demos.minimax_m3.utils.checkpoint import is_aux_key, plain_key

_WANT = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")
_MERGE_PREFIX = "patch_merge_mlp."
_TOWER_PREFIX = "vision_tower.vision_model."
# Name prefixes of the checkpoint's own (transformers 4.52.4) vision tensors: the ones _remap renames.
CHECKPOINT_PREFIXES = (_TOWER_PREFIX, _MERGE_PREFIX)


def _remap(key: str) -> str:
    """Checkpoint (transformers 4.52.4) key -> v5.12 model key layout."""
    if key.startswith(_MERGE_PREFIX):
        return "multi_modal_projector.merge_" + key[len(_MERGE_PREFIX) :]
    if key.startswith(_TOWER_PREFIX):
        rest = key[len(_TOWER_PREFIX) :]
        rest = rest.replace("embeddings.patch_embedding", "embeddings.proj")
        rest = rest.replace("encoder.layers.", "layers.")
        return "vision_tower." + rest
    return key  # multi_modal_projector.linear_* unchanged


def _checkpoint_dir(snapshot_dir: str) -> str:
    """snapshot_dir, or ValueError for an empty one, which would name files in the current directory."""
    if not snapshot_dir:
        raise ValueError(f"snapshot_dir must be a MiniMax-M3 checkpoint directory, got {snapshot_dir!r}")
    return snapshot_dir


def load_vision_state_dict(snapshot_dir: str, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    """Return {v5.12 key: tensor of dtype} for the vision tower and projector, remapped from the checkpoint: fp32 keeps
    every stored value, bf16 holds them as the chip does, in half the memory. Raises ValueError if the checkpoint's
    index lists no vision tensor."""
    weights = SafetensorsWeights(_checkpoint_dir(snapshot_dir), prefix="", dtype=dtype)
    keys = [plain_key(key) for key in weights.weight_map if key.startswith(_WANT) and not is_aux_key(key)]
    if not keys:
        index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
        raise ValueError(f"{index_path} lists no tensor starting with {', '.join(_WANT)}")
    return {_remap(key): weights.get(key) for key in keys}
