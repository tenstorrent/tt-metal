# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Loads the MiniMax-M3 vision weights from the checkpoint, without transformers.

Opens only the shards that hold vision tensors and dequantizes quantized weights as the text model's loader does.
The checkpoint uses transformers 4.52.4 names; _remap renames them to the transformers 5.12 names that
M3VLVisionModel.from_state_dict takes.
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
    """Rename a checkpoint key (transformers 4.52.4) to the transformers 5.12 model's name."""
    if key.startswith(_MERGE_PREFIX):
        return "multi_modal_projector.merge_" + key[len(_MERGE_PREFIX) :]
    if key.startswith(_TOWER_PREFIX):
        rest = key[len(_TOWER_PREFIX) :]
        rest = rest.replace("embeddings.patch_embedding", "embeddings.proj")
        rest = rest.replace("encoder.layers.", "layers.")
        return "vision_tower." + rest
    return key  # multi_modal_projector.linear_* unchanged


def _checkpoint_dir(snapshot_dir: str) -> str:
    """Return snapshot_dir, or raise ValueError for an empty one, which would read the current directory."""
    if not snapshot_dir:
        raise ValueError(f"snapshot_dir must be a MiniMax-M3 checkpoint directory, got {snapshot_dir!r}")
    return snapshot_dir


def load_vision_state_dict(snapshot_dir: str, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    """Read the vision tower and projector weights of the checkpoint in snapshot_dir.

    Returns {transformers 5.12 name: tensor of dtype}. float32 keeps every stored value; bfloat16 is what the chip
    holds, in half the memory. Raises ValueError for an empty snapshot_dir or an index that lists no vision tensor.
    """
    weights = SafetensorsWeights(_checkpoint_dir(snapshot_dir), prefix="", dtype=dtype)
    keys = [plain_key(key) for key in weights.weight_map if key.startswith(_WANT) and not is_aux_key(key)]
    if not keys:
        index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
        raise ValueError(f"{index_path} lists no tensor starting with {', '.join(_WANT)}")
    return {_remap(key): weights.get(key) for key in keys}
