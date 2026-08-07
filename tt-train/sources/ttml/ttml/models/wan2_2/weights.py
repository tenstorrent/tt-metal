# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load one Wan expert from a diffusers checkpoint into a WanTransformer3D.

Checkpoint tensors are bf16, which numpy cannot represent, so safetensors' torch reader
is used and the arrays are converted to fp32 on the way out. torch appears here only as
the weight reader, as in ttml/models/qwen3/weights.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import ttnn

import ttml

from .patch_embed import conv3d_weight_to_linear

# Checkpoint leaf -> ttml leaf. Inverse of the export renames in the example's lora_export.
# Anchored on start-of-string or a dot so a submodule state_dict ("ffn.net.2.weight") maps as
# well as a full one ("blocks.0.ffn.net.2.weight").
_LEAF_RENAMES = [
    (re.compile(r"(^|\.)patch_embedding\."), r"\1patch_embed."),
    (re.compile(r"(^|\.)to_out\.0\."), r"\1to_out."),
    (re.compile(r"(^|\.)ffn\.net\.0\.proj\."), r"\1ffn.ff1."),
    (re.compile(r"(^|\.)ffn\.net\.2\."), r"\1ffn.ff2."),
]


def to_ttml_name(key: str) -> str:
    for pattern, replacement in _LEAF_RENAMES:
        key = pattern.sub(replacement, key)
    return key


def to_ttml_array(name: str, array: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Reshape a checkpoint tensor to its ttml parameter shape.

    Linear weights arrive (out, in) and ttml wants (1, 1, out, in); biases and norm
    weights arrive (n,) and ttml wants (1, 1, 1, n); the patch embed arrives as a 5-D
    conv kernel; the modulation tables arrive (1, chunks, dim).
    """
    if name.startswith("patch_embed.") and array.ndim == 5:
        array = conv3d_weight_to_linear(array)
    while array.ndim < len(target_shape):
        array = array[None]
    if tuple(array.shape) != tuple(target_shape):
        raise ValueError(f"{name}: checkpoint gives {array.shape} after reshape, model wants {tuple(target_shape)}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _resolve_files(model_id: str, subfolder: str) -> list[Path]:
    """Local directory or HuggingFace repo -> the shard files for one expert."""
    local = Path(model_id) / subfolder
    if not local.is_dir():
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(repo_id=model_id, allow_patterns=[f"{subfolder}/*"])) / subfolder
    if not local.is_dir():
        raise FileNotFoundError(f"no {subfolder}/ under {model_id}")

    index = next(local.glob("*.index.json"), None)
    if index is not None:
        weight_map = json.loads(index.read_text())["weight_map"]
        return sorted({local / shard for shard in weight_map.values()})
    shards = sorted(local.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors in {local}")
    return shards


def _read_shard(path: Path) -> dict[str, np.ndarray]:
    from safetensors.torch import load_file

    return {name: tensor.float().numpy() for name, tensor in load_file(str(path)).items()}


def load_expert_from_safetensors(model, model_id: str, subfolder: str = "transformer", *, strict: bool = True) -> int:
    """Fill `model`'s parameters from one expert subfolder. Returns the count loaded."""
    params = dict(model.named_parameters())
    remaining = set(params)
    loaded, unexpected = 0, []

    for shard in _resolve_files(model_id, subfolder):
        for key, array in _read_shard(shard).items():
            name = to_ttml_name(key)
            target = params.get(name)
            if target is None:
                unexpected.append(key)
                continue
            value = to_ttml_array(name, array, tuple(target.shape()))
            new_tensor = ttml.autograd.Tensor.from_numpy(value, ttnn.Layout.TILE, ttnn.bfloat16)
            target.set_value(new_tensor.get_value())
            remaining.discard(name)
            loaded += 1

    if unexpected:
        print(f"[wan] {len(unexpected)} checkpoint tensors had no home, e.g. {unexpected[:4]}")
    if remaining:
        message = f"[wan] {len(remaining)} parameters were not initialised, e.g. {sorted(remaining)[:4]}"
        if strict:
            raise RuntimeError(message)
        print(message)

    print(f"[wan] loaded {loaded} tensors from {subfolder}")
    return loaded
