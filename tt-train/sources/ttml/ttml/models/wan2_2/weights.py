# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load one Wan expert from a diffusers checkpoint into a WanTransformer3D."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import ttnn

import ttml
from ttml.modules import ColumnParallelLinear, RowParallelLinear

from .patch_embed import conv3d_weight_to_linear

# Anchored on start-of-string or a dot so a submodule state_dict maps as well as a full one.
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
    """Reshape a checkpoint tensor to its ttml parameter shape."""
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


def _shard_plan(model) -> dict:
    """param name -> (mapper, sharded_dim, tp_size). Absent means replicated.

    ColumnParallelLinear shards its weight on dim 2 but its bias on dim 3; RowParallelLinear
    leaves its bias replicated, since it is added after the all-reduce.
    """
    mesh = ttml.maybe_mesh()
    if mesh is None or mesh.num_devices() == 1:
        return {}

    plan: dict = {}
    for prefix, module in model.named_modules():
        if isinstance(module, ColumnParallelLinear):
            entries = [("weight", 2)] + ([("bias", 3)] if module.bias is not None else [])
        elif isinstance(module, RowParallelLinear):
            entries = [("weight", 3)]
        else:
            continue
        size = mesh.axis_size(module.axis_name)
        for leaf, tdim in entries:
            name = f"{prefix}.{leaf}" if prefix else leaf
            plan[name] = (mesh.axis_mapper(module.axis_name, tdim=tdim), tdim, size)

    # A prefix mismatch here would silently load sharded weights as replicated.
    unknown = sorted(set(plan) - {n for n, _ in model.named_parameters()})
    if unknown:
        raise RuntimeError(
            f"shard plan does not match named_parameters(), e.g. {unknown[:4]}; "
            f"sharded weights would be loaded as replicated"
        )
    return plan


def _global_shape(local_shape, tdim, tp_size) -> tuple:
    """Parameter.shape() is per-device, so undo the shard to get the checkpoint's shape."""
    shape = list(local_shape)
    if tdim is not None:
        shape[tdim] *= tp_size
    return tuple(shape)


def load_expert_from_safetensors(model, model_id: str, subfolder: str = "transformer", *, strict: bool = True) -> int:
    """Fill `model`'s parameters from one expert subfolder. Returns the count loaded."""
    params = dict(model.named_parameters())
    plan = _shard_plan(model)
    remaining = set(params)
    loaded, unexpected = 0, []

    for shard in _resolve_files(model_id, subfolder):
        for key, array in _read_shard(shard).items():
            name = to_ttml_name(key)
            target = params.get(name)
            if target is None:
                unexpected.append(key)
                continue
            mapper, tdim, tp_size = plan.get(name, (None, None, 1))
            expected = _global_shape(target.shape(), tdim, tp_size)
            value = to_ttml_array(name, array, expected)
            new_tensor = ttml.autograd.Tensor.from_numpy(value, ttnn.Layout.TILE, ttnn.bfloat16, mapper)
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

    sharded = len(plan)
    print(f"[wan] loaded {loaded} tensors from {subfolder} ({sharded} sharded, " f"{loaded - sharded} replicated)")
    return loaded
