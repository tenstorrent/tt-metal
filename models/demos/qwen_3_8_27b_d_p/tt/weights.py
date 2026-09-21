# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint resolution and the safetensors walk.

**Resolution order** (recipe section 4, step 0): ``PREFILL_HF_MODEL`` / ``HF_MODEL`` /
``QWEN35_HF_MODEL`` override anything; then the shared flat org dir
``/mnt/models/Qwen/Qwen3.8-27B``; then the shared read-only hub cache. Downloading is a last
resort and is not done implicitly — a missing checkpoint should be loud on minute one.

**No dequantization.** Qwen3.8-27B ships unquantized bf16: the safetensors index carries no scale
tensors and no packed blocks, so there is nothing to dequantize and no scale to fail loudly on.
The single dtype exit is the ``.to(torch.bfloat16)`` below.

**What is dropped.** The published checkpoint is a VL package. Keys are ``model.language_model.*``
(the text tower), ``model.visual.*`` (the vision tower) and ``mtp.*`` (the multi-token-prediction
head), plus a top-level ``lm_head.weight``. Only the text tower and ``lm_head`` are loaded; the
prefix is stripped so the keys match what the TT modules and the torch reference both expect
(``embed_tokens.weight``, ``layers.N.*``, ``norm.weight``, ``lm_head.weight``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

import torch
from loguru import logger
from tqdm import tqdm

TEXT_PREFIX = "model.language_model."
DROPPED_PREFIXES = ("model.visual.", "mtp.")

_SHARED_FLAT = Path("/mnt/models/Qwen/Qwen3.8-27B")
_SHARED_HUB = Path("/mnt/models/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots")


def resolve_checkpoint_path(required: bool = True) -> Optional[Path]:
    """The Qwen3.8-27B checkpoint directory, or ``None`` when absent and not required."""
    for var in ("PREFILL_HF_MODEL", "QWEN35_HF_MODEL", "HF_MODEL"):
        value = os.environ.get(var)
        if value and Path(value).is_dir():
            return Path(value)
    if (_SHARED_FLAT / "config.json").exists():
        return _SHARED_FLAT
    if _SHARED_HUB.is_dir():
        snapshots = sorted(p for p in _SHARED_HUB.iterdir() if (p / "config.json").exists())
        if snapshots:
            return snapshots[-1]
    if required:
        raise FileNotFoundError(
            "No Qwen3.8-27B checkpoint found. Set PREFILL_HF_MODEL / HF_MODEL, or stage it at "
            f"{_SHARED_FLAT} (needs an HF token: huggingface-cli login)."
        )
    return None


def _shards_for_layers(weight_map: dict[str, str], layers: Optional[Iterable[int]]) -> list[str]:
    """The subset of shards holding the requested layers (plus embed / norm / lm_head).

    A depth-reduced isolation run should not read all 52 GB. Reading fewer shards makes the run
    reduced in the recipe's sense, which the caller is responsible for labelling.
    """
    if layers is None:
        return sorted(set(weight_map.values()))
    wanted = set(layers)
    keep: set[str] = set()
    for key, shard in weight_map.items():
        if not key.startswith(TEXT_PREFIX):
            if not key.startswith(DROPPED_PREFIXES):
                keep.add(shard)  # top-level lm_head
            continue
        stripped = key[len(TEXT_PREFIX) :]
        if stripped.startswith("layers."):
            if int(stripped.split(".")[1]) in wanted:
                keep.add(shard)
        else:
            keep.add(shard)  # embed_tokens / norm
    return sorted(keep)


def load_text_backbone_state_dict(
    path: Path | str,
    *,
    layers: Optional[Iterable[int]] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Read the checkpoint and return the text tower's state dict with prefixes stripped."""
    from safetensors.torch import load_file

    path = Path(path)
    with open(path / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    shards = _shards_for_layers(weight_map, layers)
    if layers is not None:
        logger.warning(
            f"REDUCED load: {len(shards)} of {len(set(weight_map.values()))} shards for layers "
            f"{sorted(set(layers))} — any number produced from this is a reduced-run number"
        )

    state_dict: dict[str, torch.Tensor] = {}
    for shard in tqdm(shards, desc="Qwen3.8-27B text backbone (bf16 safetensors)"):
        for key, value in load_file(str(path / shard)).items():
            if key.startswith(DROPPED_PREFIXES):
                continue
            if key.startswith(TEXT_PREFIX):
                new_key = key[len(TEXT_PREFIX) :]
            elif key == "lm_head.weight":
                new_key = key
            else:
                continue
            if layers is not None and new_key.startswith("layers."):
                if int(new_key.split(".")[1]) not in set(layers):
                    continue
            state_dict[new_key] = value.to(dtype) if value.dtype != dtype else value
    logger.info(f"loaded {len(state_dict)} text-tower tensors from {path}")
    return state_dict


def assert_unquantized(path: Path | str) -> None:
    """Fail loudly if the checkpoint turns out to carry quantization metadata.

    The loader has no dequantization step because this checkpoint needs none. If a future
    checkpoint does, that must be a hard error here rather than weights silently read as if the
    packed blocks were values.
    """
    path = Path(path)
    with open(path / "config.json") as f:
        cfg = json.load(f)
    assert "quantization_config" not in cfg, (
        f"{path}/config.json carries a quantization_config; this loader has no dequantizer "
        f"(see the module docstring) and would read packed blocks as values"
    )
    with open(path / "model.safetensors.index.json") as f:
        keys = json.load(f)["weight_map"].keys()
    scales = [k for k in keys if k.endswith(("_scale", "_scale_inv", "scales", "weight_scale"))]
    assert not scales, f"{path} carries scale tensors ({scales[:3]}...) but the loader has no dequantizer"
