# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Loader for the Wan2.2-S2V-14B checkpoint.

The reference checkpoint at ``Wan-AI/Wan2.2-S2V-14B`` is not published as a
Diffusers wrapper — there's no ``transformer/`` subfolder with the standard
Diffusers config + per-module weight layout. Instead, the repo ships:

  * ``config.json`` — the ``WanModel_S2V`` constructor kwargs.
  * ``diffusion_pytorch_model.safetensors.index.json`` + 4 safetensors shards
    holding all 1260 weight keys in the reference repo's native naming
    convention (``blocks.0.self_attn.q.weight``, ``head.head.weight``, etc.).

This module provides:

  * :func:`load_s2v_config` — parse ``config.json`` into a plain dict.
  * :func:`load_s2v_state_dict` — merge the 4 shards into a single CPU
    ``state_dict``. Cheap (just a flat torch dict).

The weight-name translation from the reference's native scheme to tt_dit's
Diffusers-style module hierarchy is a **separate** ~200-400 line task tracked
under #20. This loader's output is the input to that translator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

try:
    from safetensors.torch import load_file as _load_safetensors_file
except ImportError:  # pragma: no cover - safetensors should be available with diffusers
    _load_safetensors_file = None


def find_s2v_snapshot(model_id: str = "Wan-AI/Wan2.2-S2V-14B") -> Path:
    """Locate the latest HF cache snapshot for the S2V model.

    Raises ``FileNotFoundError`` if the model isn't on disk. Download with:

        huggingface-cli download Wan-AI/Wan2.2-S2V-14B
    """
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not repo_dir.exists():
        raise FileNotFoundError(f"HF snapshot dir not found: {repo_dir}")
    snaps = sorted(repo_dir.iterdir())
    if not snaps:
        raise FileNotFoundError(f"No snapshots under {repo_dir}")
    return snaps[-1]


def load_s2v_config(snapshot_dir: Path | str) -> dict[str, Any]:
    """Parse the S2V model's ``config.json`` into a plain dict."""
    cfg_path = Path(snapshot_dir) / "config.json"
    with cfg_path.open() as f:
        return json.load(f)


def load_s2v_state_dict(snapshot_dir: Path | str) -> dict[str, torch.Tensor]:
    """Merge the 4 safetensors shards into a single CPU ``state_dict``.

    The resulting dict is keyed by the reference repo's native module names
    (e.g. ``blocks.0.self_attn.q.weight``, ``head.modulation``,
    ``casual_audio_encoder.encoder.conv1_local.conv.weight``,
    ``audio_injector.injector.0.norm_q.weight``).

    Translating these keys to tt_dit's module hierarchy is a separate step
    (see ``wan_s2v_weight_map.py`` — TODO, tracked under #20).
    """
    if _load_safetensors_file is None:
        raise ImportError("safetensors is required to load Wan2.2-S2V-14B weights")

    snapshot_dir = Path(snapshot_dir)
    index_path = snapshot_dir / "diffusion_pytorch_model.safetensors.index.json"
    with index_path.open() as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    merged: dict[str, torch.Tensor] = {}
    for shard, _keys in sorted(shard_to_keys.items()):
        shard_path = snapshot_dir / shard
        shard_dict = _load_safetensors_file(str(shard_path))
        # Only keep the keys the index says belong to this shard (the
        # safetensors files may contain extras for sharing/dedup).
        for k in _keys:
            merged[k] = shard_dict[k]

    if len(merged) != len(weight_map):
        missing = set(weight_map) - set(merged)
        raise RuntimeError(
            f"Loaded {len(merged)} keys but expected {len(weight_map)}; "
            f"missing {len(missing)} (first few: {sorted(missing)[:5]})"
        )

    return merged
