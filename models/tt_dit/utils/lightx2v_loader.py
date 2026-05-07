# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Loader for lightx2v Wan2.2 distilled weights.

The ``lightx2v/Wan2.2-Distill-Models`` HF repo ships flat ``.safetensors`` files
containing only the DiT weights (no ``transformer/``, ``vae/``, ``text_encoder/``
subfolders). This module reads those files and returns a state dict that the
existing :class:`WanTransformer3DModel._prepare_torch_state` hook can consume.

If the safetensors keys diverge from the diffusers-canonical layout, populate
``KEY_REMAP`` and the loader will rename keys before returning.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import torch
from loguru import logger
from safetensors.torch import load_file

# Renames applied to lightx2v safetensors keys before they reach
# WanTransformer3DModel._prepare_torch_state. Empty by default; populate if a
# strict-load against the diffusers TorchWanTransformer3DModel reveals diffs.
KEY_REMAP: dict[str, str] = {}


class WeightsNotFoundError(FileNotFoundError):
    """Raised when a lightx2v weight file is not on disk and downloads are disabled."""


def _hf_cache_root() -> Path:
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")


def _resolve_path(
    repo_id: str,
    filename: str,
    *,
    allow_download: bool,
    local_dir: str | None,
) -> Path:
    if local_dir is not None:
        candidate = Path(local_dir) / filename
        if candidate.is_file():
            return candidate

    repo_cache = _hf_cache_root() / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if repo_cache.is_dir():
        for snap in sorted(repo_cache.iterdir()):
            cand = snap / filename
            if cand.is_file():
                return cand

    if not allow_download:
        raise WeightsNotFoundError(
            f"lightx2v weights '{filename}' not found for repo '{repo_id}'.\n"
            f"To download from HuggingFace, re-run with TT_DIT_ALLOW_HF_DOWNLOAD=1 "
            f"or pass allow_download=True to the pipeline.\n"
            f"Alternatively, place the file at: "
            f"{local_dir or _hf_cache_root() / 'hub'}/{filename}"
        )

    from huggingface_hub import hf_hub_download

    logger.info(f"Downloading lightx2v weights from HuggingFace: {repo_id}/{filename}")
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def load_lightx2v_state_dict(
    repo_id: str,
    filename: str,
    *,
    allow_download: bool = False,
    local_dir: str | None = None,
    key_remap: Mapping[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Load a lightx2v safetensors file and return a torch state dict.

    Args:
        repo_id: HuggingFace repo, e.g. ``"lightx2v/Wan2.2-Distill-Models"``.
        filename: File within the repo, e.g.
            ``"wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors"``.
        allow_download: When ``False`` (default), the file must already exist
            in ``local_dir`` or the HF cache. When ``True``, missing files are
            fetched via ``huggingface_hub.hf_hub_download``.
        local_dir: Optional directory to check before the HF cache.
        key_remap: Per-call key renames. Merged with module-level ``KEY_REMAP``.

    Returns:
        State dict with diffusers-canonical keys ready for
        ``WanTransformer3DModel.load_torch_state_dict``.
    """
    path = _resolve_path(repo_id, filename, allow_download=allow_download, local_dir=local_dir)
    logger.info(f"Loading lightx2v state dict from '{path}'")
    state = load_file(str(path))

    remap = dict(KEY_REMAP)
    if key_remap:
        remap.update(key_remap)
    if remap:
        state = {remap.get(k, k): v for k, v in state.items()}
    return state
