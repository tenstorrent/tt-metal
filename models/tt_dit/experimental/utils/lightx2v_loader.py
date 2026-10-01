# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Loader for lightx2v Wan2.2 distilled weights.

The ``lightx2v/Wan2.2-Distill-Models`` HF repo ships flat ``.safetensors`` files
containing only the DiT weights (no ``transformer/``, ``vae/``, ``text_encoder/``
subfolders). The keys use the *original Wan* naming
(``blocks.X.self_attn.q.weight``, ``blocks.X.modulation``, ``head.head.weight``,
``time_embedding.0.weight``, ...), but tt_dit's ``WanTransformer3DModel``
consumes the *diffusers* layout (``blocks.X.attn1.to_q.weight``,
``blocks.X.scale_shift_table``, ``proj_out.weight``,
``condition_embedder.time_embedder.linear_1.weight``, ...).

:func:`load_lightx2v_state_dict` applies :func:`rename_lightx2v_to_diffusers`
to the loaded state dict using :func:`rename_substate` from the DiT library,
so the returned dict matches what ``WanTransformer3DModel._prepare_torch_state``
expects.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import torch
from loguru import logger
from safetensors.torch import load_file

from models.tt_dit.utils.substate import rename_substate

# Per-block prefix renames (lightx2v → diffusers), applied under each
# ``blocks.<i>.`` subtree. Uses rename_substate for prefix-based renaming.
_BLOCK_PREFIX_REMAP: list[tuple[str, str]] = [
    # self-attention
    ("self_attn.q", "attn1.to_q"),
    ("self_attn.k", "attn1.to_k"),
    ("self_attn.v", "attn1.to_v"),
    ("self_attn.o", "attn1.to_out.0"),
    ("self_attn.norm_q", "attn1.norm_q"),
    ("self_attn.norm_k", "attn1.norm_k"),
    # cross-attention
    ("cross_attn.q", "attn2.to_q"),
    ("cross_attn.k", "attn2.to_k"),
    ("cross_attn.v", "attn2.to_v"),
    ("cross_attn.o", "attn2.to_out.0"),
    ("cross_attn.norm_q", "attn2.norm_q"),
    ("cross_attn.norm_k", "attn2.norm_k"),
    # feedforward
    ("ffn.0", "ffn.net.0.proj"),
    ("ffn.2", "ffn.net.2"),
    # cross-attn input norm
    ("norm3", "norm2"),
]

# Single-key renames that don't fit the prefix pattern (leaf key → leaf key).
_BLOCK_KEY_REMAP: dict[str, str] = {
    "modulation": "scale_shift_table",
}

# Top-level prefix renames (lightx2v → diffusers).
_TOP_LEVEL_PREFIX_REMAP: list[tuple[str, str]] = [
    ("text_embedding.0", "condition_embedder.text_embedder.linear_1"),
    ("text_embedding.2", "condition_embedder.text_embedder.linear_2"),
    ("time_embedding.0", "condition_embedder.time_embedder.linear_1"),
    ("time_embedding.2", "condition_embedder.time_embedder.linear_2"),
    ("time_projection.1", "condition_embedder.time_proj"),
    ("head.head", "proj_out"),
]

_TOP_LEVEL_KEY_REMAP: dict[str, str] = {
    "head.modulation": "scale_shift_table",
}


def _detect_num_blocks(state: dict[str, torch.Tensor]) -> int:
    max_idx = -1
    for k in state:
        if k.startswith("blocks."):
            parts = k.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
    return max_idx + 1


def rename_lightx2v_to_diffusers(state: dict[str, torch.Tensor]) -> None:
    """Rename all keys in *state* from lightx2v/native-Wan to diffusers layout, in place.

    Uses :func:`rename_substate` from the DiT library for prefix-based renames.
    """
    num_blocks = _detect_num_blocks(state)

    for i in range(num_blocks):
        for src, dst in _BLOCK_PREFIX_REMAP:
            rename_substate(state, f"blocks.{i}.{src}", f"blocks.{i}.{dst}")
        for src, dst in _BLOCK_KEY_REMAP.items():
            old_key = f"blocks.{i}.{src}"
            if old_key in state:
                state[f"blocks.{i}.{dst}"] = state.pop(old_key)

    for src, dst in _TOP_LEVEL_PREFIX_REMAP:
        rename_substate(state, src, dst)

    for src, dst in _TOP_LEVEL_KEY_REMAP.items():
        if src in state:
            state[dst] = state.pop(src)


def wan_lightx2v_to_diffusers_key(key: str) -> str:
    """Rename one lightx2v key to its diffusers-canonical equivalent.

    Convenience wrapper kept for callers that need per-key conversion
    (e.g. LoRA key targeting). Internally builds a single-key dict and
    applies :func:`rename_lightx2v_to_diffusers`.
    """
    tmp = {key: torch.empty(0)}
    rename_lightx2v_to_diffusers(tmp)
    return next(iter(tmp))


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
        for snap in sorted(repo_cache.iterdir(), reverse=True):
            cand = snap / filename
            if cand.is_file():
                return cand

    if not allow_download:
        raise WeightsNotFoundError(
            f"lightx2v weights '{filename}' not found for repo '{repo_id}'.\n"
            f"Searched: {repo_cache}\n"
            f"To download from HuggingFace, re-run with TT_DIT_ALLOW_HF_DOWNLOAD=1 "
            f"or pass allow_download=True to the pipeline.\n"
            f"Alternatively, place the file under a snapshot directory in: {repo_cache}"
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
        key_remap: Flat-dict overrides applied after the lightx2v→diffusers
            rename. Useful for one-off renames in tests.

    Returns:
        State dict with diffusers-canonical keys ready for
        ``WanTransformer3DModel.load_torch_state_dict``.
    """
    path = _resolve_path(repo_id, filename, allow_download=allow_download, local_dir=local_dir)
    logger.info(f"Loading lightx2v state dict from '{path}'")
    state = load_file(str(path))

    rename_lightx2v_to_diffusers(state)

    if key_remap:
        for old_k, new_k in key_remap.items():
            if old_k in state:
                state[new_k] = state.pop(old_k)
    return state
