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

:func:`load_lightx2v_state_dict` applies :func:`wan_lightx2v_to_diffusers_key`
to every key by default so the returned dict matches what
``WanTransformer3DModel._prepare_torch_state`` expects.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Mapping

import torch
from loguru import logger
from safetensors.torch import load_file

# Per-block suffix renames: applied to the part after ``blocks.<i>.``.
# Verified 1:1 (matching shapes) against
# Wan-AI/Wan2.2-I2V-A14B-Diffusers/transformer for all 40 blocks.
_BLOCK_SUFFIX_REMAP: dict[str, str] = {
    # self-attention
    "self_attn.q.weight": "attn1.to_q.weight",
    "self_attn.q.bias": "attn1.to_q.bias",
    "self_attn.k.weight": "attn1.to_k.weight",
    "self_attn.k.bias": "attn1.to_k.bias",
    "self_attn.v.weight": "attn1.to_v.weight",
    "self_attn.v.bias": "attn1.to_v.bias",
    "self_attn.o.weight": "attn1.to_out.0.weight",
    "self_attn.o.bias": "attn1.to_out.0.bias",
    "self_attn.norm_q.weight": "attn1.norm_q.weight",
    "self_attn.norm_k.weight": "attn1.norm_k.weight",
    # cross-attention
    "cross_attn.q.weight": "attn2.to_q.weight",
    "cross_attn.q.bias": "attn2.to_q.bias",
    "cross_attn.k.weight": "attn2.to_k.weight",
    "cross_attn.k.bias": "attn2.to_k.bias",
    "cross_attn.v.weight": "attn2.to_v.weight",
    "cross_attn.v.bias": "attn2.to_v.bias",
    "cross_attn.o.weight": "attn2.to_out.0.weight",
    "cross_attn.o.bias": "attn2.to_out.0.bias",
    "cross_attn.norm_q.weight": "attn2.norm_q.weight",
    "cross_attn.norm_k.weight": "attn2.norm_k.weight",
    # feedforward (linear-gelu_approx-linear)
    "ffn.0.weight": "ffn.net.0.proj.weight",
    "ffn.0.bias": "ffn.net.0.proj.bias",
    "ffn.2.weight": "ffn.net.2.weight",
    "ffn.2.bias": "ffn.net.2.bias",
    # cross-attn input norm
    "norm3.weight": "norm2.weight",
    "norm3.bias": "norm2.bias",
    # per-block 6-way modulation table
    "modulation": "scale_shift_table",
}

# Top-level renames. ``patch_embedding.{weight,bias}`` matches verbatim and is
# omitted here so it passes through unchanged.
_TOP_LEVEL_REMAP: dict[str, str] = {
    "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
    "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
    "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
    "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
    "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
    "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
    "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
    "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
    "time_projection.1.weight": "condition_embedder.time_proj.weight",
    "time_projection.1.bias": "condition_embedder.time_proj.bias",
    "head.head.weight": "proj_out.weight",
    "head.head.bias": "proj_out.bias",
    "head.modulation": "scale_shift_table",
}

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")


def wan_lightx2v_to_diffusers_key(key: str) -> str:
    """Rename one lightx2v key to its diffusers-canonical equivalent.

    Returns the key unchanged when no rename applies (e.g. ``patch_embedding.*``).
    """
    if (m := _BLOCK_RE.match(key)) is not None:
        idx, suffix = m.group(1), m.group(2)
        new_suffix = _BLOCK_SUFFIX_REMAP.get(suffix, suffix)
        return f"blocks.{idx}.{new_suffix}"
    return _TOP_LEVEL_REMAP.get(key, key)


# Optional flat-dict overrides applied AFTER wan_lightx2v_to_diffusers_key.
# Empty by default — populate or pass ``key_remap=`` to handle future variants
# without modifying the canonical Wan rename above.
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
        for snap in sorted(repo_cache.iterdir(), reverse=True):
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
    rename_fn: Callable[[str], str] | None = wan_lightx2v_to_diffusers_key,
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
        rename_fn: Per-key rename function applied first. Defaults to the
            Wan2.2 lightx2v→diffusers map. Pass ``None`` to skip.
        key_remap: Flat-dict overrides applied after ``rename_fn`` (and
            after ``KEY_REMAP``). Useful for one-off renames in tests.

    Returns:
        State dict with diffusers-canonical keys ready for
        ``WanTransformer3DModel.load_torch_state_dict``.
    """
    path = _resolve_path(repo_id, filename, allow_download=allow_download, local_dir=local_dir)
    logger.info(f"Loading lightx2v state dict from '{path}'")
    state = load_file(str(path))

    if rename_fn is not None:
        state = {rename_fn(k): v for k, v in state.items()}

    remap = dict(KEY_REMAP)
    if key_remap:
        remap.update(key_remap)
    if remap:
        state = {remap.get(k, k): v for k, v in state.items()}
    return state
