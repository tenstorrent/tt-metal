# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Rename PaddleOCR-VL's vision weights onto the qwen36 vision tower's keys.

The two checkpoints are dimensionally identical, so this reuses
``models/demos/blackhole/qwen36/tt/vision`` and only renames keys onto its
layout. Three tensors have no qwen36 counterpart and are returned separately
for the host seam to consume instead: the patch embedding, the position
table, and ``ln_post``.
"""

from __future__ import annotations

import re

import torch

# Keys the host seam consumes. Everything else under ``visual.`` is a block weight.
PATCH_EMBED_WEIGHT = "visual.vision_model.embeddings.patch_embedding._linear.weight"
PATCH_EMBED_BIAS = "visual.vision_model.embeddings.patch_embedding._linear.bias"
POS_EMBED = "visual.vision_model.embeddings.position_embedding.positional_embedding"
LN_POST_PREFIX = "visual.vision_model.ln_post"

_LAYER_RE = re.compile(r"^visual\.vision_model\.encoder\.layers\.(\d+)\.(.+)$")

# Suffix rename within one encoder layer. Order matters only for readability;
# the match is exact on the leading component.
_LAYER_RENAMES = {
    "attn.wq": "attention.wq",
    "attn.wk": "attention.wk",
    "attn.wv": "attention.wv",
    "attn.wo": "attention.wo",
    "ln_1": "norm1",
    "ln_2": "norm2",
    "mlp.c_fc": "feed_forward.linear_fc1",
    "mlp.c_proj": "feed_forward.linear_fc2",
}

_MERGER_RENAMES = {
    "projector.pre_norm": "visual.merger.norm",
    "projector.linear_1": "visual.merger.linear_fc1",
    "projector.linear_2": "visual.merger.linear_fc2",
}


class UnmappedVisionKeys(RuntimeError):
    """Raised when a vision weight has nowhere to go.

    Silently dropping a weight produces a model that loads cleanly and is subtly
    wrong, which is far more expensive to debug than an import-time failure.
    """


def split_host_tensors(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pull out the tensors the host seam owns (patch embed, pos table, ln_post)."""
    host = {}
    for k in (PATCH_EMBED_WEIGHT, PATCH_EMBED_BIAS, POS_EMBED):
        if k in state_dict:
            host[k] = state_dict[k]
    for suffix in ("weight", "bias"):
        k = f"{LN_POST_PREFIX}.{suffix}"
        if k in state_dict:
            host[k] = state_dict[k]
    return host


def _rename_layer_key(layer_num: str, rest: str) -> str | None:
    for src, dst in _LAYER_RENAMES.items():
        if rest == src or rest.startswith(src + "."):
            tail = rest[len(src) :]
            return f"visual.blocks.{layer_num}.{dst}{tail}"
    return None


def map_vision_state_dict(
    state_dict: dict[str, torch.Tensor], *, strict: bool = True
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split a converted PaddleOCR-VL state dict into (device weights, host weights).

    Returns the renamed vision/merger weights the qwen36 modules load, and the
    host-side tensors. Text weights (``layers.*``, ``tok_embeddings``, ``norm``,
    ``output``) are passed through untouched so the caller can hand the same
    dict to both the text and vision models.
    """
    host = split_host_tensors(state_dict)
    device: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []

    for k, v in state_dict.items():
        if k in host:
            continue

        if not k.startswith(("visual.", "projector.")):
            device[k] = v  # text-side weight, already in meta naming
            continue

        m = _LAYER_RE.match(k)
        if m:
            renamed = _rename_layer_key(m.group(1), m.group(2))
            if renamed is None:
                unmapped.append(k)
            else:
                device[renamed] = v
            continue

        for src, dst in _MERGER_RENAMES.items():
            if k.startswith(src + "."):
                device[dst + k[len(src) :]] = v
                break
        else:
            unmapped.append(k)

    if unmapped and strict:
        raise UnmappedVisionKeys(
            f"{len(unmapped)} vision weight(s) have no destination; the checkpoint layout "
            f"changed or a rename is missing. First few: {sorted(unmapped)[:8]}"
        )

    return device, host


def summarize(device: dict, host: dict) -> str:
    n_blocks = len({k.split(".")[2] for k in device if k.startswith("visual.blocks.")})
    n_merger = sum(1 for k in device if k.startswith("visual.merger."))
    n_text = sum(1 for k in device if not k.startswith("visual."))
    return (
        f"device: {len(device)} keys ({n_blocks} vision blocks, {n_merger} merger, {n_text} text) | "
        f"host: {len(host)} keys ({sorted(host)})"
    )
