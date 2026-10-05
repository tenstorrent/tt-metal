# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Rename PaddleOCR-VL vision weights onto qwen36's keys; patch embedding and position table go to the host."""

from __future__ import annotations

import re

import torch

from models.tt_transformers.tt.load_checkpoints import reverse_permute

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
    # Tower-final LayerNorm; applied on device between the blocks and the merger.
    LN_POST_PREFIX: "visual.ln_post",
}


class UnmappedVisionKeys(RuntimeError):
    """A vision weight with no destination, raised rather than silently dropped."""


def split_host_tensors(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pull out the tensors the host seam owns (patch embedding, position table)."""
    return {k: state_dict[k] for k in (PATCH_EMBED_WEIGHT, PATCH_EMBED_BIAS, POS_EMBED) if k in state_dict}


def _rename_layer_key(layer_num: str, rest: str) -> str | None:
    for src, dst in _LAYER_RENAMES.items():
        if rest == src or rest.startswith(src + "."):
            tail = rest[len(src) :]
            return f"visual.blocks.{layer_num}.{dst}{tail}"
    return None


def _to_meta_rope_format(key: str, tensor: torch.Tensor, vision_head_dim: int) -> torch.Tensor:
    """Permute vision q/k into meta RoPE layout with the vision head dim; ModelArgs only converts text weights."""
    if not (".attention.wq." in key or ".attention.wk." in key):
        return tensor
    n_heads = tensor.shape[0] // vision_head_dim
    if tensor.dim() == 2:
        return reverse_permute(tensor, n_heads, tensor.shape[0], tensor.shape[1])
    return reverse_permute(tensor, n_heads, tensor.shape[0], 1).squeeze(-1)


def map_vision_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    vision_head_dim: int = 72,
    strict: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split into (device weights, host weights); text weights pass through. vision_head_dim is 72, not the text 128."""
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
                device[renamed] = _to_meta_rope_format(renamed, v, vision_head_dim)
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
