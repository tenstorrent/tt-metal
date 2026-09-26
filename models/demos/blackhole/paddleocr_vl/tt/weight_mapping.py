# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Rename PaddleOCR-VL's vision weights onto the qwen36 vision tower's keys.

The two checkpoints are dimensionally identical, so this reuses
``models/demos/blackhole/qwen36/tt/vision`` and only renames keys onto its
layout. Three tensors have no qwen36 counterpart and are returned separately
for the host seam to consume instead: the patch embedding, the position
table, and ``ln_post`` (kept on device rather than the host, unlike the other
two, since it feeds straight into the merger).
"""

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
    """Raised when a vision weight has nowhere to go.

    Silently dropping a weight produces a model that loads cleanly and is subtly
    wrong, which is far more expensive to debug than an import-time failure.
    """


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
    """Permute a vision q/k projection into the interleaved meta RoPE layout.

    ``ModelArgs.load_state_dict`` converts the *text* projections (with the text
    head dim) but passes vision weights through untouched -- see
    ``map_hf_to_meta_keys_vision_only``, which only renames. The reused
    ``VisionAttention`` applies ``rotary_embedding_llama``, which expects the
    meta interleaving, so the permute has to happen here and with the *vision*
    head dim of 72. This mirrors what qwen36 does when it converts its
    vision-only state dict.

    Only q and k are affected; v and the output projection carry no rotation.
    """
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
    """Split a converted PaddleOCR-VL state dict into (device weights, host weights).

    Returns the renamed vision/merger weights the qwen36 modules load, and the
    host-side tensors. Text weights (``layers.*``, ``tok_embeddings``, ``norm``,
    ``output``) are passed through untouched so the caller can hand the same
    dict to both the text and vision models.

    ``vision_head_dim`` is the tower's own head dim (1152/16), not the text
    decoder's 128; it only affects the q/k RoPE permute.
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


def summarize(device: dict, host: dict) -> str:
    n_blocks = len({k.split(".")[2] for k in device if k.startswith("visual.blocks.")})
    n_merger = sum(1 for k in device if k.startswith("visual.merger."))
    n_text = sum(1 for k in device if not k.startswith("visual."))
    return (
        f"device: {len(device)} keys ({n_blocks} vision blocks, {n_merger} merger, {n_text} text) | "
        f"host: {len(host)} keys ({sorted(host)})"
    )
