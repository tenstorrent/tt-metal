# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight conversion helpers for MiniMax-M3.

MiniMax-M3 uses PARTIAL rotary (rotary_dim=64 of head_dim=128). The shared
tt_transformers ``convert_hf_qkv_to_meta_format`` reverse-permutes the *full*
head_dim into Meta RoPE (interleaved) layout, which is wrong when
rotary_dim < head_dim: only the rotary slice participates in RoPE and must be
interleaved; the pass-through tail must keep its natural order.
"""

import torch


def meta_partial_permute(tensor: torch.Tensor, n_heads: int, head_dim: int, rotary_dim: int) -> torch.Tensor:
    """Per-head Meta-RoPE swizzle for partial rotary.

    Interleaves the two halves (HF half-split -> Meta interleaved) of only the
    first ``rotary_dim`` dims of each head, leaving ``[rotary_dim:]`` untouched.
    Works for a 2D projection weight ``[n_heads*head_dim, X]`` and a 1D norm gain
    ``[n_heads*head_dim]``. With ``rotary_dim == head_dim`` this is identical to
    the shared ``reverse_permute`` (full rotary).
    """
    is_1d = tensor.dim() == 1
    x = 1 if is_1d else tensor.shape[1]
    t = tensor.reshape(n_heads, head_dim, x)
    rot = t[:, :rotary_dim, :]
    pas = t[:, rotary_dim:, :]
    rot = rot.reshape(n_heads, 2, rotary_dim // 2, x).transpose(1, 2).reshape(n_heads, rotary_dim, x)
    out = torch.cat([rot, pas], dim=1).reshape(n_heads * head_dim, x)
    return out.squeeze(-1) if is_1d else out


def convert_hf_qkv_to_meta_format_partial(loaded_weights: dict, head_dim: int, rotary_dim: int) -> dict:
    """Partial-rotary-aware version of convert_hf_qkv_to_meta_format.

    Permutes q/k projection weights AND the full-width q/k-norm gains so they
    line up with the rotary-only Meta layout. MiniMax-M3 has no q/k/v/o biases,
    so only the ``.weight`` keys are handled. v_proj / o_proj are left unchanged.
    """
    converted = {}
    for key, tensor in loaded_weights.items():
        if any(s in key for s in ("q_proj.weight", "k_proj.weight", "q_norm.weight", "k_norm.weight")):
            n_heads = tensor.shape[0] // head_dim
            converted[key] = meta_partial_permute(tensor, n_heads, head_dim, rotary_dim)
        else:
            converted[key] = tensor
    return converted
