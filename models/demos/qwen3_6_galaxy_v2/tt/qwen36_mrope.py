# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V3-MROPE-1: M-RoPE 3D cos/sin tables for qwen3.6 text decoder.

qwen3.6's `text_config.rope_parameters`:
  mrope_interleaved: True
  mrope_section:     [11, 11, 10]    (sums to 32 = partial_rotary_dim / 2)
  partial_rotary_factor: 0.25
  rope_theta:        10_000_000
  rope_type:         'default'

head_dim = 256, partial_rotary_dim = head_dim * partial_rotary_factor = 64.

The M-RoPE math (mirrors HF transformers:
`models/qwen3_vl/modeling_qwen3_vl.py:Qwen3VLTextRotaryEmbedding.forward`).
HF's `compute_default_rope_parameters` actually uses
`dim = head_dim * partial_rotary_factor = partial_rotary_dim` (qwen3.6: 64)
when partial_rotary_factor < 1.0 — NOT the full head_dim:

  inv_freq = rope_theta^(-arange(0, partial_rotary_dim, 2) / partial_rotary_dim)
             shape [partial_rotary_dim // 2 = 32 for qwen3.6]
  Given position_ids [3, B, S] (T/H/W axes):
    freqs = inv_freq @ position_ids  → [3, B, S, partial_rotary_dim // 2]
    freqs_interleaved = apply_interleaved_mrope(freqs, mrope_section)
                      → [B, S, partial_rotary_dim // 2]
  Where `apply_interleaved_mrope` starts from axis-T's freqs and OVERWRITES
  specific indices with axis-H's freqs (offset=1, stride=3, length=mrope_section[1]*3=33)
  and axis-W's freqs (offset=2, stride=3, length=mrope_section[2]*3=30). For
  qwen3.6 with mrope_section sum=32=partial_rotary_dim/2, the interleaving
  touches all 32 freq indices.

  emb = cat(freqs_interleaved, freqs_interleaved, dim=-1) → [B, S, partial_rotary_dim]
  cos = emb.cos()
  sin = emb.sin()

For TEXT-ONLY input, all 3 axes of position_ids are identical (a 1D ramp
broadcast to 3 dims). The interleaved freqs then equal the 1D freqs, so the
output is mathematically identical to standard 1D RoPE. This is the
backward-compatibility guarantee that lets us drop M-RoPE into the existing
qwen3.6 v2 text decoder without breaking text-only tests.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def build_mrope_inv_freq(rope_theta: float, partial_rotary_dim: int) -> torch.Tensor:
    """Compute inverse-frequency tensor for M-RoPE.

    HF's `compute_default_rope_parameters` uses `dim = head_dim * partial_rotary_factor`
    (i.e. partial_rotary_dim) when partial_rotary_factor < 1.0. For qwen3.6 with
    head_dim=256 and partial_rotary_factor=0.25, that's dim=64.

    Returns: `[partial_rotary_dim // 2]` float32.
    """
    return 1.0 / (
        rope_theta
        ** (torch.arange(0, partial_rotary_dim, 2, dtype=torch.int64).to(dtype=torch.float32) / partial_rotary_dim)
    )


def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: Sequence[int]) -> torch.Tensor:
    """Replicates HF's `Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope`.

    Args:
        freqs: `[3, B, S, partial_rotary_dim // 2]` — per-axis (T, H, W) freqs.
        mrope_section: 3-tuple; the head_dim-half / 3 cyclic split of axes.

    Returns:
        `[B, S, partial_rotary_dim // 2]` with the interleaved per-token freqs.
    """
    freqs_t = freqs[0].clone()  # start from axis T (covers all positions)
    for dim, offset in enumerate((1, 2), start=1):  # H @ offset 1, W @ offset 2
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def build_mrope_cos_sin(
    position_ids: torch.Tensor,
    *,
    rope_theta: float,
    partial_rotary_dim: int,
    mrope_section: Sequence[int],
    attention_scaling: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build M-RoPE cos/sin tables for 3D position_ids.

    HF computes cos/sin over PARTIAL head_dim (`head_dim * partial_rotary_factor`)
    when partial_rotary_factor < 1.0. For qwen3.6 that's 64 dims. The mrope_section
    interleaving touches the first `sum(mrope_section) = partial_rotary_dim // 2`
    indices.

    Args:
        position_ids: `[3, B, S]` (T, H, W axes). For text-only, all 3 axes
            should be identical to recover standard 1D RoPE.
        rope_theta: base period (qwen3.6: 10_000_000).
        partial_rotary_dim: rotary dim for partial RoPE (qwen3.6: 64).
        mrope_section: 3-tuple summing to `partial_rotary_dim // 2`
            (qwen3.6: [11, 11, 10] summing to 32).
        attention_scaling: per HF (default 1.0 for 'default' rope_type).
        dtype: output dtype.

    Returns:
        `(cos, sin)` each `[B, S, partial_rotary_dim]`.
    """
    assert (
        position_ids.ndim == 3 and position_ids.shape[0] == 3
    ), f"position_ids must be [3, B, S]; got {position_ids.shape}"
    assert (
        sum(mrope_section) == partial_rotary_dim // 2
    ), f"sum(mrope_section)={sum(mrope_section)} must equal partial_rotary_dim/2={partial_rotary_dim // 2}"

    inv_freq = build_mrope_inv_freq(rope_theta, partial_rotary_dim)  # [partial_rotary_dim // 2]

    # Per-axis freqs: [3, B, S, head_dim // 2]
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()  # [3, B, 1, S]
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)  # [3, B, S, head_dim/2]

    # Interleave T/H/W into a single per-token freq tensor (first sum(mrope_section) indices only).
    freqs_interleaved = apply_interleaved_mrope(freqs, mrope_section)  # [B, S, head_dim/2]

    # Double and convert to cos/sin
    emb = torch.cat((freqs_interleaved, freqs_interleaved), dim=-1)  # [B, S, head_dim]
    cos = (emb.cos() * attention_scaling).to(dtype=dtype)
    sin = (emb.sin() * attention_scaling).to(dtype=dtype)
    return cos, sin
