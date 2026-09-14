# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The one place the HF and Meta RoPE column orders meet.

Two equivalent ways to write the same rotation, differing only in how a head's 128 columns are
ordered:

* **HF half-split** — ``[x0 .. x63 | x64 .. x127]``, pair ``(j, j+64)``, rotated by ``rotate_half``
  against cos/sin of the form ``[c0..c63, c0..c63]``. This is what ``transformers`` does, what the
  reference in ``reference/model.py`` does, and what the golden trace stores (``k_layout:
  hf_half_split``).
* **Meta interleaved** — ``[x0, x64, x1, x65, ...]``, pair ``(2j, 2j+1)``, which is what
  ``ttnn.experimental.rotary_embedding_llama`` implements and therefore what the device holds.

The device is put into the Meta order once, at weight-load time, by permuting the *rows* of q_proj
and k_proj (:func:`meta_permute_proj`). Nothing downstream re-permutes: Q and K stay Meta-ordered
through attention (the scores are a dot product over the head dim, so a shared permutation of Q and
K is invariant), and the KV cache therefore stores Meta-ordered K. V never rotates and is never
permuted. The only remaining seam is the golden comparison, which uses :func:`hf_to_meta_perm`.
"""

from __future__ import annotations

import torch


def meta_permute_proj(weight: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Re-order the rows of a ``[n_heads*head_dim, in]`` q/k projection into Meta interleaved order.

    Per head, interleaves the two halves: output row ``2j`` takes input row ``j`` and output row
    ``2j+1`` takes input row ``j + head_dim/2``. Applied to q_proj and k_proj only — v_proj and
    o_proj do not participate in RoPE and must be left alone.
    """
    assert weight.shape[0] % head_dim == 0, f"rows {weight.shape[0]} not a multiple of head_dim {head_dim}"
    n_heads = weight.shape[0] // head_dim
    t = weight.reshape(n_heads, head_dim, -1)
    t = t.reshape(n_heads, 2, head_dim // 2, -1).transpose(1, 2).reshape(n_heads, head_dim, -1)
    return t.reshape(n_heads * head_dim, -1)


def hf_to_meta_perm(head_dim: int) -> torch.Tensor:
    """Column index map taking an HF half-split head vector to the Meta interleaved one.

    ``meta[m] = hf[perm[m]]`` with ``perm[m] = (head_dim/2) * (m % 2) + m // 2`` — the inverse view
    of :func:`meta_permute_proj`, expressed as a gather so it can be applied to a golden tensor
    (``g_k[..., perm]``) without touching any weights.
    """
    half = head_dim // 2
    return torch.tensor([half * (m % 2) + (m // 2) for m in range(head_dim)], dtype=torch.long)


def meta_to_hf_perm(head_dim: int) -> torch.Tensor:
    """Inverse of :func:`hf_to_meta_perm` — for reporting a device tensor in HF order."""
    return torch.argsort(hf_to_meta_perm(head_dim))


def meta_cos_sin(cos_hf: torch.Tensor, sin_hf: torch.Tensor) -> tuple:
    """Convert HF half-split ``[.., head_dim]`` cos/sin to the Meta interleaved duplication.

    HF holds ``[c0..c63, c0..c63]``; Meta wants ``[c0, c0, c1, c1, ...]``. Only the first half
    carries distinct values, so this is a stack-and-flatten of that half.
    """
    half = cos_hf.shape[-1] // 2
    c = torch.stack([cos_hf[..., :half], cos_hf[..., :half]], dim=-1).flatten(-2)
    s = torch.stack([sin_hf[..., :half], sin_hf[..., :half]], dim=-1).flatten(-2)
    return c, s
