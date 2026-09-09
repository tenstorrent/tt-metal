# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The HF -> Meta q/k weight permutation, and why this bring-up needs it.

## The problem

There are two self-consistent RoPE conventions, and they pair different head-dim columns:

* **HF (half-split)** — `emb = cat(freqs, freqs)`, rotation pairs column `i` with `i + d/2`. This is
  what `transformers` does, what `reference/model.py` implements, and how this checkpoint's q/k
  weights are laid out.
* **Meta (interleaved)** — `cos = [c0, c0, c1, c1, ...]`, rotation pairs column `2i` with `2i + 1`.

The ttnn ops this package's read path is built on — `ttnn.experimental.rotary_embedding_llama` for
the one-shot path and `ttnn.experimental.deepseek_prefill.rotary_embedding_indexed` for the chunked
one — consume **Meta-format** tables, which is what `tt_transformers`' `RotarySetup` builds. Feeding
them HF-format tables produces a plausible-looking result that is simply wrong: measured PCC 0.75
against the reference, with nothing raised.

## The fix, and why it is free

Permute the **q_proj and k_proj weight ROWS** from HF to Meta head layout at load time. Nothing else
changes:

* Attention scores are `q · k` contracted over the head dim, and a dot product is invariant under a
  permutation applied to BOTH operands. So permuting q and k together leaves every score identical.
* `v_proj` is never rotated, so it keeps HF order — and since the attention output inherits v's
  column order, `o_proj` also needs no change.

That is why only two of the four projections are touched. `reverse_permute` is `tt_transformers`'
own HF->Meta transform, imported rather than re-derived so the two cannot drift.

The alternative — HF tables via `get_rot_mats_hf` + `ttnn.experimental.rotary_embedding_hf`, leaving
the weights alone — works for the one-shot path but has no counterpart for `rotary_embedding_indexed`,
which the chunked prefill of P2 depends on. Permuting the weights keeps both paths on the ops the
donor packages actually exercised.
"""

import torch
from models.tt_transformers.tt.load_checkpoints import reverse_permute


def hf_to_meta_qk(weight: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Permute a `[n_heads * head_dim, in_features]` q or k projection from HF to Meta head layout.

    Args:
        weight: HF `nn.Linear` weight, `[out, in]`.
        n_heads: the head count this projection produces — `num_attention_heads` for q,
            `num_key_value_heads` for k. Passing the wrong one silently mis-permutes.
    """
    assert weight.dim() == 2, f"expected a 2D [out, in] weight, got {tuple(weight.shape)}"
    out, in_features = weight.shape
    assert out % (n_heads * 2) == 0, f"out dim {out} is not 2*n_heads({n_heads}) divisible"
    return reverse_permute(weight, n_heads, out, in_features)


def hf_to_meta_head_dim(x: torch.Tensor) -> torch.Tensor:
    """Permute the LAST dim of a per-head tensor from HF half-split to Meta interleaved order.

    `[r0..r_{d/2-1}, i0..i_{d/2-1}]` -> `[r0, i0, r1, i1, ...]`. The activation-space counterpart of
    :func:`hf_to_meta_qk`, used by tests that need to compare a rotated tensor across the two
    conventions rather than to permute a weight.
    """
    d = x.shape[-1]
    assert d % 2 == 0
    reals, imags = x[..., : d // 2], x[..., d // 2 :]
    return torch.stack((reals, imags), dim=-1).flatten(start_dim=x.dim() - 1)
