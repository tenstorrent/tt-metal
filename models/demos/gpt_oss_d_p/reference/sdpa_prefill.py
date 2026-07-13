# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Reference implementation (torch golden) of the GPT-OSS prefill SDPA — the
correctness oracle for ``ttnn.transformer.scaled_dot_product_attention`` as
called in ``tt/attention/prefill.py``.

This mirrors ``models/demos/minimax_m3/reference/sparse_gqa_prefill.py`` in
style: pure-torch, no ttnn, no HF — drop it in, run it, get shapes and numbers.

WHAT MAKES THIS OP NON-TRIVIAL vs STANDARD CAUSAL SDPA
-------------------------------------------------------
1. **GQA** (grouped-query attention): Hkv (8) < Hq (64). K and V are shared
   across groups of 8 query heads. K/V are NOT repeated before the call;
   the reference expands them on-the-fly (GQA group = Hq/Hkv = 8).

2. **Sliding-window local attention** (half the layers): Query at position i
   attends only to positions in [max(0, i-W+1), i]. W=128 in GPT-OSS 120B.
   ``sliding_window=None`` means full causal (the "full_attention" layers).

3. **Attention sinks**: A learned per-head additive bias (shape [Hq]) added to
   the raw (unscaled) QK logits before scaling. Allows each head to have a
   content-independent baseline score.  In the TT kernel, sinks are passed in
   "pre-divided" form (``sinks / scale``) because the kernel multiplies by scale
   internally; the reference adds the original sinks to the scaled scores so the
   math is identical to HF.

ARCHITECTURE VERIFIED (GPT-OSS 120B, checked against config.json)
------------------------------------------------------------------
  hidden_size          = 2880
  num_attention_heads  = 64    (Hq)
  num_key_value_heads  = 8     (Hkv)  ->  GQA group = 64/8 = 8
  head_dim             = 64    (D)    ->  scale = 64**-0.5 = 0.125
  sliding_window       = 128   (W)    (only "sliding_attention" layers; None otherwise)
  layer_types alternates: sliding_attention, full_attention, ...

REAL SHAPES (single chip, TP=1, batch=1):
  q        [1, 64,  S, 64]  bf16   (B=1, Hq=64, head_dim=64)
  k, v     [1,  8,  S, 64]  bf16   (B=1, Hkv=8)
  sinks    [1, 64,  1,  1]  float  (per-head bias, broadcast over B and S)
  scale               0.125
  sliding_window      128 or None
  ->  out  [1, 64,  S, 64]  bf16

EQUIVALENT FORMS (both are verified against each other below in __main__)
-------------------------------------------------------------------------
  ``causal_sdpa``       — manual gather/mask implementation. Validate the
                          kernel against this.
  ``causal_sdpa_torch`` — uses torch.nn.functional.scaled_dot_product_attention
                          with an explicit attention_mask. Equivalence oracle only.
"""

import torch
import torch.nn.functional as F

# Real GPT-OSS 120B defaults (single chip, TP=1)
HEAD_DIM = 64
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS  # 8
SCALE = HEAD_DIM**-0.5  # 0.125
SLIDING_WINDOW = 128  # None for full_attention layers


def _build_causal_mask(S: int, sliding_window: int | None, device=None) -> torch.Tensor:
    """Build additive causal mask [S, S] with -inf for masked positions.

    Standard causal: positions j > i are -inf.
    Sliding window: additionally mask positions j < i - W + 1.
    """
    i = torch.arange(S, device=device).unsqueeze(1)  # [S, 1]
    j = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
    causal = j > i  # future tokens are masked
    if sliding_window is not None:
        out_of_window = j < (i - sliding_window + 1)  # tokens too far back
        masked = causal | out_of_window
    else:
        masked = causal
    mask = torch.zeros(S, S, device=device)
    mask[masked] = float("-inf")
    return mask  # [S, S]


def causal_sdpa(
    q: torch.Tensor,  # [B, Hq, S, D]
    k: torch.Tensor,  # [B, Hkv, S, D]
    v: torch.Tensor,  # [B, Hkv, S, D]
    sinks: torch.Tensor | None = None,  # [1, Hq, 1, 1] or None
    scale: float = SCALE,
    sliding_window: int | None = SLIDING_WINDOW,
) -> torch.Tensor:  # [B, Hq, S, D]
    """
    GQA causal SDPA with optional sliding window and per-head attention sinks.

    Sinks are additive per-head biases on the SCALED logits (after multiplying by
    ``scale``), matching HF GPT-OSS behavior:
        scores = (q @ k^T) * scale + sinks
    """
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    assert Hq % Hkv == 0, f"Hq ({Hq}) must be divisible by Hkv ({Hkv})"
    group = Hq // Hkv

    # Expand KV heads to match Q heads for GQA
    k_exp = k.repeat_interleave(group, dim=1)  # [B, Hq, S, D]
    v_exp = v.repeat_interleave(group, dim=1)

    # Scaled dot products: [B, Hq, S, S]
    scores = torch.einsum("bhsd,bhtd->bhst", q, k_exp) * scale

    # Additive per-head attention sinks
    if sinks is not None:
        scores = scores + sinks  # broadcast [1, Hq, 1, 1] -> [B, Hq, S, S]

    # Causal (+ optional sliding-window) mask
    mask = _build_causal_mask(S, sliding_window, device=q.device)  # [S, S]
    scores = scores + mask  # broadcast over B, Hq

    probs = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.einsum("bhst,bhtd->bhsd", probs, v_exp)  # [B, Hq, S, D]
    return out


def causal_sdpa_torch(
    q: torch.Tensor,  # [B, Hq, S, D]
    k: torch.Tensor,  # [B, Hkv, S, D]
    v: torch.Tensor,  # [B, Hkv, S, D]
    sinks: torch.Tensor | None = None,  # [1, Hq, 1, 1] or None
    scale: float = SCALE,
    sliding_window: int | None = SLIDING_WINDOW,
) -> torch.Tensor:  # [B, Hq, S, D]
    """Dense-mask golden using torch.nn.functional.scaled_dot_product_attention.

    Passes an explicit additive attention_mask to SDPA. This is the equivalence
    oracle; validate ``causal_sdpa`` against this.
    """
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    group = Hq // Hkv

    k_exp = k.repeat_interleave(group, dim=1)
    v_exp = v.repeat_interleave(group, dim=1)

    mask = _build_causal_mask(S, sliding_window, device=q.device)  # [S, S]

    if sinks is not None:
        # torch SDPA does not accept a per-head bias natively; bake sinks into
        # the mask by expanding to [1, Hq, S, S] and adding the sink offset.
        attn_mask = mask.unsqueeze(0).unsqueeze(0) + sinks  # [1, Hq, S, S]
        attn_mask = attn_mask.expand(B, Hq, S, S)
    else:
        attn_mask = mask  # [S, S], broadcast by SDPA

    return F.scaled_dot_product_attention(
        q,
        k_exp,
        v_exp,
        attn_mask=attn_mask,
        scale=scale,
    )


def make_sdpa_inputs(
    S: int,
    B: int = 1,
    Hq: int = NUM_Q_HEADS,
    Hkv: int = NUM_KV_HEADS,
    D: int = HEAD_DIM,
    with_sinks: bool = True,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build random (q, k, v, sinks) matching GPT-OSS prefill producer contract."""
    g = torch.Generator()
    g.manual_seed(seed)
    q = torch.randn(B, Hq, S, D, generator=g, dtype=dtype)
    k = torch.randn(B, Hkv, S, D, generator=g, dtype=dtype)
    v = torch.randn(B, Hkv, S, D, generator=g, dtype=dtype)
    sinks = torch.randn(1, Hq, 1, 1, generator=g, dtype=dtype) if with_sinks else None
    return q, k, v, sinks


if __name__ == "__main__":
    torch.manual_seed(0)

    for S, W, label in [
        (64, SLIDING_WINDOW, "sliding S=64"),
        (256, SLIDING_WINDOW, "sliding S=256"),
        (256, None, "full S=256"),
    ]:
        q, k, v, sinks = make_sdpa_inputs(S)

        out_manual = causal_sdpa(q, k, v, sinks=sinks, sliding_window=W)
        out_torch = causal_sdpa_torch(q, k, v, sinks=sinks, sliding_window=W)

        max_diff = (out_manual - out_torch).abs().max().item()
        print(f"[{label}] max_diff={max_diff:.2e}  shape={tuple(out_manual.shape)}")
        assert max_diff < 1e-4, f"causal_sdpa vs causal_sdpa_torch mismatch: {max_diff}"

    print("all checks passed")
