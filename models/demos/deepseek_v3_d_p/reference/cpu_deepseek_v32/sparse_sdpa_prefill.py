# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Reference implementation (torch golden) of the DSA sparse-MLA prefill op.

Mirrors the interface of ``models/demos/deepseek_v32/tt/ops.py::sparse_mla``
(branch ``mvasilijevic/dsa_functional``) so it can be a drop-in golden / the
on-device op can be a drop-in replacement. Single op, single output tensor.

Difference from the host fallback in ``ops.py``: masking is **fully baked into
the indices** (FlashMLA sparse contract — "indices already encode it"). Any
slot that must not be attended carries the sentinel ``MASKED_INDEX``:

    index == 0xFFFFFFFF  ->  this slot scores -inf  (contributes 0 to softmax)

So there is **no position/causal math in the op** — ``start_pos`` is accepted
only for signature parity with ``sparse_mla`` and is ignored. The upstream
indexer is responsible for emitting the sentinel for causal/pad slots.

Absorbed MQA over the top-k selected latents (one latent KV head, shared across
all query heads):

    q       [1, H, S, 576]   absorbed queries (q_nope·wkv_b ++ q_pe)
    kvpe    [T, 576]         full latent prefix (512 latent ++ 64 rope), one head
    indices [1, 1, S, k]     uint32 global key positions; 0xFFFFFFFF = masked
    scale   scalar           softmax scale (with YaRN mscale)
    ->      [1, H, S, 512]   bf16-equivalent; latent-512 space (wkv_b value
                             absorption 512->128 happens AFTER this op)

Latent KV layout: ``kvpe[..., :512]`` is the latent kv (== V); ``kvpe[..., 512:]``
is k_pe (rope). K used for scoring is the full 576; V used for output is the
first 512 (a view).

Two equivalent forms:
* ``sparse_mla``        — the op (gather k selected slots). Validate the kernel
                          against this.
* ``sparse_mla_masked`` — dense-mask golden (equivalence oracle only).
"""

import torch

MASKED_INDEX = 0xFFFFFFFF  # sentinel: slot is masked out (scores -inf)

LATENT_DIM = 512  # kv_lora_rank — V is kvpe[..., :LATENT_DIM]
ROPE_DIM = 64  # qk_rope_head_dim
KV_DIM = LATENT_DIM + ROPE_DIM  # 576 — full K width


def _prep(q, kvpe, indices):
    """Normalize the sparse_mla-shaped inputs to [B,H,S,Dk] / [B,T,Dk] / [B,S,k]."""
    B, H, S, Dk = q.shape
    k = indices.shape[-1]
    idx = indices.reshape(B, S, k)  # tolerate [1,1,S,k] / [B,1,S,k] / [B,S,k]
    if kvpe.dim() == 2:  # [T, 576] shared across batch (the sparse_mla convention)
        kv = kvpe.unsqueeze(0).expand(B, kvpe.shape[0], Dk)
    else:  # [B, T, 576]
        kv = kvpe
    return B, H, S, Dk, k, idx, kv


def sparse_mla(
    q: torch.Tensor,  # [1, H, S, 576] absorbed queries (nope·wkv_b ++ rope)
    kvpe: torch.Tensor,  # [T, 576] full latent prefix (K=full 576, V=[:512])
    indices: torch.Tensor,  # [1, 1, S, k] uint32 global key positions; 0xFFFFFFFF = masked
    scale: float,
    start_pos: int = 0,  # accepted for parity with ops.py::sparse_mla; IGNORED (indices encode causality)
) -> torch.Tensor:  # [1, H, S, 512]
    """
    Absorbed MQA over the top-k selected latents only. Masking is baked into
    ``indices`` via ``MASKED_INDEX``; no causal/position math here.

    For each (query row s) gather the k latents named by ``indices[..., s, :]``
    (one set, shared across all heads), score against every head's query over
    the full 576 width, set ``-inf`` on sentinel slots, softmax over the k axis,
    and weighted-sum the gathered V views (first 512).
    """
    del start_pos  # causality is encoded in `indices` (sentinel), not positions
    B, H, S, Dk, k, idx, kv = _prep(q, kvpe, indices)
    T = kv.shape[1]

    masked = idx == MASKED_INDEX  # [B,S,k]
    # Clamp sentinels in-bounds so the gather is legal; the gathered data is
    # discarded by the -inf score, so the value is moot.
    idx_safe = torch.where(masked, torch.zeros_like(idx), idx).to(torch.int64)

    # gather selected KV (shared across heads): [B,T,Dk] -> [B,S,k,Dk]
    idx_g = idx_safe.view(B, S, k, 1).expand(B, S, k, Dk)
    sel = torch.gather(kv.unsqueeze(1).expand(B, S, T, Dk), 2, idx_g)  # [B,S,k,Dk]

    # scores over the full 576 width: <q[b,h,s,:], sel[b,s,j,:]>
    scores = torch.einsum("bhsd,bsjd->bhsj", q, sel) * scale  # [B,H,S,k]
    scores = scores.masked_fill(masked.view(B, 1, S, k), float("-inf"))

    probs = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)  # [B,H,S,k]
    out = torch.einsum("bhsj,bsjd->bhsd", probs, sel[..., :LATENT_DIM])  # [B,H,S,512]
    return out


def sparse_mla_masked(
    q: torch.Tensor,  # [1, H, S, 576]
    kvpe: torch.Tensor,  # [T, 576]
    indices: torch.Tensor,  # [1, 1, S, k]; 0xFFFFFFFF = masked
    scale: float,
    start_pos: int = 0,
) -> torch.Tensor:  # [1, H, S, 512]
    """Dense-mask golden (equivalence oracle). Softmax over the full T axis."""
    del start_pos
    B, H, S, Dk, k, idx, kv = _prep(q, kvpe, indices)
    T = kv.shape[1]

    masked = idx == MASKED_INDEX
    # Route sentinels to a throwaway column T (sliced off) so they never enable
    # a real position; valid entries keep their index.
    idx_safe = torch.where(masked, torch.full_like(idx, T), idx).to(torch.int64)

    scores = torch.einsum("bhsd,btd->bhst", q, kv) * scale  # [B,H,S,T]
    index_mask = torch.full((B, S, T + 1), float("-inf"), device=q.device)
    index_mask.scatter_(-1, idx_safe, 0.0)
    index_mask = index_mask[..., :T]  # drop throwaway col

    scores = scores + index_mask.unsqueeze(1)  # [B,1,S,T] over heads
    probs = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.einsum("bhst,btd->bhsd", probs, kv[..., :LATENT_DIM])  # [B,H,S,512]
    return out


def _selfcheck() -> None:
    """Prove the op == dense-mask golden on sentinel-padded, sparse_mla-shaped inputs."""
    torch.manual_seed(0)
    H, S = 8, 16
    T = S  # full prefill
    k = 8  # selected slots per row (< T => non-trivial)
    scale = KV_DIM**-0.5

    q = torch.randn(1, H, S, KV_DIM)
    kvpe = torch.randn(T, KV_DIM)  # [T,576] — sparse_mla convention

    # sentinel-padded indices [1,1,S,k]: random unique valid prefix per row, rest masked.
    indices = torch.full((1, 1, S, k), MASKED_INDEX, dtype=torch.int64)
    for s in range(S):
        n = int(torch.randint(1, k + 1, (1,)))
        indices[0, 0, s, :n] = torch.randperm(T)[:n]

    a = sparse_mla(q, kvpe, indices, scale)
    b = sparse_mla_masked(q, kvpe, indices, scale)
    assert a.shape == (1, H, S, LATENT_DIM), a.shape
    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
    print(f"OK: sparse_mla == dense-mask golden; out {tuple(a.shape)} (K=576, V=512, sentinel-masked)")


if __name__ == "__main__":
    _selfcheck()
