# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The frames the SC1 caches are stored in, shared by prefill and decode.

Prefill (models/demos/deepseek_v3_d_p, the Hy4 adapter of tenstorrent/tt-metal#65935 and the KV
layout of #65886) writes the caches decode reads, so both sides must store the same numbers. The
checkpoint and ../reference are in the HF ``hy_v4`` frame; the device modules here work in the
frame the prefill stack already uses:

* RoPE pairs are INTERLEAVED, ``(2i, 2i + 1)``, where HF pairs ``(i, i + d / 2)`` (half-split).
  The cached ``k_rope`` and the queries' rope dims are in that order.
* An index key is ``[rope 64, interleaved | pass 64]``, where HF has ``[pass 64 | rope 64]``, and it
  is stored in the orthonormal Hadamard basis (so are the indexer's queries).

Nothing is computed differently: the output rows of the projections are reordered when the weights
are loaded (``attention_weights``, ``indexer_weights``), so ``q . k`` is the HF value, and the
Hadamard matrix is orthonormal, so the indexer's scores are too. ``*_to_device`` map a reference
tensor into the device frame, for tests and for anything that reads a cache on the host.

Pure torch, no device.
"""

from __future__ import annotations

import torch

from models.experimental.hy4_preview.reference.attention import rope_pairing


def interleave_perm(d: int, pairing: str | None = None) -> torch.Tensor:
    """``x[..., perm]`` turns a RoPE vector in the reference's order into the device's interleaved one at the same
    frequencies. The reference's pairing is the process's (reference/attention.py rope_pairing, HY4_ROPE_PAIRING):
    "half_split" (pairs ``(i, i + d/2)``): ``perm = [0, d/2, 1, d/2 + 1, ...]``; "interleaved" (pairs ``(2i, 2i + 1)``
    already): the identity. Every frame conversion of this module goes through this one function, so the switch moves
    the projections' row order, the cached ``k_rope``'s order, the index key's and the cos / sin tables together."""
    assert d % 2 == 0, f"RoPE width must be even, got {d}"
    if rope_pairing(pairing) == "interleaved":
        return torch.arange(d)
    h = d // 2
    return torch.stack([torch.arange(h), torch.arange(h) + h], dim=-1).reshape(d)


def index_key_perm(head_dim: int, rope_dim: int) -> torch.Tensor:
    """Index-key columns, HF ``[pass | rope, half-split]`` -> device ``[rope, interleaved | pass]``."""
    assert 0 < rope_dim <= head_dim, (rope_dim, head_dim)
    pass_dim = head_dim - rope_dim
    return torch.cat([pass_dim + interleave_perm(rope_dim), torch.arange(pass_dim)])


def _per_head_rows(w: torch.Tensor, head_dim: int, perm: torch.Tensor) -> torch.Tensor:
    """Apply ``perm`` to every head's block of output rows of a ``[heads * head_dim, in]`` weight."""
    assert w.shape[0] % head_dim == 0, (tuple(w.shape), head_dim)
    return w.reshape(w.shape[0] // head_dim, head_dim, -1)[:, perm].reshape(w.shape).contiguous()


# ---------------------------------------------------------------------------------------------
# Weights: checkpoint (HF) frame -> device frame
# ---------------------------------------------------------------------------------------------
def attention_weights(weights: dict, cfg) -> dict:
    """The attention weights with the rope rows of ``kv_a_proj_with_mqa`` (``[kv_pass | k_rope]``)
    and of every head of ``q_b_proj`` (``[q_nope | q_rope]``) reordered to interleaved pairs."""
    rp = interleave_perm(cfg.qk_rope_head_dim)
    out = dict(weights)
    kv = weights["kv_a_proj_with_mqa"]
    assert kv.shape[0] == cfg.kv_lora_rank + cfg.qk_rope_head_dim, tuple(kv.shape)
    out["kv_a_proj_with_mqa"] = torch.cat([kv[: cfg.kv_lora_rank], kv[cfg.kv_lora_rank :][rp]]).contiguous()
    rows = torch.cat([torch.arange(cfg.qk_nope_head_dim), cfg.qk_nope_head_dim + rp])
    out["q_b_proj"] = _per_head_rows(weights["q_b_proj"], cfg.qk_head_dim, rows)
    return out


def indexer_weights(weights: dict, cfg) -> dict:
    """The indexer weights with the key dims reordered (``index_key_perm``): the rows of ``wk``, of
    every head of ``wq_b``, and the key LayerNorm's weight and bias (a LayerNorm commutes with a
    permutation of its features when its affine parameters move with them)."""
    perm = index_key_perm(cfg.head_dim, cfg.qk_rope_head_dim)
    out = dict(weights)
    assert weights["wk.weight"].shape[0] == cfg.head_dim, tuple(weights["wk.weight"].shape)
    out["wk.weight"] = weights["wk.weight"][perm].contiguous()
    out["k_norm.weight"] = weights["k_norm.weight"][perm].contiguous()
    out["k_norm.bias"] = weights["k_norm.bias"][perm].contiguous()
    out["wq_b.weight"] = _per_head_rows(weights["wq_b.weight"], cfg.head_dim, perm)
    return out


# ---------------------------------------------------------------------------------------------
# Constants the device modules load
# ---------------------------------------------------------------------------------------------
def rope_tables(cos: torch.Tensor, sin: torch.Tensor):
    """HF cos / sin tables ``[..., d]`` (``cat(freqs, freqs)``) -> the interleaved tables the device
    modules take: every frequency on both dims of its pair."""
    perm = interleave_perm(cos.shape[-1])
    return cos[..., perm], sin[..., perm]


def rope_swap(d: int) -> torch.Tensor:
    """``[d, d]`` matrix P with ``(x @ P)[2i] = -x[2i + 1]`` and ``(x @ P)[2i + 1] = x[2i]``, so
    interleaved RoPE is ``x * cos + (x @ P) * sin``."""
    p = torch.zeros(d, d)
    even = torch.arange(0, d, 2)
    p[even + 1, even] = -1.0
    p[even, even + 1] = 1.0
    return p


def hadamard(dim: int) -> torch.Tensor:
    """The Sylvester-order orthonormal Hadamard matrix ``[dim, dim]`` the index keys are stored in,
    its entries rounded to bf16 as the prefill indexer loads it."""
    assert dim > 0 and dim & (dim - 1) == 0, f"Hadamard dimension must be a power of two, got {dim}"
    m = torch.ones(1, 1)
    while m.shape[0] < dim:
        m = torch.cat((torch.cat((m, m), dim=1), torch.cat((m, -m), dim=1)), dim=0)
    return (m * dim**-0.5).to(torch.bfloat16).float()


# ---------------------------------------------------------------------------------------------
# Reference tensors -> device frame (host)
# ---------------------------------------------------------------------------------------------
def rope_to_device(x: torch.Tensor) -> torch.Tensor:
    """A roped vector ``[..., rope]`` (the cached ``k_rope``), HF order -> interleaved."""
    return x[..., interleave_perm(x.shape[-1])]


def tail_to_device(x: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """A vector whose trailing ``rope_dim`` dims are roped (the latent ``k_pass | k_rope``, a query or
    a key ``nope | rope``): those dims reordered, the rest untouched."""
    split = x.shape[-1] - rope_dim
    return torch.cat([x[..., :split], rope_to_device(x[..., split:])], dim=-1)


def index_to_device(x: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """An index key or indexer query ``[..., head_dim]`` in the HF frame -> the stored frame:
    ``[rope, interleaved | pass]`` in the Hadamard basis."""
    head_dim = x.shape[-1]
    return x.float()[..., index_key_perm(head_dim, rope_dim)] @ hadamard(head_dim)


# ---------------------------------------------------------------------------------------------
# Device frame -> reference frame (host): the inverses, for reading a cache row back
# ---------------------------------------------------------------------------------------------
def rope_from_device(x: torch.Tensor) -> torch.Tensor:
    """The inverse of rope_to_device."""
    return x[..., torch.argsort(interleave_perm(x.shape[-1]))]


def tail_from_device(x: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """The inverse of tail_to_device: a stored latent ``k_pass | k_rope`` back in the HF order."""
    split = x.shape[-1] - rope_dim
    return torch.cat([x[..., :split], rope_from_device(x[..., split:])], dim=-1)


def index_from_device(x: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """The inverse of index_to_device: a stored index key (or indexer query) back in the HF frame.
    The Hadamard matrix is symmetric and, with its bf16-rounded entry, ``H @ H = gain * I``."""
    head_dim = x.shape[-1]
    h = hadamard(head_dim)
    gain = (h[0, 0] ** 2 * head_dim).item()
    return (x.float() @ h / gain)[..., torch.argsort(index_key_perm(head_dim, rope_dim))]
