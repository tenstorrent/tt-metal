# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP64 gated sparse MLA (DSA attention) of a layer, over a given key selection: what a DSA device output is scored
against.

The golden holds each row's FP64 keys (``latent``, ``kpe``) and its FP64 indexer selection, but the FP64 attention
output only on sampled rows (attention is quadratic in the chain), so the scorer recomputes it here on the rows it
scores. Sizes (``Dims``) and FP64 weights come from the case; nothing here names a model.

``weights`` are the layer's FP64 tensors by their layer-local checkpoint names (``WEIGHTS``, the DeepSeek-style MLA
names). Per scored row, three FP64 attentions over the golden's keys, each against the reference (attention over the
golden's selection) (``score_rows``):

* the device output itself;
* ``sel``, the selection-only arm: attention over the device's own selection, i.e. what the device's selection costs;
* ``tail``, the near-tie baseline: attention over the golden's selection without its ``m`` lowest-ranked keys, ``m``
  the device's miss count on that row, i.e. what dropping the cut's tail costs;

and the largest attention weight of one key the device missed (``selection_error.missed_key_max``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from models.demos.common.prefill.long_context_accuracy.dsa.selection_error import missed_key_max
from models.demos.common.prefill.long_context_accuracy.windowed_errors import row_sums

ATTEND_BATCH = 8  # query rows per batched FP64 attention (bounds the [rows, k, lora] gather)

# The layer's tensors the attention reads, by layer-local checkpoint name.
WEIGHTS = (
    "input_layernorm.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.g_proj.weight",
    "self_attn.o_proj.weight",
)


@dataclass(frozen=True)
class Dims:
    """The attention's sizes: ``heads``, the query/key ``nope`` and ``rope`` head dims, the value head dim ``v``, the
    latent rank ``lora``, the selection size ``topk`` and the RMS norms' ``eps``."""

    heads: int
    nope: int
    rope: int
    v: int
    lora: int
    topk: int
    eps: float


def rms64(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def normed_input(read: torch.Tensor, w: Mapping[str, torch.Tensor], dims: Dims) -> torch.Tensor:
    """``input_layernorm`` of the layer's input ``read`` in FP64. The device is fed this rounded once to bf16; the
    reference attends from it unrounded."""
    return rms64(read.double(), w["input_layernorm.weight"], dims.eps)


def queries(x: torch.Tensor, w: Mapping[str, torch.Tensor], dims: Dims) -> dict[str, torch.Tensor]:
    """FP64 per row of the normed input ``x``: the absorbed query on the latent (``q_nope W_k``, ``[n, heads,
    lora]``), the rope query (``[n, heads, rope]``) and the output gate (``[n, heads * v]``)."""
    x = x.double()
    qr = rms64(x @ w["self_attn.q_a_proj.weight"].T, w["self_attn.q_a_layernorm.weight"], dims.eps)
    q = (qr @ w["self_attn.q_b_proj.weight"].T).reshape(-1, dims.heads, dims.nope + dims.rope)
    kvb = w["self_attn.kv_b_proj.weight"].reshape(dims.heads, dims.nope + dims.v, dims.lora)
    return {
        "q_latent": torch.einsum("nhd,hdl->nhl", q[..., : dims.nope], kvb[:, : dims.nope]),
        "q_pe": q[..., dims.nope :],
        "gate": torch.sigmoid(x @ w["self_attn.g_proj.weight"].T),
    }


def scores(q: Mapping[str, torch.Tensor], lo: int, latent, kpe, selection: torch.Tensor, dims: Dims):
    """Pre-softmax scores ``[b, heads, k]`` of query rows ``lo + [0, b)`` of ``q`` over their selected keys (-inf where
    an entry selects nothing), and the gathered latent ``[b, k, lora]``."""
    keys = selection.clamp_min(0)
    lat, pe = latent[keys].double(), kpe[keys].double()
    b = selection.shape[0]
    s = torch.bmm(q["q_latent"][lo : lo + b], lat.transpose(1, 2))
    s += torch.bmm(q["q_pe"][lo : lo + b], pe.transpose(1, 2))
    s *= (dims.nope + dims.rope) ** -0.5
    return s.masked_fill_((selection < 0).unsqueeze(1), -math.inf), lat


def heads(p: torch.Tensor, lat: torch.Tensor, w: Mapping[str, torch.Tensor], dims: Dims) -> torch.Tensor:
    """Per-head value outputs ``[b, heads * v]`` of softmax weights ``p`` over the gathered latent ``lat``."""
    kvb = w["self_attn.kv_b_proj.weight"].reshape(dims.heads, dims.nope + dims.v, dims.lora)[:, dims.nope :]
    mixed = torch.bmm(p, lat)  # [b, heads, lora]
    return torch.einsum("bhl,hvl->bhv", mixed, kvb).reshape(p.shape[0], -1)


def output(q: Mapping[str, torch.Tensor], per_head: torch.Tensor, w: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Gated ``o_proj`` of per-head outputs ``[n, heads * v]``: the layer's attention output ``[n, hidden]``."""
    return (per_head * q["gate"]) @ w["self_attn.o_proj.weight"].T


def tail_dropped(want: torch.Tensor, misses: torch.Tensor) -> torch.Tensor:
    """``want`` (best first, -1 past the valid keys) without each row's ``misses`` lowest-ranked valid keys."""
    valid = (want >= 0).sum(1, keepdim=True)
    rank = torch.arange(want.shape[1]).unsqueeze(0)
    return want.masked_fill(rank >= valid - misses.unsqueeze(1), -1)


@dataclass
class RowArms:
    """Per scored row: the row sums (``windowed_errors.row_sums``) of the device output, the selection arm and the tail
    baseline against the reference, and the largest missed-key weight."""

    rows: torch.Tensor
    device: torch.Tensor
    sel: torch.Tensor
    tail: torch.Tensor
    missed_key_max: torch.Tensor


def score_rows(
    rows: torch.Tensor,
    normed: torch.Tensor,
    device_out: torch.Tensor,
    device_sel: torch.Tensor,
    want_sel: torch.Tensor,
    want_missed: torch.Tensor,
    latent: torch.Tensor,
    kpe: torch.Tensor,
    w: Mapping[str, torch.Tensor],
    dims: Dims,
) -> tuple[RowArms, torch.Tensor]:
    """The arms of the scored ``rows`` (absolute), given per row their FP64 normed input, the device's output and
    selection, the golden's selection (``[n, k]``, -1 for no key) and which of its keys the device misses
    (``want_missed``, ``[n, k]`` bool: ``~topk_selection.topk_hits & valid``); ``latent`` / ``kpe`` hold the golden's
    keys of rows ``[0, > max(rows)]``. Also returns the FP64 reference output ``[rows, hidden]`` (for the recompute
    check)."""
    q = queries(normed, w, dims)
    ref_heads, sel_heads, tail_heads, key_max = [], [], [], []
    for lo in range(0, rows.shape[0], ATTEND_BATCH):
        want, got = want_sel[lo : lo + ATTEND_BATCH], device_sel[lo : lo + ATTEND_BATCH]
        missed = want_missed[lo : lo + ATTEND_BATCH]
        ref_scores, lat = scores(q, lo, latent, kpe, want, dims)
        p_ref = ref_scores.softmax(-1)
        sel_scores, sel_lat = scores(q, lo, latent, kpe, got, dims)
        p_sel = sel_scores.softmax(-1)
        # the tail baseline drops the reference selection's `missed` lowest-ranked keys: a mask on the same scores
        tail_scores = ref_scores.masked_fill((tail_dropped(want, missed.sum(1)) < 0).unsqueeze(1), -math.inf)
        ref_heads.append(heads(p_ref, lat, w, dims))
        sel_heads.append(heads(p_sel, sel_lat, w, dims))
        tail_heads.append(heads(tail_scores.softmax(-1), lat, w, dims))
        key_max.append(missed_key_max(p_ref, missed))
    ref = output(q, torch.cat(ref_heads), w)
    arms = RowArms(
        rows=rows,
        device=row_sums(ref, device_out.double()),
        sel=row_sums(ref, output(q, torch.cat(sel_heads), w)),
        tail=row_sums(ref, output(q, torch.cat(tail_heads), w)),
        missed_key_max=torch.cat(key_max),
    )
    return arms, ref
