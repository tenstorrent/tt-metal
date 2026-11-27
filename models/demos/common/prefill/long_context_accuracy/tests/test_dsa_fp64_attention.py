# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FP64 sparse MLA (``dsa/fp64_attention.py``): hand-checked attention, the near-tie baseline's selection, and the
arms of scored rows against an unbatched oracle. CPU only, tiny dimensions."""

from __future__ import annotations

import math

import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.dsa import fp64_attention as fa
from models.demos.common.prefill.long_context_accuracy.topk_selection import topk_hits
from models.demos.common.prefill.long_context_accuracy.windowed_errors import SUM_EE

ONE = fa.Dims(heads=1, nope=1, rope=1, v=1, lora=1, topk=4, eps=1e-5)
F64 = torch.float64


def unit_weights() -> dict[str, torch.Tensor]:
    """W_k = W_v = 1 and o_proj = 1: the output is the gate times the softmax-weighted mean of the latents."""
    return {
        "self_attn.kv_b_proj.weight": torch.ones(2, 1, dtype=F64),
        "self_attn.o_proj.weight": torch.ones(1, 1, dtype=F64),
    }


def hand_query(score_on_kpe: float) -> dict[str, torch.Tensor]:
    """One query row: no latent score, ``score_on_kpe`` on the rope part, gate 0.5."""
    return {
        "q_latent": torch.zeros(1, 1, 1, dtype=F64),
        "q_pe": torch.full((1, 1, 1), score_on_kpe, dtype=F64),
        "gate": torch.full((1, 1), 0.5, dtype=F64),
    }


LATENT = torch.tensor([[2.0], [4.0], [10.0]], dtype=F64)


def attend(q, latent, kpe, selection, w, dims):
    """Test oracle, one query row at a time in the unbatched form: gated absorbed MLA of ``q`` (``fa.queries``) over
    each row's selected keys (negative entries select nothing)."""
    kvb = w["self_attn.kv_b_proj.weight"].reshape(dims.heads, dims.nope + dims.v, dims.lora)
    out = []
    for i, sel in enumerate(selection):
        keys = sel[sel >= 0]
        scores = (q["q_latent"][i] @ latent[keys].T + q["q_pe"][i] @ kpe[keys].T) * (dims.nope + dims.rope) ** -0.5
        mixed = scores.softmax(-1) @ latent[keys]  # [heads, lora]
        heads = torch.einsum("hl,hvl->hv", mixed, kvb[:, dims.nope :]).reshape(-1)
        out.append((heads * q["gate"][i]) @ w["self_attn.o_proj.weight"].T)
    return torch.stack(out)


def production(q, latent, kpe, selection, w, dims):
    """The batched path ``score_rows`` computes its reference with."""
    scores, lat = fa.scores(q, 0, latent, kpe, selection, dims)
    return fa.output(q, fa.heads(scores.softmax(-1), lat, w, dims), w)


@pytest.mark.parametrize("path", [attend, production], ids=["oracle", "production"])
def test_attention_by_hand(path) -> None:
    kpe = torch.tensor([[1.0], [0.0], [0.0]], dtype=F64)
    w = unit_weights()
    # Equal scores: 0.5 * mean(2, 4) = 1.5; a -1 entry selects nothing.
    assert path(hand_query(0.0), LATENT, kpe, torch.tensor([[0, 1, -1]]), w, ONE).item() == pytest.approx(1.5)
    assert path(hand_query(0.0), LATENT, kpe, torch.tensor([[0, 2]]), w, ONE).item() == pytest.approx(3.0)
    # Score sqrt(2) ln 3 on key 0 at scale 2^-0.5: weights 3/4, 1/4, so 0.5 * (0.75 * 2 + 0.25 * 4) = 1.25.
    q = hand_query(math.sqrt(2) * math.log(3))
    assert path(q, LATENT, kpe, torch.tensor([[0, 1]]), w, ONE).item() == pytest.approx(1.25)


def test_tail_dropped_removes_the_lowest_ranked_valid_keys() -> None:
    want = torch.tensor([[5, 3, 1, -1], [7, 6, 4, 2]])
    assert fa.tail_dropped(want, torch.tensor([1, 0])).tolist() == [[5, 3, -1, -1], [7, 6, 4, 2]]
    assert fa.tail_dropped(want, torch.tensor([3, 2])).tolist() == [[-1, -1, -1, -1], [7, 6, -1, -1]]


def small_case(seed: int = 0):
    """Random weights and keys of a small layer: 2 heads, 6 keys, top-3, hidden 8."""
    dims = fa.Dims(heads=2, nope=4, rope=2, v=3, lora=5, topk=3, eps=1e-5)
    g = torch.Generator().manual_seed(seed)
    hidden, q_lora = 8, 6
    w = {
        "input_layernorm.weight": torch.ones(hidden, dtype=F64),
        "self_attn.q_a_proj.weight": torch.randn(q_lora, hidden, generator=g, dtype=F64),
        "self_attn.q_a_layernorm.weight": torch.ones(q_lora, dtype=F64),
        "self_attn.q_b_proj.weight": torch.randn(dims.heads * (dims.nope + dims.rope), q_lora, generator=g, dtype=F64),
        "self_attn.kv_b_proj.weight": torch.randn(dims.heads * (dims.nope + dims.v), dims.lora, generator=g, dtype=F64),
        "self_attn.g_proj.weight": torch.randn(dims.heads * dims.v, hidden, generator=g, dtype=F64),
        "self_attn.o_proj.weight": torch.randn(hidden, dims.heads * dims.v, generator=g, dtype=F64),
    }
    assert set(w) == set(fa.WEIGHTS)
    latent = torch.randn(6, dims.lora, generator=g, dtype=F64)
    kpe = torch.randn(6, dims.rope, generator=g, dtype=F64)
    normed = fa.normed_input(torch.randn(2, hidden, generator=g), w, dims)
    return dims, w, latent, kpe, normed


def missed_of(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    return ~topk_hits(got, want) & (want >= 0)


def test_score_rows_arms() -> None:
    dims, w, latent, kpe, normed = small_case()
    rows = torch.tensor([4, 5])
    want = torch.tensor([[3, 1, 0], [5, 2, 4]])
    q = fa.queries(normed, w, dims)
    ref = attend(q, latent, kpe, want, w, dims)
    # Same selection, exact output: every arm is exact and nothing is missed.
    arms, got_ref = fa.score_rows(rows, normed, ref, want, want, missed_of(want, want), latent, kpe, w, dims)
    assert torch.allclose(got_ref, ref)
    for sums in (arms.device, arms.sel, arms.tail):
        assert sums[:, SUM_EE].abs().max() < 1e-24
    assert arms.missed_key_max.tolist() == [0.0, 0.0]
    # Row 4 misses key 0 (its lowest-ranked) for key 2: the selection arm is FP64 attention over [3, 1, 2], the tail
    # baseline over [3, 1]; row 5 is unchanged.
    got = torch.tensor([[3, 1, 2], [5, 2, 4]])
    arms, _ = fa.score_rows(rows, normed, ref, got, want, missed_of(got, want), latent, kpe, w, dims)
    sel = attend(q, latent, kpe, got, w, dims)
    tail = attend(q, latent, kpe, torch.tensor([[3, 1, -1], [5, 2, 4]]), w, dims)
    assert arms.sel[:, SUM_EE].tolist() == pytest.approx((sel - ref).square().sum(1).tolist())
    assert arms.tail[:, SUM_EE].tolist() == pytest.approx((tail - ref).square().sum(1).tolist())
    assert arms.sel[0, SUM_EE] > 0 and arms.missed_key_max[0] > 0 and arms.missed_key_max[1] == 0
