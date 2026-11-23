# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the GLM-5.4 reference ops, on hand-verifiable values."""

import math

import torch

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.sparse_sdpa_prefill import MASKED_INDEX, sparse_mla
from models.demos.deepseek_v3_d_p.reference.glm_5_4 import (
    damp_gate_reference,
    pass1_indices,
    two_bank_sparse_mla,
    window_indices,
)

S_ = MASKED_INDEX


def _gate(x_prev, h, w, b, gain=None):
    gain = torch.ones(x_prev.shape[-1]) if gain is None else gain
    return damp_gate_reference(
        x_prev, h, norm_weight=gain, proj_weight=torch.tensor(w), proj_bias=torch.tensor(b), eps=0.0
    )


def test_gate_scalar_bias_sets_beta():
    # Zero weight: beta = sigmoid(bias). bias 0 -> 1/2, ln 3 -> 3/4.
    x_prev, h = torch.ones(1, 1, 4), torch.full((1, 1, 4), 3.0)
    x, beta = _gate(x_prev, h, [[0.0] * 4], [0.0])
    assert torch.equal(beta, torch.full((1, 1, 1), 0.5))
    assert torch.equal(x, torch.full((1, 1, 4), 2.0))
    x, beta = _gate(x_prev, h, [[0.0] * 4], [math.log(3.0)])
    torch.testing.assert_close(x, torch.full((1, 1, 4), 2.5))


def test_gate_reads_pass_input_through_rmsnorm():
    # x_prev = [2,0,0,0]: rms 1, so RMSNorm(x_prev) = x_prev; logit = 2*1 - 2 = 0 -> beta 1/2.
    # Reading h = [0,0,0,4] instead would give logit -2.
    x_prev = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]])
    h = torch.tensor([[[0.0, 0.0, 0.0, 4.0]]])
    x, beta = _gate(x_prev, h, [[1.0, 0.0, 0.0, 0.0]], [-2.0])
    assert torch.equal(beta, torch.full((1, 1, 1), 0.5))
    assert torch.equal(x, torch.tensor([[[1.0, 0.0, 0.0, 2.0]]]))
    # The norm gain scales the normed input: gain 2 on channel 0 -> logit 2 -> beta sigmoid(2).
    _, beta = _gate(x_prev, h, [[1.0, 0.0, 0.0, 0.0]], [-2.0], gain=torch.tensor([2.0, 1.0, 1.0, 1.0]))
    torch.testing.assert_close(beta, torch.sigmoid(torch.full((1, 1, 1), 2.0)))


def test_gate_per_channel_width_from_weight():
    x_prev, h = torch.ones(1, 1, 4), torch.full((1, 1, 4), 5.0)
    x, beta = _gate(x_prev, h, [[0.0] * 4] * 4, [0.0, math.log(3.0), -math.log(3.0), 0.0])
    torch.testing.assert_close(beta, torch.tensor([[[0.5, 0.75, 0.25, 0.5]]]))
    torch.testing.assert_close(x, torch.tensor([[[3.0, 4.0, 2.0, 3.0]]]))


def test_gate_rejects_other_widths(expect_error):
    with expect_error(ValueError, "gate weight must be"):
        _gate(torch.ones(1, 1, 4), torch.ones(1, 1, 4), [[0.0] * 4] * 2, [0.0, 0.0])


def test_window_indices_cover_p_minus_w_plus_1_to_p(expect_error):
    assert window_indices(0, 4, 3).tolist() == [[[[S_, S_, 0], [S_, 0, 1], [0, 1, 2], [1, 2, 3]]]]
    # A chunk at positions 5-6 over a bank-B buffer whose row 0 is position 3 (a 2-row carry).
    assert window_indices(5, 2, 3, kv_start=3).tolist() == [[[[0, 1, 2], [1, 2, 3]]]]
    with expect_error(ValueError, "window needs position 3"):
        window_indices(5, 2, 3, kv_start=4)  # position 3 is needed but not in the buffer


def test_pass1_indices_mask_future_and_pad():
    # CPU indexer at q_start 0, end 3: every row gets 3 positions; row 0 may only see 0, row 1 only 0-1.
    topk = torch.tensor([[[2, 0, 1], [1, 2, 0], [2, 1, 0]]])
    assert pass1_indices(topk, 0, 4).tolist() == [[[[S_, 0, S_, S_], [1, S_, 0, S_], [2, 1, 0, S_]]]]
    # Same rows as a later chunk starting at 10: everything is causal.
    assert pass1_indices(topk, 10, 3).tolist() == [[[[2, 0, 1], [1, 2, 0], [2, 1, 0]]]]


def _bank_rows(rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_two_bank_single_softmax_and_scale():
    # latent 2 + rope 1; q = [2,0,0], scale 1/2 -> score = row[0]. A row [ln3,1,0] vs B row [0,5,0]:
    # weights 3/4 and 1/4 under one softmax -> V[1] = 3/4 * 1 + 1/4 * 5 = 2.
    q = torch.tensor([[[[2.0, 0.0, 0.0]]]])
    kvpe_a = _bank_rows([[math.log(3.0), 1.0, 0.0]])
    kvpe_b = _bank_rows([[0.0, 5.0, 0.0]])
    idx = torch.tensor([[[[0]]]])
    out = two_bank_sparse_mla(q, kvpe_a, idx, kvpe_b, idx, 0.5, latent_dim=2)
    torch.testing.assert_close(out, torch.tensor([[[[0.75 * math.log(3.0), 2.0]]]]))


def test_two_bank_keeps_overlapping_position_twice():
    # q = 0 -> uniform weights over the valid keys. Position 0 in bank A (V 1), positions 0 and 1 in bank B
    # (V 1 and 4): kept as three keys -> (1 + 1 + 4) / 3 = 2; de-duplicating position 0 would give 2.5.
    q = torch.zeros(1, 1, 1, 3)
    kvpe_a = _bank_rows([[1.0, 0.0, 0.0]])
    kvpe_b = _bank_rows([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    out = two_bank_sparse_mla(
        q, kvpe_a, torch.tensor([[[[0, S_]]]]), kvpe_b, torch.tensor([[[[0, 1]]]]), 1.0, latent_dim=2
    )
    torch.testing.assert_close(out, torch.tensor([[[[2.0, 0.0]]]]))


def _dense_two_bank(q, kvpe_a, idx_a, kvpe_b, idx_b, scale, latent_dim):
    """Oracle: dense scores over every row of both banks, {0, -inf} masks from the index sets, one softmax."""

    def mask(idx, t):
        m = torch.full((idx.shape[-2], t + 1), float("-inf"))
        safe = torch.where(idx[0, 0] == MASKED_INDEX, torch.full_like(idx[0, 0], t), idx[0, 0])
        return m.scatter_(-1, safe.long(), 0.0)[:, :t]

    probs = torch.cat(
        [
            torch.einsum("hsd,td->hst", q[0], kvpe_a) * scale + mask(idx_a, kvpe_a.shape[0]),
            torch.einsum("hsd,td->hst", q[0], kvpe_b) * scale + mask(idx_b, kvpe_b.shape[0]),
        ],
        dim=-1,
    ).softmax(-1)
    values = torch.cat([kvpe_a, kvpe_b])[:, :latent_dim]
    return torch.einsum("hst,tc->hsc", probs, values).unsqueeze(0)


def test_two_bank_matches_dense_oracle():
    g = torch.Generator().manual_seed(3)
    heads, seq, ta, d, latent, k = 4, 12, 12, 24, 16, 5
    q = torch.randn(1, heads, seq, d, generator=g)
    kvpe_a = torch.randn(ta, d, generator=g)
    kvpe_b = torch.randn(seq, d, generator=g)
    topk = torch.stack([torch.randperm(ta, generator=g)[:k] for _ in range(seq)]).unsqueeze(0)
    idx_a = pass1_indices(topk, 0, k + 2)
    idx_b = window_indices(0, seq, 4)
    out = two_bank_sparse_mla(q, kvpe_a, idx_a, kvpe_b, idx_b, d**-0.5, latent_dim=latent)
    ref = _dense_two_bank(q, kvpe_a, idx_a, kvpe_b, idx_b, d**-0.5, latent)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_two_bank_with_empty_window_is_glm53_sparse_mla():
    # Bank B fully masked: the op reduces to the GLM-5.3 single-bank op over bank A.
    g = torch.Generator().manual_seed(5)
    heads, seq, t, d, k = 2, 6, 6, 576, 3
    q = torch.randn(1, heads, seq, d, generator=g)
    kvpe_a = torch.randn(t, d, generator=g)
    idx_a = pass1_indices(torch.stack([torch.arange(k)] * seq).unsqueeze(0), 0, k)
    idx_b = torch.full((1, 1, seq, 2), MASKED_INDEX, dtype=torch.int64)
    out = two_bank_sparse_mla(q, kvpe_a, idx_a, torch.zeros(1, d), idx_b, d**-0.5, latent_dim=512)
    torch.testing.assert_close(out, sparse_mla(q, kvpe_a, idx_a, d**-0.5), rtol=1e-5, atol=1e-5)
