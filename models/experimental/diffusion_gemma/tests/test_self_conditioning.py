# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the self-conditioning gated MLP reference (#47461/#47463)."""

import pytest
import torch

from models.experimental.diffusion_gemma.reference.self_conditioning import SelfConditioning


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _embedding(vocab, hidden, seed=0):
    return torch.randn(vocab, hidden, generator=_gen(seed))


def test_output_shape_matches_hidden():
    batch, length, vocab, hidden = 2, 8, 32, 16
    mod = SelfConditioning(hidden)
    out = mod(torch.randn(batch, length, vocab, generator=_gen(1)), _embedding(vocab, hidden))
    assert out.shape == (batch, length, hidden)


def test_encoder_pass_is_zeroed():
    batch, length, vocab, hidden = 2, 8, 32, 16
    mod = SelfConditioning(hidden)
    out = mod(torch.randn(batch, length, vocab, generator=_gen(2)), _embedding(vocab, hidden), enabled=False)
    assert out.shape == (batch, length, hidden)
    assert torch.all(out == 0)  # zeroed on encoder passes (prefill / commit)


def test_soft_embedding_onehot_recovers_token_row():
    vocab, hidden = 20, 12
    mod = SelfConditioning(hidden)
    emb = _embedding(vocab, hidden, seed=3)
    # near-one-hot logits at token 7 -> soft embedding ~= emb[7]
    logits = torch.full((1, 1, vocab), -1e4)
    logits[..., 7] = 1e4
    soft = mod.soft_embedding(logits, emb)
    assert torch.allclose(soft[0, 0], emb[7], atol=1e-4)


def test_soft_embedding_is_convex_combination():
    vocab, hidden = 6, 4
    mod = SelfConditioning(hidden)
    emb = _embedding(vocab, hidden, seed=4)
    soft = mod.soft_embedding(torch.randn(1, 5, vocab, generator=_gen(5)), emb)
    # every soft embedding lies within the bounding box of the embedding rows
    assert torch.all(soft <= emb.max(dim=0).values + 1e-5)
    assert torch.all(soft >= emb.min(dim=0).values - 1e-5)


def test_activation_silu_supported_and_unknown_rejected():
    SelfConditioning(8, activation="silu")  # ok
    with pytest.raises(ValueError):
        SelfConditioning(8, activation="relu6")._act(torch.zeros(1))


def test_deterministic_under_fixed_weights():
    batch, length, vocab, hidden = 1, 4, 10, 8
    torch.manual_seed(0)
    mod = SelfConditioning(hidden)
    emb = _embedding(vocab, hidden, seed=6)
    logits = torch.randn(batch, length, vocab, generator=_gen(7))
    a = mod(logits, emb)
    b = mod(logits.clone(), emb.clone())
    assert torch.equal(a, b)
