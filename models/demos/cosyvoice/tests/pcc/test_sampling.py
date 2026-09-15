# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""`nucleus_filter` against a literal transcription of upstream's loop.

The shipped implementation vectorises what CosyVoice writes as a sort plus a
Python walk. That is a rewrite of the one function whose output decides every
generated token, so it is checked against the thing it replaces rather than
against a description of it -- `_upstream_nucleus_filter` below is transcribed
from `cosyvoice.utils.common`, and the tests assert the two agree.

No device needed: this is all host arithmetic.
"""
from __future__ import annotations

import pytest
import torch

from models.demos.cosyvoice.tt.llm.sampling import is_repetitive, nucleus_filter, ras_sampling


def _upstream_nucleus_filter(probs: torch.Tensor, top_p: float = 0.8, top_k: int = 25):
    """Upstream's own form: full sort, Python accumulation, break on either bound."""
    sorted_value, sorted_idx = probs.sort(descending=True, stable=True)
    keep, cum = 0, 0.0
    for i in range(len(sorted_idx)):
        if cum < top_p and keep < top_k:
            cum += float(sorted_value[i])
            keep += 1
        else:
            break
    return sorted_value[:keep], sorted_idx[:keep]


@pytest.mark.parametrize("top_p", [0.5, 0.8, 0.95, 1.0])
@pytest.mark.parametrize("top_k", [1, 5, 25, 60])
def test_nucleus_filter_matches_upstream(top_p, top_k):
    """Same retained set, same order, across a spread of distribution shapes.

    The `scale` sweep matters more than the seed count: a peaked distribution
    reaches `top_p` in one or two tokens and exercises the `top_p` bound, a flat
    one runs into `top_k` instead, and only covering both proves the two bounds
    are not being confused for each other.
    """
    for seed in range(8):
        for scale in (0.5, 2.0, 8.0):
            torch.manual_seed(seed)
            probs = (torch.randn(4097) * scale).softmax(dim=0)
            want_v, want_i = _upstream_nucleus_filter(probs, top_p, top_k)
            got_v, got_i = nucleus_filter(probs, top_p, top_k)
            assert got_i.tolist() == want_i.tolist(), f"indices differ at scale={scale}, seed={seed}"
            assert torch.equal(got_v, want_v), f"values differ at scale={scale}, seed={seed}"


def test_nucleus_filter_keeps_the_token_that_crosses_the_threshold():
    """Retained mass is `>= top_p`, not `<=` -- the boundary CosyVoice picks.

    `[0.6, 0.3, 0.1]` with `top_p = 0.8`: the first is kept because 0 < 0.8, the
    second because 0.6 < 0.8 (taking the mass to 0.9, past the threshold), and the
    third is dropped because 0.9 is not. A `<=` reading keeps only the first.
    """
    probs = torch.tensor([0.6, 0.3, 0.1])
    value, idx = nucleus_filter(probs, top_p=0.8, top_k=25)
    assert idx.tolist() == [0, 1]
    assert float(value.sum()) > 0.8


def test_nucleus_filter_never_returns_empty():
    """A single dominant token must still be selectable.

    With `top_p` at or below the leading probability the loop would keep one
    element because it tests the mass *before* adding, which starts at zero. The
    vectorised form has to reproduce that rather than return an empty tensor for
    `multinomial` to reject.
    """
    probs = torch.tensor([0.99, 0.005, 0.005])
    value, idx = nucleus_filter(probs, top_p=0.0, top_k=25)
    assert len(idx) == 1 and idx[0] == 0
    assert len(value) == 1


def test_nucleus_filter_respects_top_k_on_a_flat_distribution():
    """A uniform distribution never reaches `top_p` early, so `top_k` binds."""
    probs = torch.full((4097,), 1.0 / 4097)
    _, idx = nucleus_filter(probs, top_p=0.8, top_k=25)
    assert len(idx) == 25


def test_is_repetitive_fires_on_a_single_repeat():
    """`win_size * tau_r` is 1.0 with the shipped defaults, and the test is `>=`."""
    assert is_repetitive([5, 1, 2, 3], top_ids=5, win_size=10, tau_r=0.1)
    assert not is_repetitive([1, 2, 3], top_ids=5, win_size=10, tau_r=0.1)
    # outside the window, so it does not count
    assert not is_repetitive([5] + list(range(20, 40)), top_ids=5, win_size=10, tau_r=0.1)


def test_ras_resamples_when_the_nucleus_pick_repeats():
    """The repetition branch must not return the token it just rejected.

    Driven with a distribution so peaked that the nucleus draw is deterministic,
    and a history containing that token, so the branch is guaranteed to fire.
    """
    scores = torch.full((64,), -20.0)
    scores[7] = 20.0
    scores[8] = 5.0
    torch.manual_seed(0)
    assert ras_sampling(scores.clone(), decoded_tokens=[7], win_size=10, tau_r=0.1) != 7
