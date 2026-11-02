# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the host-side decode strategy (_sample_token in tt/ttnn_xtts_model.py).

_sample_token transcribes coqui's sampling: mel head -> repetition penalty -> temperature -> top-k
-> top-p -> one multinomial draw from the request's own RNG. It picks WHAT the model says, and no
other test in the suite reaches it — the PCC gates all compare latents, which the strategy never
sees, and the traced-decode gates compare argmax, which skips it entirely.

The penalty is vectorised (one gather/scatter over the `seen` set) because a per-token loop is
quadratic per utterance. `_reference_sample` is that per-token form, written plainly, so the
equivalence test holds the optimisation to the shape it claims to be equivalent to.

Logits are driven directly instead of through a real head: with mh_w all zeros the head reduces to
its bias, so mh_b IS the logit vector, and each stage can be given an input that forces a known
outcome.

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_sampling.py
"""
import math

import pytest
import torch

from models.experimental.xtts_v2.tt.ttnn_xtts_model import (
    REPETITION_PENALTY,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    _sample_token,
)

VOCAB = 1026  # mel head width: 1024 codes + START + STOP
DRAWS = 50  # one drawn index often coincides by chance; a run of them pins the distribution


def _head(logits):
    """(latent, mh_w, mh_b) that make _sample_token see exactly `logits`."""
    return torch.zeros(1, 1, 1024), torch.zeros(VOCAB, 1024), logits.clone()


def _draw(logits, seen=(), seed=0):
    latent, w, b = _head(logits)
    return _sample_token(latent, set(seen), torch.Generator().manual_seed(seed), w, b)


def _reference_sample(logits, seen, gen):
    """coqui's strategy written the obvious way: one Python step per penalised token."""
    x = logits.clone().float()
    for i in sorted(seen):
        x[i] = x[i] / REPETITION_PENALTY if x[i] > 0 else x[i] * REPETITION_PENALTY
    x = x / TEMPERATURE
    if TOP_K < x.numel():
        x[x < torch.topk(x, TOP_K).values[-1]] = -math.inf
    order = torch.sort(x, descending=True).indices
    cum = torch.softmax(x[order], dim=-1).cumsum(dim=-1)
    drop = torch.zeros_like(order, dtype=torch.bool)
    drop[1:] = cum[:-1] > TOP_P  # shifted, so the leading candidate always survives
    x[order[drop]] = -math.inf
    return int(torch.multinomial(torch.softmax(x, dim=-1), 1, generator=gen))


@pytest.mark.parametrize("case", range(8))
def test_vectorised_penalty_matches_the_per_token_form(case):
    data = torch.Generator().manual_seed(case)
    # Two regimes, because the penalty is only observable when the tokens it touches are in
    # contention: peaked and positive, then flat and negative, where a wrongly-divided logit
    # becomes the leader instead of dropping out.
    scale, offset = ((3.0, 0.0), (0.3, -1.0))[case % 2]
    logits = torch.randn(VOCAB, generator=data) * scale + offset
    n_seen = (0, 1, 5, 40)[case % 4]
    # The leaders, as in a real request: `seen` holds codes already chosen, which scored highest.
    seen = set(torch.topk(logits, n_seen).indices.tolist()) if n_seen else set()
    latent, w, b = _head(logits)
    got = [_sample_token(latent, seen, torch.Generator().manual_seed(s), w, b) for s in range(DRAWS)]
    want = [_reference_sample(logits, seen, torch.Generator().manual_seed(s)) for s in range(DRAWS)]
    if got != want:
        at = next(i for i, (a, b_) in enumerate(zip(got, want)) if a != b_)
        raise AssertionError(f"case {case} ({n_seen} seen): draw {at} gave {got[at]}, per-token gave {want[at]}")


def test_repetition_penalty_pushes_a_negative_logit_down():
    """A seen token must become LESS likely, so a negative logit is multiplied, not divided.
    Dividing would move -1.0 to -0.1 and leave it leading."""
    logits = torch.full((VOCAB,), -50.0)
    logits[7], logits[8] = -1.0, -5.0
    assert _draw(logits) == 7, "unpenalised, 7 leads"
    assert _draw(logits, seen=(7,)) == 8, "penalised to -10.0, 7 must fall behind 8"


def test_top_k_drops_everything_below_the_kth_logit():
    """Near-flat across the WHOLE vocab, so top-p alone would keep hundreds of codes and top-k is
    the only thing that can hold the draw inside the leading TOP_K."""
    logits = -0.001 * torch.arange(VOCAB, dtype=torch.float32)  # strictly ranked by index
    drawn = {_draw(logits, seed=s) for s in range(DRAWS)}
    beyond = sorted(d for d in drawn if d >= TOP_K)
    assert not beyond, f"drew {beyond} from beyond the top-{TOP_K}"
    assert len(drawn) > 1, "a near-flat distribution should not collapse to one code"


def test_top_p_keeps_the_leading_candidate():
    """When the leader alone exceeds TOP_P, the shifted mask must still keep it — otherwise every
    logit becomes -inf and the draw is meaningless."""
    logits = torch.full((VOCAB,), 0.0)
    logits[3] = 20.0
    assert _draw(logits) == 3


def test_the_draw_follows_the_given_generator():
    logits = torch.randn(VOCAB, generator=torch.Generator().manual_seed(3))
    assert _draw(logits, seed=5) == _draw(logits, seed=5), "same seed must repeat"
    assert len({_draw(logits, seed=s) for s in range(DRAWS)}) > 1, "the generator is not being consumed"
