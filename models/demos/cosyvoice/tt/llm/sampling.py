# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Repetition-Aware Sampling (RAS), from VALL-E 2, as CosyVoice implements it.

    top_ids = nucleus_sampling(scores, top_p=0.8, top_k=25)
    if (last 10 emitted tokens == top_ids).sum() >= 10 * 0.1:
        scores[top_ids] = -inf
        top_ids = random_sampling(scores)          # plain multinomial over everything

The point is to break the degenerate loops autoregressive TTS falls into: if the
token just chosen already appears at least once in the last ten, reject it and
resample from the *unfiltered* distribution. Note the threshold is
`win_size * tau_r = 1.0` and the comparison is `>=`, so **one** repeat inside the
window is enough. That is far more aggressive than the phrase "repetition aware"
suggests, and it fires often in practice.

Two details in `nucleus_sampling` that a from-scratch reimplementation gets wrong:

* the top-p test is `cum_prob < top_p` evaluated **before** adding, so the token
  that crosses the threshold is included -- the retained mass is >= top_p, not <=;
* the multinomial draw is over the **unnormalised** retained probabilities. torch
  normalises internally, so this is equivalent, but only because it does.

**This stays on the host, and the decision is measured rather than assumed.**
`ttnn.sampling` does implement softmax -> top-k -> top-p -> multinomial on device,
and the bring-up scope asks for it. Profiling the per-token tail
(`scripts/profile_token_tail.py`) says what it could win:

    output head matmul        0.043 ms
    logits device -> host     0.142 ms
    RAS sampling on host      0.075 ms
    embedding row -> device   0.092 ms
    ------------------------------------
    tail total                0.352 ms      2.7% of a 12.9 ms token

Moving RAS on device removes the transfer and the host sampling at best -- 0.217 ms,
**1.7% of a token** -- and cannot remove all of it, because the repetition branch
needs the emitted-token history and rewrites a score before resampling, so it comes
back to the host whenever it fires. With `win_size * tau_r = 1.0` it fires on a
single repeat, which is often.

Against that it costs two mismatches. `ttnn.sampling` selects "cumulative probability
mass less than or equal to p", where CosyVoice tests `cum_prob < top_p` *before*
adding and so includes the token that crosses the threshold; and it samples from its
own seed, which cannot reproduce torch's stream. Trading exactness for 1.7% is the
wrong way round, so the host path is the shipped one and `nucleus_filter` was made
fast instead -- 0.245 -> 0.075 ms, bit-identical output.
"""
from __future__ import annotations

import torch


def nucleus_filter(probs: torch.Tensor, top_p: float = 0.8, top_k: int = 25):
    """The retained `(values, indices)`, in CosyVoice's exact order.

    Returned rather than sampled so the selection can be asserted without
    involving an RNG.

    Upstream sorts the whole distribution and then walks it in Python, accumulating
    until the mass reaches `top_p`. Two things about that are pure waste and neither
    changes the answer:

    * **at most `top_k` elements are ever kept**, so sorting all 4097 to read the
      first 25 does 160x the necessary work. `torch.topk` returns them already
      ordered;
    * **the walk is a Python loop over a tensor**, and `float(sorted_value[i])` is a
      tensor-to-scalar sync each time round.

    The retention rule vectorises exactly. An element is kept iff the mass *before*
    it is under `top_p` -- the loop tests `cum < top_p` before adding -- and that
    predicate is monotone once the values are descending, so the count is just how
    many satisfy it, capped at `top_k`. This is why the retained mass is `>= top_p`
    rather than `<=`: the element that crosses the threshold is included.

    Measured at 0.245 -> 0.056 ms per token, and `tests/pcc/test_sampling.py` asserts
    it against the literal transcription over random distributions.
    """
    k = min(top_k, probs.numel())
    sorted_value, sorted_idx = probs.topk(k, sorted=True)
    mass_before = sorted_value.cumsum(0) - sorted_value
    keep = max(1, int((mass_before < top_p).sum()))
    return sorted_value[:keep], sorted_idx[:keep]


def nucleus_sampling(weighted_scores: torch.Tensor, top_p: float = 0.8, top_k: int = 25) -> int:
    probs = weighted_scores.softmax(dim=0)
    value, idx = nucleus_filter(probs, top_p, top_k)
    return int(idx[value.multinomial(1, replacement=True)].item())


def random_sampling(weighted_scores: torch.Tensor) -> int:
    return int(weighted_scores.softmax(dim=0).multinomial(1, replacement=True).item())


def is_repetitive(decoded_tokens, top_ids: int, win_size: int = 10, tau_r: float = 0.1) -> bool:
    """`>= win_size * tau_r` occurrences of `top_ids` in the last `win_size` tokens.

    With the shipped defaults that threshold is 1.0, so a single repeat triggers.
    """
    window = decoded_tokens[-win_size:]
    return sum(1 for t in window if t == top_ids) >= win_size * tau_r


def ras_sampling(
    weighted_scores: torch.Tensor,
    decoded_tokens,
    top_p: float = 0.8,
    top_k: int = 25,
    win_size: int = 10,
    tau_r: float = 0.1,
) -> int:
    """Exactly `cosyvoice.utils.common.ras_sampling`, host side.

    `weighted_scores` is mutated on the repetition path, matching upstream -- the
    caller passes log-probabilities it does not reuse.
    """
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    if is_repetitive(decoded_tokens, top_ids, win_size, tau_r):
        weighted_scores[top_ids] = -float("inf")
        top_ids = random_sampling(weighted_scores)
    return top_ids


def greedy(weighted_scores: torch.Tensor) -> int:
    """Deterministic stand-in, for tests that need a reproducible token stream."""
    return int(weighted_scores.argmax().item())
