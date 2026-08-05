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

`ttnn.sampling` implements top-k then top-p on device with its own seed. It cannot
reproduce torch's RNG stream -- nothing can -- so this module keeps an exact host
implementation for verification and offers the device path for generation, where
matching the reference's *distribution* is what matters, not its draws.
"""
from __future__ import annotations

import torch


def nucleus_filter(probs: torch.Tensor, top_p: float = 0.8, top_k: int = 25):
    """The retained `(values, indices)`, in CosyVoice's exact order.

    Returned rather than sampled so the selection can be asserted without
    involving an RNG.
    """
    sorted_value, sorted_idx = probs.sort(descending=True, stable=True)
    keep, cum = 0, 0.0
    for i in range(len(sorted_idx)):
        if cum < top_p and keep < top_k:
            cum += float(sorted_value[i])
            keep += 1
        else:
            break
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
