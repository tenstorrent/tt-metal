# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Repetition-Aware Sampling (RAS), from VALL-E 2, exactly as CosyVoice2 implements it.

Confirmed against the real upstream source (`cosyvoice/utils/common.py`, downloaded and
read directly for this port, not assumed from a secondhand summary):

    def ras_sampling(weighted_scores, decoded_tokens, sampling, top_p=0.8, top_k=25,
                      win_size=10, tau_r=0.1):
        top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
        rep_num = (torch.tensor(decoded_tokens[-win_size:]) == top_ids).sum().item()
        if rep_num >= win_size * tau_r:
            weighted_scores[top_ids] = -float('inf')
            top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
        return top_ids

    def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
        prob, indices = [], []
        cum_prob = 0.0
        sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
        for i in range(len(sorted_idx)):
            if cum_prob < top_p and len(prob) < top_k:
                cum_prob += sorted_value[i]
                prob.append(sorted_value[i])
                indices.append(sorted_idx[i])
            else:
                break
        ...
        return indices[prob.multinomial(1, replacement=True)].item()

The point is to break the degenerate loops autoregressive TTS falls into: if the token
just chosen already appears at least once in the last ten emitted tokens, reject it and
resample from the *unfiltered* distribution. `win_size * tau_r = 1.0` with the shipped
defaults, and the comparison is `>=`, so **one** repeat in the window is enough -- far more
aggressive than "repetition aware" suggests, and it fires often in practice.

Two details a from-scratch reimplementation gets wrong, both preserved exactly here:

* the top-p test is `cum_prob < top_p` evaluated **before** adding the current element, so
  the element that crosses the threshold is still included -- the retained mass ends up
  `>= top_p`, not `<= top_p`;
* the multinomial draw is over the **unnormalised** retained probabilities (softmax output
  restricted to the kept indices) -- `torch.multinomial` renormalises internally, so this
  is equivalent to drawing from the true conditional distribution, but only because it does.

`nucleus_filter` below is a vectorised, `torch.topk`-based reimplementation of the same
retention rule (upstream sorts and walks the *whole* distribution in a Python loop to read
at most `top_k` elements off the front of it -- wasted work that does not change the
answer, since the retention predicate is monotone once the values are sorted descending).
It is verified token-for-token against a literal transcription of the loop above, over many
random distributions, in `tests/pcc/test_qwen2lm_generate.py`.

On-device vs. host: this module -- and CosyVoice2's own `nucleus_sampling` -- always runs on
host. The primary, non-repetitive draw CAN also run through `TtDeviceNucleusSampler` (see
qwen2lm.py), which is the path this bring-up wires up per the plan: on-device nucleus
sampling is not a drop-in replacement for this exact algorithm, because `ttnn.topk`-backed
sampling retains mass `<= top_p` (inclusive after adding), not `< top_p` (exclusive before
adding) -- so its retained set can legitimately differ from `nucleus_filter`'s at the
boundary token. The repetition-triggered resample (`weighted_scores[top_ids] = -inf` then a
fresh draw) always needs this exact host implementation: it needs the emitted-token history,
and it rewrites one score before resampling, which is not a batched device operation.
"""
from __future__ import annotations

import torch


def nucleus_filter(probs: torch.Tensor, top_p: float = 0.8, top_k: int = 25):
    """The retained `(values, indices)`, in the same order upstream's loop would produce.

    An element is kept iff the probability mass *strictly before* it (in descending order)
    is under `top_p` -- upstream's loop tests `cum_prob < top_p` before adding the current
    element -- and that predicate is monotone once values are sorted descending, so the kept
    count is just how many elements satisfy it, capped at `top_k`.
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
    """`>= win_size * tau_r` occurrences of `top_ids` in the last `win_size` emitted tokens.

    With the shipped defaults (`win_size=10, tau_r=0.1`) that threshold is 1.0, so a single
    repeat triggers the resample.
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

    `weighted_scores` is mutated on the repetition path (the banned token's score is set to
    `-inf` before the fallback draw), matching upstream -- callers that need the original
    scores afterward must pass a copy.
    """
    top_ids = nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
    if is_repetitive(decoded_tokens, top_ids, win_size, tau_r):
        weighted_scores[top_ids] = -float("inf")
        top_ids = random_sampling(weighted_scores)
    return top_ids


def greedy(weighted_scores: torch.Tensor) -> int:
    """Deterministic stand-in used for correctness tests: argmax, no sampling."""
    return int(weighted_scores.argmax().item())
