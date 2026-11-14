# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verification of k MTP drafts by one (k+1)-row verify step of the main model (pure python, no ttnn).

Row layout (per user, positions P..P+k): input tokens ``tokens = [x_P, d_1, .., d_k]`` (x_P = the pending token, not yet in the KV; d_i = draft tokens).
Row j of the verify step outputs the target distribution of the token at position P+j+1. ``a_j`` = the target's token for that row.

Greedy: a_j = argmax(row j). Draft d_{j+1} is accepted iff d_{j+1} == a_j and all earlier drafts were accepted. m = number of accepted drafts (0..k),
the round emits ``d_1 .. d_m, a_m`` (m + 1 tokens); a_m is the "bonus" (m == k) or the correction (m < k) and becomes the next pending token x_{P+m+1}.

Sampled: the MTP drafts are greedy (argmax of the draft head), i.e. the draft distribution q is one-hot, so the speculative-sampling rule (accept d with prob
min(1, p(d)/q(d)) = p(d); on reject resample from norm(max(0, p - q))) is EXACTLY "sample y_j ~ p_j at every row, accept d_{j+1} iff y_j == d_{j+1}": P(y == d) = p(d),
and y | y != d ~ p restricted to tokens != d, renormalised. ``sample_then_compare`` is therefore the same code path as the greedy accept (a_j = the sampled token of row j,
drawn by the in-trace sampler with its own per-row randomness): no probability tensors leave the device. ``rejection_accept`` is the textbook form, kept as the oracle for
the equivalence test (and for a future non-greedy draft).
"""
import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class RoundResult:
    m: int  # accepted drafts, 0..k
    emitted: tuple  # tokens emitted this round: d_1..d_m, a_m  (len m + 1)
    next_pending: int  # = a_m = emitted[-1]: input token of row 0 of the next round (position P + m + 1)
    new_len: int  # committed KV length after the round = P + m + 1 (positions 0 .. P+m hold valid KV)


def accept_count(drafts: Sequence[int], target_tokens: Sequence[int]) -> int:
    """Leading run of matches: drafts[i] == target_tokens[i] for i < m. ``target_tokens`` has len(drafts) + 1 entries (rows 0..k); drafts has k."""
    assert len(target_tokens) == len(drafts) + 1, (len(target_tokens), len(drafts))
    m = 0
    for d, a in zip(drafts, target_tokens):
        if d != a:
            break
        m += 1
    return m


def finish_round(
    P: int, drafts: Sequence[int], target_tokens: Sequence[int], cap: Optional[int] = None, k_eff: Optional[int] = None
) -> RoundResult:
    """Greedy / sample-then-compare round result. ``cap`` = max drafts accepted (stop at max_tokens: tokens allowed - 1; also the DSV4 'force/cap' hook),
    ``k_eff`` = drafts that are real (rows beyond are filler of a padded verify block: never accept more than k_eff)."""
    m = accept_count(drafts, target_tokens)
    if k_eff is not None:
        m = min(m, k_eff)
    if cap is not None:
        m = min(m, max(cap, 0))
    emitted = tuple(drafts[:m]) + (target_tokens[m],)
    return RoundResult(m=m, emitted=emitted, next_pending=target_tokens[m], new_len=P + m + 1)


def truncate_at_stop(emitted: Sequence[int], stop_ids: Sequence[int], budget: int) -> tuple:
    """Cut the emitted run at the first stop token (inclusive) and at ``budget`` tokens. -> (kept tokens, finished?)."""
    out = []
    for t in emitted:
        if len(out) >= budget:
            return tuple(out), True
        out.append(t)
        if t in stop_ids:
            return tuple(out), True
    return tuple(out), len(out) >= budget


def _norm(p):
    s = sum(p)
    return [x / s for x in p]


def rejection_accept(
    drafts: Sequence[int],
    row_probs: Sequence[Sequence[float]],
    rng: random.Random,
    draft_probs: Optional[Sequence[Sequence[float]]] = None,
):
    """Textbook speculative sampling (Leviathan et al.), the oracle. row_probs[j] = target distribution of row j (len k+1, after temperature / top-p / top-k),
    draft_probs[i] = draft distribution that produced drafts[i] (None = one-hot at the draft = greedy draft). Returns the emitted tokens (m accepted + correction / bonus).
    """
    out = []
    k = len(drafts)
    for i in range(k):
        p, d = row_probs[i], drafts[i]
        q_d = 1.0 if draft_probs is None else draft_probs[i][d]
        if rng.random() < min(1.0, p[d] / q_d):
            out.append(d)
            continue
        if draft_probs is None:
            res = [0.0 if t == d else x for t, x in enumerate(p)]
        else:
            res = [max(0.0, x - y) for x, y in zip(p, draft_probs[i])]
        out.append(_sample(_norm(res), rng))
        return out
    out.append(_sample(row_probs[k], rng))  # bonus token from the last row
    return out


def _sample(p, rng):
    u, c = rng.random(), 0.0
    for t, x in enumerate(p):
        c += x
        if u < c:
            return t
    return len(p) - 1


def sample_then_compare(drafts: Sequence[int], row_probs: Sequence[Sequence[float]], rng: random.Random):
    """The device-friendly exact equivalent for greedy drafts: sample y_j ~ p_j for EVERY row, then the greedy accept rule on the y's. Returns the emitted tokens."""
    y = [_sample(p, rng) for p in row_probs]
    return list(finish_round(0, drafts, y).emitted)


def softmax(logits, temperature=1.0):
    mx = max(logits)
    e = [math.exp((x - mx) / temperature) for x in logits]
    return _norm(e)
