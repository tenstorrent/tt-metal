# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The sampling-correct speculative round: exactness contract as a CPU reference.

Stdlib only. The target model is abstract: a callable from
the committed prefix to the logits of the next token. The served sampler is modelled as the device tail
(``tt/v41/batch_decode.py:DeviceSampler`` and ``ttnn.sampling``'s kernels,
``ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/{compute/sampling.cpp,dataflow/writer_interleaved.cpp}``):

- the candidates are the row's top-128 logits (four TP chips' top-32 each), of which the kernel keeps the
  **top-32** (``nearest32_K = 32``) and masks all but the row's ``k`` (``k`` in 1..32: the generator clamps
  the request's ``top_k`` to 32 and maps "no top_k" to 32, ``tt/v41/spec_verify.py:_host_inputs``);
- the kept logits are scaled by ``1 / T`` and soft-maxed (``mul_block_bcast_scalar_inplace``, ``reduce_c``,
  ``sub_exp_block_bcast_cols_inplace``);
- the nucleus is the shortest sorted prefix whose cumulative probability **exceeds** ``p``
  (``writer_interleaved.cpp``: ``if (cum_prob_f > p_f) { top_p_cutoff = i + 1; break; }``; ``p = 0`` keeps the
  top candidate alone: the plugin's argmax convention for ``top_k == 1``, ``input_batch.py:343-353``);
- one uniform ``u`` in ``[0, 1)`` per row per call (``rand_tile``; the cores' generators run across replays),
  the token is the first of the nucleus whose renormalised cumulative probability exceeds ``u``.

``served_probs`` is that distribution, ``served_draw`` one draw from it. Temperature 0 is the argmax
(``k = 1, p = 0``), as the generator sends it.

**The round.** Position convention: the block ``[x, d_1 .. d_K]`` is verified at
positions ``p .. p + K``; row ``j`` (the row of input ``j``) carries the target's distribution ``P_j`` for
position ``p + j + 1`` given the committed prefix plus ``x, d_1 .. d_j``. The drafter is **greedy**: its
proposal ``q_j`` is a point mass at ``d_{j+1}``.

Two rules, both exact, both implemented here:

- ``round_reject`` (standard speculative sampling, Leviathan et al. 2023 / Chen et al. 2023, with ``q`` a
point
  mass): at row ``j``, accept ``d_{j+1}`` with probability ``min(1, P_j(d) / q_j(d)) = P_j(d)``; on rejection
  draw from the residual ``max(0, P_j - q_j)`` normalised, which for the point mass is ``P_j`` with ``d``
  removed and renormalised by ``1 - P_j(d)``; after ``K`` accepts draw the bonus from ``P_K``. Two uniforms a
  row at most (the accept test and the residual draw).
- ``round_match`` (target-sample / match): at row ``j`` draw
  ``x_j ~ P_j``; if ``j < K`` and ``x_j == d_{j+1}`` accept and continue, else commit ``x_j`` and stop. One
  uniform a row. This is the device step as it stands (``ttnn.sampling`` on every verify row with the
  request's ``k, p, T``) followed by the greedy ``accept`` rule (``spec_verify.py:accept``: the longest prefix
  with ``block[a + 1] == sampled[a]``), so the served path needs **no new kernel**.

**Proof that both commit the target's distribution.** Fix the committed prefix and the drafts. Under
``round_reject`` the token committed at row ``j`` (given rows ``0 .. j - 1`` accepted, i.e. given the prefix
``x, d_1 .. d_j``) is ``d`` with probability ``P_j(d)`` and ``y != d`` with probability
``(1 - P_j(d)) * P_j(y) / (1 - P_j(d)) = P_j(y)``: the committed token is a draw from ``P_j``. Under
``round_match`` it is ``x_j ~ P_j`` by construction. In both, "accept" means "the committed token equals the
draft", which happens with probability ``P_j(d_{j+1})`` and leaves the prefix for row ``j + 1`` equal to the
prefix the next sequential draw would condition on; the round stops at the first committed token that is not
the next draft, and the next round's rows condition on exactly the committed tokens. By induction over the
stream, the ``n``-th committed token is distributed as ``P(. | the first n - 1 committed tokens)``: the
sequential sampler's law. Nothing depends on how the drafts were chosen (the proof holds for any
deterministic ``d``; a draft outside the nucleus is simply rejected with probability one). The two rules
differ only in how they spend uniforms; their committed distributions are identical. At ``T = 0`` every
``P_j`` is a point mass at the argmax and both rules reduce to the greedy longest-prefix match.

What is **not** exact and why it is the served contract anyway: ``P_j`` is the served top-32 / top-p / ``T``
tail, not the full-vocabulary softmax. The plain served step draws from the same tail, so "exact" here means
"the same distribution as plain serving on this stack", which is what
the served sampler and the benchmark honour (temperature 1, no ``top_k``, ``top_p`` 1: ``k = 32``, every
nucleus the 32 candidates).

"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

Logits = Sequence[float]
TargetFn = Callable[[Sequence[int]], Logits]  # the committed prefix -> the next token's logits
DrafterFn = Callable[[Sequence[int], int], list[int]]  # the prefix (ending in x) and K -> the K drafts

CANDIDATES = 32  # the kernel's top-32 of the gathered top-128
TOPK_MAX = 32  # the generator clamps the request's top_k to it; 0 / none -> 32


def _sorted_candidates(logits: Logits) -> list[tuple[float, int]]:
    """The row's candidates in the kernel's order: descending value, ties by the lowest index (stable top-k)."""
    order = sorted(range(len(logits)), key=lambda i: (-float(logits[i]), i))
    return [(float(logits[i]), i) for i in order[:CANDIDATES]]


def served_probs(logits: Logits, k: int, p: float, temperature: float) -> list[tuple[int, float]]:
    """The served sampler's distribution over the row: ``[(token, probability)]`` in the kernel's order (the
    nucleus; probabilities sum to 1). ``temperature <= 0`` is the argmax (``k = 1, p = 0``)."""
    if temperature <= 0:
        k, p, temperature = 1, 0.0, 1.0
    k = TOPK_MAX if k <= 0 else max(1, min(int(k), TOPK_MAX))
    cand = _sorted_candidates(logits)[:k]
    scaled = [v / temperature for v, _ in cand]
    m = max(scaled)
    w = [math.exp(s - m) for s in scaled]
    z = sum(w)
    probs = [x / z for x in w]
    cum, nucleus = 0.0, len(probs)
    for i, q in enumerate(probs):
        cum += q
        if cum > p:
            nucleus = i + 1
            break
    kept = probs[:nucleus]
    z = sum(kept)
    return [(cand[i][1], kept[i] / z) for i in range(nucleus)]


def draw(dist: Sequence[tuple[int, float]], u: float) -> int:
    """The kernel's multinomial: the first token whose cumulative probability exceeds ``u`` (the last one if
    rounding leaves ``u`` above the sum)."""
    cum = 0.0
    for token, q in dist:
        cum += q
        if cum > u:
            return token
    return dist[-1][0]


def served_draw(logits: Logits, k: int, p: float, temperature: float, rng: random.Random) -> int:
    return draw(served_probs(logits, k, p, temperature), rng.random())


def prob_of(dist: Sequence[tuple[int, float]], token: int) -> float:
    for t, q in dist:
        if t == token:
            return q
    return 0.0


def residual(dist: Sequence[tuple[int, float]], token: int) -> list[tuple[int, float]]:
    """``max(0, P - q)`` normalised for a point mass ``q`` at ``token``: ``P`` without ``token``."""
    rest = [(t, q) for t, q in dist if t != token]
    z = sum(q for _, q in rest)
    assert z > 0, "the residual is empty: the target put all its mass on the draft"
    return [(t, q / z) for t, q in rest]


class Params:
    """A request's sampling parameters as the generator hands them to the step."""

    def __init__(self, k: int = 0, p: float = 1.0, temperature: float = 1.0):
        self.k, self.p, self.temperature = k, p, temperature

    def probs(self, logits: Logits) -> list[tuple[int, float]]:
        return served_probs(logits, self.k, self.p, self.temperature)


def round_match(
    target: TargetFn, prefix: Sequence[int], block: Sequence[int], params: Params, rng: random.Random
) -> tuple[int, list[int]]:
    """The target-sample / match rule over the block ``[x, d_1 .. d_K]`` verified after ``prefix``: the
    accepted
    count ``a`` and the emitted tokens (``a`` drafts and the committed token of row ``a``). One uniform a row,
    rows after the first mismatch never drawn (the device draws them; the rule ignores them)."""
    history = list(prefix)
    emitted: list[int] = []
    a = 0
    for j, token in enumerate(block):
        history.append(token)
        x = draw(params.probs(target(history)), rng.random())
        if j < len(block) - 1 and x == block[j + 1]:
            a += 1
            emitted.append(x)
            continue
        emitted.append(x)
        break
    return a, emitted


def round_reject(
    target: TargetFn, prefix: Sequence[int], block: Sequence[int], params: Params, rng: random.Random
) -> tuple[int, list[int]]:
    """Standard speculative sampling with a point-mass proposal: accept ``d_{j+1}`` with probability
    ``P_j(d_{j+1})`` (the uniform below it), else draw from the residual; the bonus from ``P_K``."""
    history = list(prefix)
    emitted: list[int] = []
    a = 0
    for j, token in enumerate(block):
        history.append(token)
        dist = params.probs(target(history))
        if j < len(block) - 1:
            d = block[j + 1]
            if rng.random() < prob_of(dist, d):
                a += 1
                emitted.append(d)
                continue
            emitted.append(draw(residual(dist, d), rng.random()))
            break
        emitted.append(draw(dist, rng.random()))
    return a, emitted


def round_wrong(
    target: TargetFn, prefix: Sequence[int], block: Sequence[int], params: Params, rng: random.Random
) -> tuple[int, list[int]]:
    """The negative control: greedy acceptance (the draft equals the row's argmax) with a sampled bonus. Not
    exact: it commits the draft whenever it is the mode, so the mode is over-represented."""
    history = list(prefix)
    emitted: list[int] = []
    a = 0
    for j, token in enumerate(block):
        history.append(token)
        dist = params.probs(target(history))
        mode = max(dist, key=lambda tq: tq[1])[0]
        if j < len(block) - 1 and block[j + 1] == mode:
            a += 1
            emitted.append(mode)
            continue
        emitted.append(draw(dist, rng.random()))
        break
    return a, emitted


RoundFn = Callable[[TargetFn, Sequence[int], Sequence[int], Params, random.Random], tuple[int, list[int]]]


def plain_stream(target: TargetFn, prompt: Sequence[int], n: int, params: Params, rng: random.Random) -> list[int]:
    """``n`` tokens after ``prompt`` by the served sampler, one a step: the law every round must match."""
    history = list(prompt)
    out: list[int] = []
    for _ in range(n):
        x = draw(params.probs(target(history)), rng.random())
        history.append(x)
        out.append(x)
    return out


def spec_stream(
    target: TargetFn,
    drafter: DrafterFn,
    prompt: Sequence[int],
    n: int,
    k: int,
    params: Params,
    rng: random.Random,
    round_fn: RoundFn = round_match,
) -> tuple[list[int], list[int]]:
    """``n`` tokens after ``prompt`` by speculative rounds of ``k`` drafts (the first token of the stream is a
    plain draw, as the served request's first step is a plain row): the tokens and the accepted count of every
    round."""
    history = list(prompt)
    out: list[int] = []
    x = draw(params.probs(target(history)), rng.random())
    out.append(x)
    accepted: list[int] = []
    while len(out) < n:
        drafts = drafter(history + [x], k)
        a, emitted = round_fn(target, history, [x] + drafts, params, rng)
        accepted.append(a)
        history += [x] + emitted[:-1]
        x = emitted[-1]
        out += emitted
    return out[:n], accepted
