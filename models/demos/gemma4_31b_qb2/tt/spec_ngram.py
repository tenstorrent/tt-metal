# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host n-gram drafter with a cost-aware row budget for the 32-row QB2 verify step.

Drafts come from the request's own token history (prompt, including prefix-cached tokens, plus output): the most
recent earlier occurrence of the longest matching suffix (up to MAX_N tokens) proposes the tokens that followed
it. Draft lengths are chosen across all decoding requests at once: the decode matmuls hold one 32-row tile, and the
measured step time is nearly flat up to ~16 rows, so draft rows are granted greedily by expected accepted tokens
per marginal millisecond, and a step verifies only if its expected extra tokens are worth the extra rows plus the
fixed verify overhead; otherwise it stays a plain (pipelined) decode.

Acceptance estimate: first-token acceptance by matched suffix length, measured offline on real T=1.0 agent traffic
(experiment-29 spec-decode LOG L2); depth j is valued q(m)**j.
"""
import os

import numpy as np

MAX_N = 8
MATCH_CAP = 32
ROW_BUDGET = 32
VERIFY_OVERHEAD_MS = float(os.environ.get("GEMMA_SPEC_OVERHEAD_MS", "4.0"))
MAX_DRAFT = int(os.environ.get("GEMMA_SPEC_MAX_DRAFT", "7"))
# (min matched length, first-token acceptance)
P_FIRST = ((25, 0.95), (13, 0.86), (9, 0.76), (6, 0.67), (4, 0.59), (3, 0.48), (2, 0.38), (1, 0.30))
# Full 60-layer plain decode step (ms) on QB2 by rows, at 4K/17K/32K context (spec-decode LOG L6).
STEP_ROWS = (1, 4, 8, 12, 16, 20, 24, 28, 32)
STEP_MS = {
    4096: (43.5, 43.6, 43.6, 43.8, 44.1, 44.7, 45.6, 47.7, 49.6),
    17408: (48.2, 48.2, 48.4, 48.9, 50.3, 52.5, 54.9, 58.4, 61.8),
    32768: (53.6, 53.8, 54.0, 55.6, 57.3, 61.4, 65.5, 70.6, 75.8),
}


def _interp(x, xs, ys):
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1] + (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) * (x - xs[-1])
    i = int(np.searchsorted(xs, x, side="right")) - 1
    return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])


def step_ms(rows, ctx):
    ctxs = sorted(STEP_MS)
    return _interp(ctx, ctxs, [_interp(rows, STEP_ROWS, STEP_MS[c]) for c in ctxs])


def first_token_acceptance(m):
    for mm, q in P_FIRST:
        if m >= mm:
            return q
    return 0.0


def propose_one(history, k):
    """Longest-suffix match (n <= MAX_N, most recent occurrence). Returns (draft list, matched length)."""
    a = np.asarray(history)
    L = a.shape[0]
    if L < 2 or k <= 0:
        return [], 0
    # candidates j: a[j] == a[L-1], j <= L-2, continuation starts at j+1 <= L-1
    cand = np.flatnonzero(a[: L - 1] == a[L - 1])
    if cand.size == 0:
        return [], 0
    length = np.ones(cand.size, dtype=np.int32)
    alive = np.ones(cand.size, dtype=bool)
    for d in range(1, MATCH_CAP):
        pos = cand - d
        ok = alive & (pos >= 0)
        if not ok.any():
            break
        same = np.zeros(cand.size, dtype=bool)
        same[ok] = a[pos[ok]] == a[L - 1 - d]
        alive = ok & same
        length += alive
        if not alive.any():
            break
    # Longest match, capped at MAX_N for selection (longer matches only raise confidence); most recent on ties.
    sel_len = np.minimum(length, MAX_N)
    best = sel_len.max()
    idx = np.flatnonzero(sel_len == best)[-1]
    j = int(cand[idx])
    draft = a[j + 1 : j + 1 + k].tolist()
    return draft, int(length[idx])


# Verify overhead over a plain step with the same rows, by verify trace width (candidates per request), measured in
# serving (F6: 1 row + 6 drafts cost 55.2 ms vs 47.6 plain, ~0.2 ms of it rows).
OVERHEAD_P4_MS = float(os.environ.get("GEMMA_SPEC_OVERHEAD_P4_MS", "5.0"))
OVERHEAD_P8_MS = float(os.environ.get("GEMMA_SPEC_OVERHEAD_P8_MS", "8.0"))
PRIOR_WEIGHT = float(os.environ.get("GEMMA_SPEC_PRIOR_WEIGHT", "20"))


def bucket(m):
    for b, (mm, _) in enumerate(P_FIRST):
        if m >= mm:
            return b
    return len(P_FIRST) - 1


class AcceptanceModel:
    """Realized acceptance by matched-length bucket and draft depth, with the offline table as prior.

    Acceptance of a deterministic draft token is the event "the sampled token equals it", so comparing a proposal
    with the tokens the request actually committed afterwards scores it whether or not it was verified.
    """

    def __init__(self, depths=8):
        self.n = np.zeros((len(P_FIRST), depths))
        self.s = np.zeros((len(P_FIRST), depths))

    def observe(self, m, draft, committed):
        b = bucket(m)
        for j, (d, c) in enumerate(zip(draft, committed)):
            if j >= self.n.shape[1]:
                break
            self.n[b, j] += 1
            if d != c:
                break
            self.s[b, j] += 1

    def q(self, m, j):
        """P(accept depth j | accepted depths < j), j >= 1."""
        b = bucket(m)
        prior = first_token_acceptance(m)
        k = min(j - 1, self.n.shape[1] - 1)
        return (self.s[b, k] + PRIOR_WEIGHT * prior) / (self.n[b, k] + PRIOR_WEIGHT)

    def summary(self):
        return {"attempts": self.n.sum(axis=0).astype(int).tolist(), "accepted": self.s.sum(axis=0).astype(int).tolist()}


MODEL = AcceptanceModel()


# 64-row verify (decoder.decode_forward_wide, GEMMA_SPEC_WIDE=1): extra ms over a plain 32-row step. Placeholder
# until measured on the QB2 (spec-decode LOG); the step costs the same whatever the number of drafts.
WIDE_EXTRA_MS = float(os.environ.get("GEMMA_SPEC_WIDE_EXTRA_MS", "9.0"))


def allocate_best(proposals, contexts, wide=False):
    """Best of the <=32-row plan and (if wide) the one-draft-per-request 64-row plan, by expected net gain."""
    narrow = allocate(proposals, contexts)
    B = len(proposals)
    if not wide or B <= 16 or B > 32:
        return narrow
    mean_ctx = float(np.mean(contexts)) if contexts else 4096.0
    t_plain = step_ms(B, mean_ctx)
    ms_per_token = t_plain / B

    def gain(take):
        rows = B + sum(take)
        exp = 0.0
        for (d, m), k in zip(proposals, take):
            v = 1.0
            for j in range(1, k + 1):
                v *= MODEL.q(m, j)
                exp += v
        over = OVERHEAD_P8_MS if max(take, default=0) > 3 else OVERHEAD_P4_MS
        if rows <= ROW_BUDGET:
            cost = (step_ms(rows, mean_ctx) - t_plain + over) if rows > B else 0.0
        else:
            cost = step_ms(32, mean_ctx) + WIDE_EXTRA_MS - t_plain + over
        return exp * ms_per_token - cost

    wide_take = [1 if (d and MODEL.q(m, 1) > 0) else 0 for d, m in proposals]
    if sum(wide_take) + B <= ROW_BUDGET:
        return narrow
    return wide_take if gain(wide_take) > max(0.0, gain(narrow)) else narrow


def _plan(proposals, mean_ctx, B, cap, row_budget, model):
    t_plain = step_ms(B, mean_ctx)
    ms_per_token = t_plain / B
    cand = []
    for i, (draft, m) in enumerate(proposals):
        v = 1.0
        for j in range(1, min(len(draft), cap) + 1):
            v *= model.q(m, j)
            cand.append((v, i, j))
    cand.sort(key=lambda x: -x[0])
    take = [0] * B
    rows = B
    expected = 0.0
    for value, i, j in cand:
        if rows >= row_budget or j != take[i] + 1:
            continue
        extra = step_ms(rows + 1, mean_ctx) - step_ms(rows, mean_ctx)
        if value * ms_per_token < extra:
            continue
        take[i] = j
        rows += 1
        expected += value
    overhead = OVERHEAD_P8_MS if max(take, default=0) > 3 else OVERHEAD_P4_MS
    gain = expected * ms_per_token - (step_ms(rows, mean_ctx) - t_plain + overhead)
    return take, gain


def allocate(proposals, contexts, row_budget=ROW_BUDGET, overhead_ms=None, model=None):
    """proposals: list of (draft list, matched length) per live request. Returns granted k per request (all zero
    when verifying is not expected to pay). Plans with at most 3 and at most 7 drafts per request are compared
    (the 8-candidate verify trace costs more)."""
    B = len(proposals)
    if B == 0:
        return []
    model = MODEL if model is None else model
    mean_ctx = float(np.mean(contexts)) if contexts else 4096.0
    best, best_gain = [0] * B, 0.0
    for cap in (3, 7):
        take, gain = _plan(proposals, mean_ctx, B, cap, row_budget, model)
        if gain > best_gain:
            best, best_gain = take, gain
    return best


def propose(histories, k_max, wide=False):
    """histories: list (per row) of 1-D int arrays, or None for rows with nothing to draft.

    Returns (drafts [rows][k_max] int list padded with 0, num_valid [rows]).
    """
    k_max = min(k_max, MAX_DRAFT)
    live = [i for i, h in enumerate(histories) if h is not None and len(h) >= 2]
    props = [propose_one(histories[i], k_max) for i in live]
    take = allocate_best(props, [len(histories[i]) for i in live], wide=wide)
    drafts = [[0] * k_max for _ in histories]
    valid = [0] * len(histories)
    matched = [0] * len(histories)
    full = [[] for _ in histories]
    for (draft, m), i, k in zip(props, live, take):
        k = min(k, len(draft))
        drafts[i][:k] = draft[:k]
        valid[i] = k
        matched[i] = m
        full[i] = list(draft)
    return drafts, valid, matched, full
