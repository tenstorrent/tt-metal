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
ONLINE_COST = os.environ.get("GEMMA_SPEC_ONLINE_COST", "1") == "1"


def batch_bucket(b):
    for hi in (1, 2, 4, 8, 12, 16, 20, 24, 28):
        if b <= hi:
            return hi
    return 32


class CostModel:
    """Serving verify tax, measured online: (verify step period) - (plain step period at the same batch) - (table
    cost of the extra rows), per verify trace width and batch bucket. A verify is synchronous (the plain path's
    async overlap of host scheduling, output processing and input upload is lost), so this tax grows with the batch
    and differs by host; the fixed priors below are what F6 measured at B1. Exponential averages; plain periods
    are only sampled between two plain steps with no prefill in between, verify periods only when no prefill ran
    in the gap after the verify."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.plain = {}
        self.tax = {}
        self.samples = {}

    def observe_plain(self, b, period_ms):
        k = batch_bucket(b)
        old = self.plain.get(k)
        self.plain[k] = period_ms if old is None else old + self.alpha * (period_ms - old)

    def observe_verify(self, b, rows, wide_p8, period_ms, ctx):
        k = batch_bucket(b)
        base = self.plain.get(k)
        if base is None:
            # No clean plain period at this batch yet (verifying every step): nearest measured bucket, moved along
            # the table; with none at all, the table itself (a pipelined plain step runs at device speed).
            if self.plain:
                near = min(self.plain, key=lambda x: abs(x - k))
                base = self.plain[near] + step_ms(b, ctx) - step_ms(near, ctx)
            else:
                base = step_ms(b, ctx)
        sample = period_ms - base - (step_ms(rows, ctx) - step_ms(b, ctx))
        key = (8 if wide_p8 else 4, k)
        old = self.tax.get(key)
        self.tax[key] = sample if old is None else old + self.alpha * (sample - old)
        self.samples[key] = self.samples.get(key, 0) + 1

    def overhead(self, p8, b):
        prior = OVERHEAD_P8_MS if p8 else OVERHEAD_P4_MS
        if not ONLINE_COST:
            return prior
        key = (8 if p8 else 4, batch_bucket(b))
        n = self.samples.get(key, 0)
        if n == 0:
            return prior
        w = min(1.0, n / 16.0)
        return (1 - w) * prior + w * self.tax[key]

    def summary(self):
        return {
            "plain_ms": {k: round(v, 2) for k, v in sorted(self.plain.items())},
            "tax_ms": {f"P{p}b{b}": round(v, 2) for (p, b), v in sorted(self.tax.items())},
            "tax_n": {f"P{p}b{b}": n for (p, b), n in sorted(self.samples.items())},
        }


COST = CostModel()


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
        return {
            "attempts": self.n.sum(axis=0).astype(int).tolist(),
            "accepted": self.s.sum(axis=0).astype(int).tolist(),
        }


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
        over = COST.overhead(max(take, default=0) > 3, B)
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
    overhead = COST.overhead(max(take, default=0) > 3, B)
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


_CTL = {"path": os.environ.get("GEMMA_SPEC_CTL"), "t": 0.0, "v": {}}


def _ctl():
    """Local A/B knob (only when GEMMA_SPEC_CTL names a JSON file): {"disable": bool, "max_draft": int,
    "overhead_p4_ms": float, "overhead_p8_ms": float}, re-read every 2 s. Never set on CI."""
    global OVERHEAD_P4_MS, OVERHEAD_P8_MS
    if not _CTL["path"]:
        return {}
    import json
    import time

    now = time.monotonic()
    if now - _CTL["t"] > 2.0:
        _CTL["t"] = now
        try:
            with open(_CTL["path"]) as h:
                _CTL["v"] = json.load(h)
        except (OSError, ValueError):
            _CTL["v"] = {}
        v = _CTL["v"]
        OVERHEAD_P4_MS = float(v.get("overhead_p4_ms", os.environ.get("GEMMA_SPEC_OVERHEAD_P4_MS", "5.0")))
        OVERHEAD_P8_MS = float(v.get("overhead_p8_ms", os.environ.get("GEMMA_SPEC_OVERHEAD_P8_MS", "8.0")))
    return _CTL["v"]


class Gate:
    """Measured-throughput gate per batch bucket: committed tokens per second of clean step time (no prefill in the
    period) with drafting allowed vs not. Drafting is allowed in a bucket only while its measured rate beats the
    plain rate; blocks of BLOCK steps run in one mode, a fraction EXPLORE of blocks (and any mode with too little
    data) run the other, so the estimate tracks traffic. Catches every cost the planner's model misses (lost async
    overlap, the reload after a verify, host work). Exponentially decayed sums."""

    BLOCK = int(os.environ.get("GEMMA_SPEC_GATE_BLOCK", "48"))
    EXPLORE = float(os.environ.get("GEMMA_SPEC_GATE_EXPLORE", "0.1"))
    MIN_S = 1.0
    DECAY = 0.998
    ENABLED = os.environ.get("GEMMA_SPEC_GATE", "1") == "1"

    def __init__(self):
        self.stats = {}
        self.cur = {}
        self.left = {}
        self.blocks = 0
        self._rng = np.random.default_rng(1234)

    def record(self, bucket, mode, tokens, seconds):
        for key in [k for k in self.stats if k[0] == bucket]:
            t, n = self.stats[key]
            self.stats[key] = (t * self.DECAY, n * self.DECAY)
        t, n = self.stats.get((bucket, mode), (0.0, 0.0))
        self.stats[(bucket, mode)] = (t + seconds, n + tokens)

    def rate(self, bucket, mode):
        t, n = self.stats.get((bucket, mode), (0.0, 0.0))
        return (n / t if t > 0 else None), t

    def mode(self, bucket):
        if not self.ENABLED:
            return True
        left = self.left.get(bucket, 0)
        if left > 0:
            self.left[bucket] = left - 1
            return self.cur[bucket]
        on, t_on = self.rate(bucket, True)
        off, t_off = self.rate(bucket, False)
        if t_on < self.MIN_S or t_off < self.MIN_S:
            best = t_on <= t_off
        else:
            best = on >= off
            # Explore less where the measured difference is large (exploring the losing mode is what it costs).
            explore = self.EXPLORE if abs(on / off - 1.0) < 0.15 else self.EXPLORE / 3
            if self._rng.random() < explore:
                best = not best
        self.cur[bucket] = best
        self.left[bucket] = self.BLOCK - 1
        self.blocks += 1
        return best

    def summary(self):
        out = {}
        for (b, m), (t, n) in sorted(self.stats.items()):
            out[f"b{b}{'on' if m else 'off'}"] = [round(t, 1), round(n / t, 1) if t else None]
        return out


GATE = Gate()


def propose(histories, k_max, wide=False, allow=True):
    """histories: list (per row) of 1-D int arrays, or None for rows with nothing to draft.

    Returns (drafts [rows][k_max] int list padded with 0, num_valid [rows]).
    """
    ctl = _ctl()
    width = k_max
    k_max = min(k_max, MAX_DRAFT, int(ctl.get("max_draft", MAX_DRAFT)))
    live = (
        [] if ctl.get("disable") or not allow else [i for i, h in enumerate(histories) if h is not None and len(h) >= 2]
    )
    props = [propose_one(histories[i], k_max) for i in live]
    take = allocate_best(props, [len(histories[i]) for i in live], wide=wide)
    drafts = [[0] * width for _ in histories]
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
