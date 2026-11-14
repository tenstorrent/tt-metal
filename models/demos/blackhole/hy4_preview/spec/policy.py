# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive speculation length for the Hy4 MTP drafter (the idea of DSV4 tt/spec_policy.py + spec_model.AdaptiveSpec.choose, pure python).

DSV4 picked, per round, the resident runner k maximising E[tokens/round](k) / round_ms(k), with E from a CONFIDENCE HEAD of the drafter. The Hy4 MTP layer has no
confidence head: the estimate here is (a) per-position conditional acceptance rates p_i = P(draft i accepted | drafts < i accepted) tracked as an EMA over finished
rounds (global across active users, DSV4 also shared one k per batch), optionally (b) the draft's own softmax top-1 probability (``conf``) mapped through a
calibration; the first is implemented, (b) is a hook (``conf`` argument of ``expected_tokens``).

Candidate set: default {1, 3} at B=4 (verify rows T = 4 (1+k) = 8 / 16 are the MoE/attention shapes that are measured), k=0 (plain round, no draft) is a candidate
with its own cheaper trace and a probe round every ``probe_every`` rounds so the EMA of the deeper positions does not go stale (same trick as DSV41_SPEC_PROBE).
One resident traced runner per candidate k (every width is a captured trace; no allocation after warm-up).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


def expected_tokens(p: Sequence[float], k: int) -> float:
    """E[tokens per round] for k drafts with conditional acceptance probs p[0..]: 1 + sum_{j=1..k} prod_{i<=j} p_i."""
    e, run = 1.0, 1.0
    for i in range(k):
        run *= p[i]
        e += run
    return e


@dataclass
class AdaptiveK:
    ks: Sequence[int] = (0, 1, 3)  # candidates (0 = plain round)
    round_ms: Dict[int, float] = field(
        default_factory=dict
    )  # measured round time per k (calibration at trace capture, then EMA)
    k_max: int = 3
    ema: float = 0.9  # weight of history
    prior: float = 0.6  # initial per-position conditional acceptance
    hyst: float = 0.02  # relative gain needed to switch away from the current k
    probe_every: int = 32
    p: List[float] = field(default_factory=list)
    _n_since_probe: int = 0
    _cur: int = 0

    def __post_init__(self):
        if not self.p:
            self.p = [self.prior] * self.k_max
        self._cur = max(self.ks)

    def observe(self, k: int, accepted: Sequence[int]) -> None:
        """One finished round of k drafts; ``accepted`` = m per active user. Position i (1-based) was TRIED by a user iff m >= i-1 and i <= k, and succeeded iff m >= i.
        Right-censored conditional estimate, EMA over rounds (each round contributes the batch mean of the tried users).
        """
        for i in range(1, k + 1):
            tried = [m for m in accepted if m >= i - 1]
            if not tried:
                continue
            frac = sum(1 for m in tried if m >= i) / len(tried)
            self.p[i - 1] = self.ema * self.p[i - 1] + (1 - self.ema) * frac

    def observe_time(self, k: int, ms: float, a: float = 0.9) -> None:
        self.round_ms[k] = ms if k not in self.round_ms else a * self.round_ms[k] + (1 - a) * ms

    def score(self, k: int) -> float:
        return expected_tokens(self.p, k) / self.round_ms[k]

    def choose(self) -> int:
        """k for the next round. Needs round_ms for every candidate (startup calibration)."""
        cand = [k for k in self.ks if k in self.round_ms]
        best = max(cand, key=self.score)
        if self._cur in cand and best != self._cur and self.score(best) < (1 + self.hyst) * self.score(self._cur):
            best = self._cur
        # probe: a k=0 stretch never refreshes p; run the largest k once in a while
        self._n_since_probe += 1
        if best < max(cand) and self.probe_every and self._n_since_probe >= self.probe_every and max(cand) > 0:
            self._n_since_probe = 0
            return max(cand)
        if best == max(cand):
            self._n_since_probe = 0
        self._cur = best
        return best
