# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance-adaptive speculation for the Gemma 4 speculative-contract rails.

Pure host arithmetic (no ttnn): the statistics an adapter keeps per request,
the step-level record choice and the exploration rules. Both contract rails
feed it the accepted counts the plugin hands back and a cost table measured
at warmup: the MTP rail (``generator_vllm.Gemma4MTPContractForCausalLM``,
several captured ``(rows, drafts)`` records,
``spec_decode.SpeculativeDecoder.serving_measure_costs``) chooses a record or
the plain step; the dFlash rail (``Gemma4DFlashContractForCausalLM``, one
block of ``K`` drafts per verify width,
``dflash_drafter.DFlashFusedDecoder.measure_widths``) chooses between its
one record and the plain step.

The model
---------

Per request row ``r`` and draft position ``j = 1..K_max`` the row keeps
``c_r[j]``, an EWMA of the *conditional* acceptance: whether draft ``j`` was
accepted given drafts ``1..j-1`` were. A verify that carried ``v`` drafts
and committed ``a + 1`` tokens (``accepted_counts = a + 1``) observed
positions ``1..min(a + 1, v)``: accepted at ``j <= a``, rejected at
``j = a + 1`` when that draft was carried; later positions were not reached
and are not updated. The greedy accept walk commits a prefix, so the
probability that draft ``j`` commits is ``p_r(j) = c_r[1] * ... * c_r[j]``,
and a step at ``K`` drafts commits

    T_r(K) = 1 + sum_{j=1..K} p_r(j)

tokens for the row in expectation (the ``1`` is the bonus or correction
token). A position nobody has observed yet (neither this row nor the batch
prior) is taken to be no easier than the position before it:
``c_eff[j] = min(c[j], c_eff[j-1])``, so a row whose first draft keeps being
rejected does not keep the deeper positions at the optimistic start value.
A row with no observation of its own starts from the batch prior, the same
EWMAs over every row's observations; the prior starts optimistic (every
``c = 1``) so the first steps of a server speculate at the largest record
and learn from there. Per-position conditionals extrapolate: a row verified
at ``K = 1`` for a while that becomes predictable lifts ``c[1]``, which
lifts ``T(5)`` through the products, so the policy can move back up without
having observed the deeper positions recently.

The choice
----------

A step with ``n`` drafting rows may replay any captured record whose rows
cover ``n`` and whose draft length is at most the rows-based cap of
``GEMMA4_SPEC_K_BY_ROWS`` (the plugin's ``SpecPlan.draft_cap`` truncates
anything longer at the verify); the dFlash rail has the one record
``(1, K)``. For record ``R = (B_v, K_v)`` at the step's verify width ``w``:

    tokens(R) = sum_r T_r(K_v)
    cost(R)   = step_ms[R, w] + reseed_ms[R] if a row reseeds this step
    gain(R)   = (tokens(R) / n) / (cost(R) / plain_ms)

i.e. expected tokens per row per step, over the step cost relative to one
plain decode step. The record with the largest gain is replayed when that
gain is at least ``GEMMA4_SPEC_MIN_GAIN``; otherwise every row proposes zero
drafts and the step is the plain decode. The decision is per step for the
whole batch: the fused replay evaluates every row's ``K_v`` columns whatever
a row offers, so a zero-draft row inside the replay saves nothing, and a
row left out of it costs an extra plain decode.

Exploration
-----------

While the policy declines, no draft is verified and the statistics do not
move, so after ``GEMMA4_SPEC_RETRY_STEPS`` consecutive plain steps one step
is replayed at the smallest draft length that covers the rows (the cheapest
observation). On the MTP rail a declined row reseeds and resumes within the
request; on the dFlash rail a plain step gives the drafter no residual tap,
so a decline ends the request's session and the retry lands on the first
proposal of a later request (``note_plain_steps`` counts the plain steps in
between). While it speculates below the cap, after
``GEMMA4_SPEC_PROBE_STEPS`` consecutive such steps one step is replayed at
the next larger draft length, so deeper positions are re-observed
(``0`` disables the probe).
"""

from __future__ import annotations

import dataclasses
import os

SPEC_ADAPTIVE_ENV = "GEMMA4_SPEC_ADAPTIVE"
SPEC_ADAPTIVE_DEFAULT = "1"
SPEC_MIN_GAIN_ENV = "GEMMA4_SPEC_MIN_GAIN"
SPEC_MIN_GAIN_DEFAULT = "1.10"
SPEC_RETRY_STEPS_ENV = "GEMMA4_SPEC_RETRY_STEPS"
SPEC_RETRY_STEPS_DEFAULT = "8"
SPEC_PROBE_STEPS_ENV = "GEMMA4_SPEC_PROBE_STEPS"
SPEC_PROBE_STEPS_DEFAULT = "32"
SPEC_EWMA_ENV = "GEMMA4_SPEC_EWMA"
SPEC_EWMA_DEFAULT = "0.2"
SPEC_PLAIN_STEP_MS_ENV = "GEMMA4_SPEC_PLAIN_STEP_MS"
#: The env knobs the policy reads, for harnesses that clear the shell's values.
SPEC_POLICY_ENV = (
    SPEC_ADAPTIVE_ENV,
    SPEC_MIN_GAIN_ENV,
    SPEC_RETRY_STEPS_ENV,
    SPEC_PROBE_STEPS_ENV,
    SPEC_EWMA_ENV,
    SPEC_PLAIN_STEP_MS_ENV,
)


def _env_float(env, name, default, *, low, high=None):
    raw = env.get(name, default)
    try:
        value = float(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < low or (high is not None and value > high):
        bound = f"at least {low}" if high is None else f"in [{low}, {high}]"
        raise ValueError(f"{name} must be {bound}, got {value}")
    return value


def _env_int(env, name, default, *, low):
    raw = env.get(name, default)
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < low:
        raise ValueError(f"{name} must be at least {low}, got {value}")
    return value


@dataclasses.dataclass(frozen=True)
class SpecPolicyConfig:
    """The policy knobs (``from_env`` reads and validates them once, at adapter construction)."""

    adaptive: bool = True
    min_gain: float = 1.10
    retry_steps: int = 8
    probe_steps: int = 32
    ewma: float = 0.2
    plain_step_ms: float | None = None

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        raw = str(env.get(SPEC_ADAPTIVE_ENV, SPEC_ADAPTIVE_DEFAULT)).strip().lower()
        if raw not in ("0", "1", "true", "false", "yes", "no"):
            raise ValueError(f"{SPEC_ADAPTIVE_ENV} must be 1 or 0, got {raw!r}")
        plain = env.get(SPEC_PLAIN_STEP_MS_ENV)
        return cls(
            adaptive=raw in ("1", "true", "yes"),
            min_gain=_env_float(env, SPEC_MIN_GAIN_ENV, SPEC_MIN_GAIN_DEFAULT, low=1.0),
            retry_steps=_env_int(env, SPEC_RETRY_STEPS_ENV, SPEC_RETRY_STEPS_DEFAULT, low=0),
            probe_steps=_env_int(env, SPEC_PROBE_STEPS_ENV, SPEC_PROBE_STEPS_DEFAULT, low=0),
            ewma=_env_float(env, SPEC_EWMA_ENV, SPEC_EWMA_DEFAULT, low=0.0, high=1.0),
            plain_step_ms=(
                None
                if plain is None or not str(plain).strip()
                else _env_float(env, SPEC_PLAIN_STEP_MS_ENV, plain, low=0.0)
            ),
        )


class AcceptanceStats:
    """Conditional per-position acceptance EWMAs of one row (or of the batch prior).

    ``cond[j]`` estimates the acceptance of draft ``j + 1`` given the drafts
    before it were accepted; ``observed[j]`` counts its updates on this
    object and ``informed[j]`` says whether the value rests on any
    observation at all (its own, or the prior's when it was opened as a
    child); ``ewma_accepted`` tracks the accepted drafts per speculative step
    and ``spec_steps`` the steps observed.
    """

    __slots__ = ("k_max", "alpha", "cond", "observed", "informed", "ewma_accepted", "spec_steps")

    def __init__(self, k_max, alpha, cond=None, informed=None):
        self.k_max = int(k_max)
        self.alpha = float(alpha)
        if self.k_max < 1:
            raise ValueError(f"AcceptanceStats needs k_max >= 1, got {k_max}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"AcceptanceStats needs alpha in [0, 1], got {alpha}")
        if cond is None:
            cond = [1.0] * self.k_max
        if len(cond) != self.k_max:
            raise ValueError(f"AcceptanceStats prior has {len(cond)} positions, expected {self.k_max}")
        if informed is None:
            informed = [False] * self.k_max
        if len(informed) != self.k_max:
            raise ValueError(f"AcceptanceStats informed flags have {len(informed)} positions, expected {self.k_max}")
        self.cond = [float(c) for c in cond]
        self.observed = [0] * self.k_max
        self.informed = [bool(f) for f in informed]
        self.ewma_accepted = None
        self.spec_steps = 0

    def child(self):
        """A fresh row that starts from this prior."""
        return AcceptanceStats(self.k_max, self.alpha, cond=self.cond, informed=self.informed)

    def observe(self, accepted, verified):
        """One verify that carried ``verified`` drafts and accepted ``accepted`` of them (a prefix)."""
        verified = min(int(verified), self.k_max)
        if verified < 1:
            return
        accepted = max(0, min(int(accepted), verified))
        reached = min(accepted + 1, verified)
        for j in range(reached):
            outcome = 1.0 if j < accepted else 0.0
            self.cond[j] += self.alpha * (outcome - self.cond[j])
            self.observed[j] += 1
            self.informed[j] = True
        if self.ewma_accepted is None:
            self.ewma_accepted = float(accepted)
        else:
            self.ewma_accepted += self.alpha * (accepted - self.ewma_accepted)
        self.spec_steps += 1

    def effective_cond(self, k):
        """``[c_eff(1), ..., c_eff(k)]``: the conditionals, an uninformed position capped by the one before it."""
        k = max(0, min(int(k), self.k_max))
        out = []
        for j in range(k):
            c = self.cond[j]
            if j and not self.informed[j]:
                c = min(c, out[-1])
            out.append(c)
        return out

    def accept_probs(self, k):
        """``[p(1), ..., p(k)]``: the probability that draft ``j`` commits (the conditionals' running product)."""
        probs, running = [], 1.0
        for c in self.effective_cond(k):
            running *= c
            probs.append(running)
        return probs

    def expected_tokens(self, k):
        """Expected committed tokens of one step at ``k`` drafts: the bonus plus the drafts that commit."""
        return 1.0 + sum(self.accept_probs(k))


@dataclasses.dataclass(frozen=True)
class RecordCandidate:
    """One record the step may replay: its shape, the verify width the step's positions select, and the
    cost of replaying it this step (the measured step cost, plus the reseed cost when a row reseeds)."""

    shape: tuple
    width: int
    cost_ms: float


@dataclasses.dataclass(frozen=True)
class RecordChoice:
    shape: tuple
    width: int
    expected_tokens: float  # summed over the drafting rows
    cost_ms: float
    gain: float  # tokens per ms relative to the plain step (1.0 = the same throughput)
    reason: str  # "gain", "retry" or "probe"


def rank_records(candidates, rows, plain_ms):
    """Every candidate scored: ``[(gain, RecordChoice)]`` sorted best first (ties: the cheaper record)."""
    n = len(rows)
    if n < 1:
        raise ValueError("rank_records needs at least one drafting row")
    plain_ms = float(plain_ms)
    if plain_ms <= 0.0:
        raise ValueError(f"rank_records needs a positive plain step cost, got {plain_ms}")
    ranked = []
    for cand in candidates:
        cost = float(cand.cost_ms)
        if cost <= 0.0:
            raise ValueError(f"record {cand.shape} at S_k={cand.width} has a non-positive cost {cost}")
        k_v = int(cand.shape[1])
        tokens = sum(stats.expected_tokens(k_v) for stats in rows)
        gain = (tokens / n) / (cost / plain_ms)
        ranked.append(RecordChoice(tuple(cand.shape), int(cand.width), tokens, cost, gain, "gain"))
    ranked.sort(key=lambda c: (-c.gain, c.cost_ms, c.shape))
    return ranked


def _smallest_k(candidates, above=None):
    """The candidate with the smallest draft length (above ``above`` when given), cheapest first; None if none."""
    pool = [c for c in candidates if above is None or int(c.shape[1]) > int(above)]
    if not pool:
        return None
    return min(pool, key=lambda c: (int(c.shape[1]), float(c.cost_ms), tuple(c.shape)))


class SpecPolicy:
    """The step-level decision with its exploration counters and the batch prior.

    ``decide`` returns the ``RecordChoice`` to replay or ``None`` for a plain
    step; ``declined`` and ``below_cap`` count the consecutive steps of each
    kind for the retry and probe rules; ``last_k`` is the draft length of the
    previous decision (0 for a plain step, None before the first).
    """

    def __init__(self, config, k_max):
        self.config = config
        self.k_max = int(k_max)
        self.prior = AcceptanceStats(self.k_max, config.ewma)
        self.declined = 0
        self.below_cap = 0
        self.last_k = None

    def new_row(self):
        return self.prior.child()

    def observe(self, row, accepted, verified):
        """Record one verify's outcome on its row and on the batch prior."""
        row.observe(accepted, verified)
        self.prior.observe(accepted, verified)

    def note_plain_steps(self, n=1):
        """Count ``n`` plain steps the policy was not asked about (no session could speculate) toward the retry."""
        self.declined += int(n)
        self.below_cap = 0
        self.last_k = 0

    def retry_due(self):
        return self.config.retry_steps > 0 and self.declined >= self.config.retry_steps

    def would_speculate(self, candidates, rows, plain_ms):
        """Whether ``decide`` would replay a record now (a due retry counts), without changing any counter."""
        if self.retry_due():
            return True
        return rank_records(candidates, rows, plain_ms)[0].gain >= self.config.min_gain

    def decide(self, candidates, rows, plain_ms, k_cap):
        """The record for a step of ``rows`` (their statistics) among ``candidates``, or None for plain.

        ``k_cap`` is the largest draft length the step may use (the rows-based
        cap); a choice below it counts toward the probe.
        """
        cfg = self.config
        candidates = list(candidates)
        if not candidates:
            raise ValueError("decide needs at least one candidate record")
        ranked = rank_records(candidates, rows, plain_ms)
        best = ranked[0]
        choice = None
        if best.gain >= cfg.min_gain:
            choice = best
        elif self.retry_due():
            retry = _smallest_k(candidates)
            choice = dataclasses.replace(next(c for c in ranked if c.shape == retry.shape), reason="retry")
        if choice is None:
            self.declined += 1
            self.below_cap = 0
            self.last_k = 0
            return None
        self.declined = 0
        if int(choice.shape[1]) < int(k_cap):
            self.below_cap += 1
            if cfg.probe_steps > 0 and self.below_cap >= cfg.probe_steps:
                larger = _smallest_k(candidates, above=choice.shape[1])
                if larger is not None:
                    choice = dataclasses.replace(next(c for c in ranked if c.shape == larger.shape), reason="probe")
                self.below_cap = 0
        else:
            self.below_cap = 0
        self.last_k = int(choice.shape[1])
        return choice


def describe_costs(costs, plain_ms):
    """One line for the warmup log: the plain step and every record's step and reseed cost in ms."""
    parts = [f"plain {plain_ms:.1f}" if plain_ms is not None else "plain unmeasured"]
    shapes = sorted({key[1] for key in costs})
    for shape in shapes:
        steps = sorted((key[2], ms) for key, ms in costs.items() if key[0] == "step" and key[1] == shape)
        reseed = costs.get(("reseed", shape))
        text = ", ".join(f"S_k {w} {ms:.1f}" for w, ms in steps)
        if reseed is not None:
            text = f"{text}; reseed {reseed:.1f}" if text else f"reseed {reseed:.1f}"
        parts.append(f"{shape}: {text}")
    return " | ".join(parts)
