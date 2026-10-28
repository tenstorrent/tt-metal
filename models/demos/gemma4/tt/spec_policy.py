# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance-adaptive speculation for the Gemma 4 speculative-contract rails.

Pure host arithmetic (no ttnn): the statistics an adapter keeps per request,
the position-aware cost table, the step-level record choice and the
exploration rules. Both contract rails feed it the accepted counts the plugin
hands back and a cost table measured at warmup: the MTP rail
(``generator_vllm.Gemma4MTPContractForCausalLM``, several captured
``(rows, drafts)`` records, ``spec_decode.SpeculativeDecoder.serving_measure_costs``)
chooses a record or the plain step; the dFlash rail
(``Gemma4DFlashContractForCausalLM``, one block of ``K`` drafts per verify
width, ``dflash_drafter.DFlashFusedDecoder.measure_widths``) chooses between
its one record and the plain step.

The model
---------

Per request row ``r`` and draft position ``j = 1..K_max`` the row keeps
``c_r[j]``, an EWMA (weight ``GEMMA4_SPEC_EWMA``) of the *conditional*
acceptance: whether draft ``j`` was accepted given drafts ``1..j-1`` were. A
verify that carried ``v`` drafts and committed ``a + 1`` tokens
(``accepted_counts = a + 1``) observed positions ``1..min(a + 1, v)``:
accepted at ``j <= a``, rejected at ``j = a + 1`` when that draft was
carried; later positions were not reached and are not updated. The greedy
accept walk commits a prefix, so the probability that draft ``j`` commits is
``p_r(j) = c_r[1] * ... * c_r[j]``, and a step at ``K`` drafts commits

    T_r(K) = 1 + sum_{j=1..K} p_r(j)

tokens for the row in expectation (the ``1`` is the bonus or correction
token). A row rests on the batch prior, the same EWMAs over every row's
observations, until it has ``ROW_MIN_OBSERVATIONS`` of its own at a
position; the prior starts optimistic (every ``c = 1``) so the first steps
of a server speculate at the largest record and learn from there. A position
nobody has observed yet is taken to be no easier than the position before
it: ``c_eff[j] = min(c[j], c_eff[j-1])``.

Costs
-----

Every cost is a line in the position (``CostLine``): the plain step is
measured at position 0 and at the top position, each record rung at the
first position it serves and at its last, each reseed at 0 and at the top;
the decision reads every line at the step's top position. For record
``R = (B_v, K_v)`` over ``n`` drafting rows:

    T(R)      = sum_r T_r(K_v) / n            expected tokens per row per step
    gain(R)   = T(R) * plain / step(R)        throughput relative to the plain step

Entering, holding, leaving
--------------------------

A record is *entered* (a row reseeds, or the dFlash drafter bootstraps) only
when ``gain >= GEMMA4_SPEC_MIN_GAIN`` and the saving over the next
``GEMMA4_SPEC_ENTRY_HORIZON`` steps covers the entry cost:

    H * (T(R) * plain - step(R)) >= entry(R)

Once speculating (no reseed this step) the best record is *held* while its
gain stays at or above ``GEMMA4_SPEC_EXIT_GAIN``, and the batch leaves for
the plain step only after ``EXIT_STEPS`` consecutive decisions below it,
never on one noisy step. The decision is per step for the whole batch: the
fused replay evaluates every row's ``K_v`` columns whatever a row offers, so
a zero-draft row inside the replay saves nothing, and a row left out of it
costs an extra plain decode.

Retrying
--------

While the plain step serves, no draft is verified and the statistics do not
move. A retry replays the cheapest candidate (entry plus step) once its
overhead is amortized: at least ``GEMMA4_SPEC_RETRY_STEPS`` plain steps
since the last speculative one, and

    (entry + step - plain) / (plain steps since * plain) <= GEMMA4_SPEC_RETRY_OVERHEAD

with the candidate's own gain estimate at least ``min_gain / 4`` (a record
the statistics already rule out is not worth a replay). On the MTP rail a
declined row reseeds and resumes within the request; on the dFlash rail a
plain step gives the drafter no residual tap, so a decline ends the
request's session and the retry lands on the first step of a later request
(``note_plain_steps`` counts the plain steps in between). While speculating
below the rows-based cap, after ``GEMMA4_SPEC_PROBE_STEPS`` consecutive such
steps one step is replayed at the next larger draft length (no reseed), so
deeper positions are re-observed (``0`` disables the probe).
"""

from __future__ import annotations

import dataclasses
import os

SPEC_ADAPTIVE_ENV = "GEMMA4_SPEC_ADAPTIVE"
SPEC_ADAPTIVE_DEFAULT = "1"
SPEC_MIN_GAIN_ENV = "GEMMA4_SPEC_MIN_GAIN"
SPEC_MIN_GAIN_DEFAULT = "1.10"
SPEC_EXIT_GAIN_ENV = "GEMMA4_SPEC_EXIT_GAIN"
SPEC_EXIT_GAIN_DEFAULT = "0.95"
SPEC_ENTRY_HORIZON_ENV = "GEMMA4_SPEC_ENTRY_HORIZON"
SPEC_ENTRY_HORIZON_DEFAULT = "64"
SPEC_RETRY_STEPS_ENV = "GEMMA4_SPEC_RETRY_STEPS"
SPEC_RETRY_STEPS_DEFAULT = "8"
SPEC_RETRY_OVERHEAD_ENV = "GEMMA4_SPEC_RETRY_OVERHEAD"
SPEC_RETRY_OVERHEAD_DEFAULT = "0.03"
SPEC_PROBE_STEPS_ENV = "GEMMA4_SPEC_PROBE_STEPS"
SPEC_PROBE_STEPS_DEFAULT = "32"
SPEC_EWMA_ENV = "GEMMA4_SPEC_EWMA"
SPEC_EWMA_DEFAULT = "0.1"
SPEC_PLAIN_STEP_MS_ENV = "GEMMA4_SPEC_PLAIN_STEP_MS"
#: The env knobs the policy reads, for harnesses that clear the shell's values.
SPEC_POLICY_ENV = (
    SPEC_ADAPTIVE_ENV,
    SPEC_MIN_GAIN_ENV,
    SPEC_EXIT_GAIN_ENV,
    SPEC_ENTRY_HORIZON_ENV,
    SPEC_RETRY_STEPS_ENV,
    SPEC_RETRY_OVERHEAD_ENV,
    SPEC_PROBE_STEPS_ENV,
    SPEC_EWMA_ENV,
    SPEC_PLAIN_STEP_MS_ENV,
)
#: A row's own conditional at a position overrides the batch prior's after this many observations of it.
ROW_MIN_OBSERVATIONS = 3
#: Consecutive decisions below the exit gain before a speculating batch leaves for the plain step.
EXIT_STEPS = 2
#: A retry needs a gain estimate of at least this fraction of the entry gain.
RETRY_MIN_GAIN_FRACTION = 0.25


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


def parse_plain_step_env(raw):
    """``GEMMA4_SPEC_PLAIN_STEP_MS``: ``ms`` (flat) or ``ms_at_0,ms_at_top`` (a line); None when unset or blank."""
    if raw is None or not str(raw).strip():
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) > 2:
        raise ValueError(f"{SPEC_PLAIN_STEP_MS_ENV} takes one or two numbers (ms, or ms_at_0,ms_at_top), got {raw!r}")
    values = []
    for part in parts:
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(f"{SPEC_PLAIN_STEP_MS_ENV} must be a number, got {raw!r}") from exc
        if value <= 0.0:
            raise ValueError(f"{SPEC_PLAIN_STEP_MS_ENV} must be positive, got {value}")
        values.append(value)
    return tuple(values)


@dataclasses.dataclass(frozen=True)
class SpecPolicyConfig:
    """The policy knobs (``from_env`` reads and validates them once, at adapter construction)."""

    adaptive: bool = True
    min_gain: float = 1.10
    exit_gain: float = 0.95
    entry_horizon: int = 64
    retry_steps: int = 8
    retry_overhead: float = 0.03
    probe_steps: int = 32
    ewma: float = 0.1
    plain_step_ms: tuple | None = None

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        raw = str(env.get(SPEC_ADAPTIVE_ENV, SPEC_ADAPTIVE_DEFAULT)).strip().lower()
        if raw not in ("0", "1", "true", "false", "yes", "no"):
            raise ValueError(f"{SPEC_ADAPTIVE_ENV} must be 1 or 0, got {raw!r}")
        min_gain = _env_float(env, SPEC_MIN_GAIN_ENV, SPEC_MIN_GAIN_DEFAULT, low=1.0)
        exit_gain = _env_float(env, SPEC_EXIT_GAIN_ENV, SPEC_EXIT_GAIN_DEFAULT, low=0.0, high=min_gain)
        return cls(
            adaptive=raw in ("1", "true", "yes"),
            min_gain=min_gain,
            exit_gain=exit_gain,
            entry_horizon=_env_int(env, SPEC_ENTRY_HORIZON_ENV, SPEC_ENTRY_HORIZON_DEFAULT, low=1),
            retry_steps=_env_int(env, SPEC_RETRY_STEPS_ENV, SPEC_RETRY_STEPS_DEFAULT, low=0),
            retry_overhead=_env_float(env, SPEC_RETRY_OVERHEAD_ENV, SPEC_RETRY_OVERHEAD_DEFAULT, low=0.0),
            probe_steps=_env_int(env, SPEC_PROBE_STEPS_ENV, SPEC_PROBE_STEPS_DEFAULT, low=0),
            ewma=_env_float(env, SPEC_EWMA_ENV, SPEC_EWMA_DEFAULT, low=0.0, high=1.0),
            plain_step_ms=parse_plain_step_env(env.get(SPEC_PLAIN_STEP_MS_ENV)),
        )


@dataclasses.dataclass(frozen=True)
class CostLine:
    """A cost in ms, linear in the position between two measurements (clamped outside them)."""

    pos_lo: int
    ms_lo: float
    pos_hi: int
    ms_hi: float

    @classmethod
    def flat(cls, ms, pos_hi=0):
        return cls(0, float(ms), int(pos_hi), float(ms))

    @classmethod
    def of(cls, value, pos_hi=0):
        """A ``CostLine`` from a number (flat), a ``(pos_lo, ms_lo, pos_hi, ms_hi)`` tuple, or a ``CostLine``."""
        if isinstance(value, cls):
            return value
        if isinstance(value, (int, float)):
            return cls.flat(value, pos_hi)
        pos_lo, ms_lo, pos_hi_, ms_hi = value
        return cls(int(pos_lo), float(ms_lo), int(pos_hi_), float(ms_hi))

    def at(self, position):
        position = int(position)
        if self.pos_hi <= self.pos_lo or position <= self.pos_lo:
            return self.ms_lo
        if position >= self.pos_hi:
            return self.ms_hi
        frac = (position - self.pos_lo) / (self.pos_hi - self.pos_lo)
        return self.ms_lo + frac * (self.ms_hi - self.ms_lo)

    def describe(self):
        if self.pos_hi <= self.pos_lo or self.ms_lo == self.ms_hi:
            return f"{self.ms_lo:.1f}"
        return f"{self.ms_lo:.1f}@{self.pos_lo}..{self.ms_hi:.1f}@{self.pos_hi}"


def cost_at(value, position):
    """The cost ``value`` (a number, a tuple or a ``CostLine``) at ``position``, in ms."""
    return CostLine.of(value).at(position)


class AcceptanceStats:
    """Conditional per-position acceptance EWMAs of one row (or of the batch prior).

    ``cond[j]`` estimates the acceptance of draft ``j + 1`` given the drafts
    before it were accepted; ``observed[j]`` counts its updates on this
    object and ``informed[j]`` says whether the value rests on any
    observation at all (its own, or the prior's when it was opened as a
    child); a row reads the ``prior`` it was opened from at a position it
    has fewer than ``min_obs`` observations of; ``ewma_accepted`` tracks
    the accepted drafts per speculative step and ``spec_steps`` the steps
    observed.
    """

    __slots__ = ("k_max", "alpha", "cond", "observed", "informed", "prior", "min_obs", "ewma_accepted", "spec_steps")

    def __init__(self, k_max, alpha, cond=None, informed=None, prior=None, min_obs=ROW_MIN_OBSERVATIONS):
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
        self.prior = prior
        self.min_obs = int(min_obs)
        self.ewma_accepted = None
        self.spec_steps = 0

    def child(self):
        """A fresh row that starts from, and keeps reading, this prior."""
        return AcceptanceStats(self.k_max, self.alpha, cond=self.cond, informed=self.informed, prior=self)

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

    def _source(self, j):
        """``(cond, informed)`` at position ``j``: the row's own once it has ``min_obs`` observations, else the prior's."""
        if self.prior is not None and self.observed[j] < self.min_obs:
            return self.prior.cond[j], self.prior.informed[j]
        return self.cond[j], self.informed[j]

    def effective_cond(self, k):
        """``[c_eff(1), ..., c_eff(k)]``: the conditionals, an uninformed position capped by the one before it."""
        k = max(0, min(int(k), self.k_max))
        out = []
        for j in range(k):
            c, informed = self._source(j)
            if j and not informed:
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
    """One record the step may replay: its shape, the verify width the step's positions select, the cost of
    one step on it (ms, at the step's position) and the cost of entering it this step (the reseed, or the
    dFlash bootstrap replay; 0 for a continuing record)."""

    shape: tuple
    width: int
    cost_ms: float
    entry_ms: float = 0.0


@dataclasses.dataclass(frozen=True)
class RecordChoice:
    shape: tuple
    width: int
    rows: int
    expected_tokens: float  # summed over the drafting rows
    cost_ms: float
    entry_ms: float
    gain: float  # tokens per ms relative to the plain step (1.0 = the same throughput), entry excluded
    reason: str  # "gain", "enter", "hold", "retry" or "probe"

    @property
    def tokens_per_row(self):
        return self.expected_tokens / self.rows


def rank_records(candidates, rows, plain_ms):
    """Every candidate scored: ``[RecordChoice]`` sorted best gain first (ties: the cheaper record)."""
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
        if float(cand.entry_ms) < 0.0:
            raise ValueError(f"record {cand.shape} at S_k={cand.width} has a negative entry cost {cand.entry_ms}")
        k_v = int(cand.shape[1])
        tokens = sum(stats.expected_tokens(k_v) for stats in rows)
        gain = (tokens / n) / (cost / plain_ms)
        ranked.append(
            RecordChoice(tuple(cand.shape), int(cand.width), n, tokens, cost, float(cand.entry_ms), gain, "gain")
        )
    ranked.sort(key=lambda c: (-c.gain, c.cost_ms, c.shape))
    return ranked


def _smallest_k(choices, above=None):
    """The choice with the smallest draft length (above ``above`` when given), cheapest first; None if none."""
    pool = [c for c in choices if above is None or int(c.shape[1]) > int(above)]
    if not pool:
        return None
    return min(pool, key=lambda c: (int(c.shape[1]), float(c.cost_ms), tuple(c.shape)))


class SpecPolicy:
    """The step-level decision with its exploration counters and the batch prior.

    ``decide`` returns the ``RecordChoice`` to replay or ``None`` for a plain
    step. ``declined`` counts the plain steps since the last speculative
    decision (the retry's amortization base), ``below_exit`` the consecutive
    decisions below the exit gain while speculating, ``below_cap`` the
    consecutive speculative steps below the rows-based cap (the probe), and
    ``last_k`` is the draft length of the previous decision (0 for a plain
    step, None before the first), and ``trial`` says that decision was a
    retry: one observation, continued only when its gain clears the entry
    threshold, never held.
    """

    def __init__(self, config, k_max):
        self.config = config
        self.k_max = int(k_max)
        self.prior = AcceptanceStats(self.k_max, config.ewma)
        self.declined = 0
        self.below_exit = 0
        self.below_cap = 0
        self.last_k = None
        self.trial = False

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
        self.below_exit = 0
        self.last_k = 0
        self.trial = False

    @property
    def speculating(self):
        return self.last_k is not None and self.last_k > 0

    def entry_pays(self, choice, plain_ms):
        """Whether the saving of ``choice`` over the entry horizon covers its entry cost (true when there is none)."""
        if choice.entry_ms <= 0.0:
            return True
        saving = choice.tokens_per_row * float(plain_ms) - choice.cost_ms
        return self.config.entry_horizon * saving >= choice.entry_ms

    def retry_choice(self, ranked, plain_ms):
        """The retry for a plain-serving batch, or None: the cheapest candidate (entry plus step) once at least
        ``retry_steps`` plain steps passed and its overhead over them is within ``retry_overhead``."""
        cfg = self.config
        if cfg.retry_steps <= 0 or self.declined < cfg.retry_steps:
            return None
        cheapest = min(ranked, key=lambda c: (c.entry_ms + c.cost_ms, c.shape))
        if cheapest.gain < cfg.min_gain * RETRY_MIN_GAIN_FRACTION:
            return None
        overhead = (cheapest.entry_ms + cheapest.cost_ms - float(plain_ms)) / (self.declined * float(plain_ms))
        if overhead > cfg.retry_overhead:
            return None
        return dataclasses.replace(cheapest, reason="retry")

    def would_speculate(self, candidates, rows, plain_ms):
        """Whether ``decide`` would replay a record now, without changing any counter (the dFlash bootstrap gate)."""
        ranked = rank_records(candidates, rows, plain_ms)
        best = ranked[0]
        if best.gain >= self.config.min_gain and self.entry_pays(best, plain_ms):
            return True
        if self.speculating and not self.trial and best.entry_ms <= 0.0 and best.gain >= self.config.exit_gain:
            return True
        return self.retry_choice(ranked, plain_ms) is not None

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
        if best.gain >= cfg.min_gain and self.entry_pays(best, plain_ms):
            choice = dataclasses.replace(best, reason="enter" if best.entry_ms > 0.0 else "gain")
            self.below_exit = 0
        elif self.speculating and not self.trial and best.entry_ms <= 0.0:
            # Hysteresis: a speculating batch holds its best record above the exit gain, and leaves only
            # after EXIT_STEPS consecutive decisions below it. A retry is a trial and gets no hold.
            if best.gain >= cfg.exit_gain:
                self.below_exit = 0
                choice = dataclasses.replace(best, reason="hold")
            else:
                self.below_exit += 1
                if self.below_exit < EXIT_STEPS:
                    choice = dataclasses.replace(best, reason="hold")
        else:
            choice = self.retry_choice(ranked, plain_ms)
        if choice is None:
            self.declined += 1
            self.below_cap = 0
            self.below_exit = 0
            self.last_k = 0
            self.trial = False
            return None
        self.declined = 0
        self.trial = choice.reason == "retry"
        if int(choice.shape[1]) < int(k_cap) and choice.entry_ms <= 0.0:
            self.below_cap += 1
            if cfg.probe_steps > 0 and self.below_cap >= cfg.probe_steps:
                larger = _smallest_k(ranked, above=choice.shape[1])
                if larger is not None:
                    choice = dataclasses.replace(larger, reason="probe")
                self.below_cap = 0
        else:
            self.below_cap = 0
        self.last_k = int(choice.shape[1])
        return choice


def describe_costs(costs, plain):
    """One line for the warmup log: the plain step and every record's step and reseed cost lines (ms)."""
    parts = [f"plain {CostLine.of(plain).describe()}" if plain is not None else "plain unmeasured"]
    shapes = sorted({key[1] for key in costs})
    for shape in shapes:
        steps = sorted((key[2], value) for key, value in costs.items() if key[0] == "step" and key[1] == shape)
        reseed = costs.get(("reseed", shape))
        text = ", ".join(f"S_k {w} {CostLine.of(value).describe()}" for w, value in steps)
        if reseed is not None:
            reseed_text = f"reseed {CostLine.of(reseed).describe()}"
            text = f"{text}; {reseed_text}" if text else reseed_text
        parts.append(f"{shape}: {text}")
    return " | ".join(parts)
