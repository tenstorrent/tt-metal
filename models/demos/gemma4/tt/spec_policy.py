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
token). A row blends its own observations with the batch prior, the same
EWMAs over every row's observations: at a position it has ``n`` own
observations of, fewer than ``ROW_MIN_OBSERVATIONS``, it reads
``n / (n + ROW_MIN_OBSERVATIONS)`` of its own mean outcome and the rest of
the prior (one verify at 3-5 accepted moves a row most of the way), and
from ``ROW_MIN_OBSERVATIONS`` on its own EWMA; the prior starts optimistic
(every ``c = 1``) so the first steps of a server speculate at the largest
record and learn from there. A request that stays plain verifies nothing
and so contributes nothing to the prior. A position
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
(``note_plain_steps`` counts the plain steps in between). Independently of
that cadence, a new request gets one *trial* at its first proposal: the
cheapest candidate (entry plus step) replays once when the prior's gain
estimate is at least ``GEMMA4_SPEC_TRIAL_GAIN`` (the reseed is the only
extra cost; a prior below that is the random-prompt regime learned on this
engine, and the request stays plain). A record serving has measured at the
rung gets no trial below the entry bar: its measured tokens per row say what
its replays deliver, a wave of new rows cannot argue with it, and a trial
that clears the bar is an entry anyway (the 12B Galaxy at 32 users
re-trialed the (8,3) at every arrival wave of random prompts, a 250 ms
reseed each, and left it within seconds; the measured 2.36 tokens a row
realized 1.9-2.0 against a 1.9x step).
A trial is never held: the next decision continues only on the entry gain,
which the trial's own verify feeds through the blend above. A batch that
just left a record waits ``retry_steps`` plain steps before any reseeding
entry (an enter, a trial or a retry; the retry's cadence counts the same
steps): those waves ran enter-exit-enter cycles a few steps apart. While speculating
below the rows-based cap, after ``GEMMA4_SPEC_PROBE_STEPS`` consecutive such
steps one step is replayed at the next larger draft length (no reseed), so
deeper positions are re-observed (``0`` disables the probe).

Calibration
-----------

The warmup table is only the prior. The adapter times every step it serves
(the wall between two consecutive proposals, the tokens the runner committed
in it) and feeds ``Calibration``: per mode (``("plain", bucket)`` for the
plain step at the decode bucket of the live rows, ``("step", shape, width)``
for a record at its rung) the clean walls as *ratios* to the warmup table
at each step's own position, kept in a window of the last ``CALIB_WINDOW``
samples whose median scales the table at decision time (so the position
stays in the model: a long-context sample does not price a short-context
step), the tokens those steps committed per row (their MEAN: a throughput;
the per-step counts are skewed, a median reads a record that commits
1,1,1,5,6 tokens as a loss), and per shape the reseed as served (a ratio
too). The measured tokens per row are pooled over the records of one draft
length at a rung: the drafter's acceptance does not depend on the record's
rows, so a sibling record reads the calibrated one's instead of the model
(which the batch prior inflates at the deep positions) and cannot outrank
it on the model alone. The decision reads the TOTAL wall of the mode in
force: every interval counts, dirty ones included (a prefill ran inside
it, the mode changed or a reseed ran at its start, the live rows changed),
because the served latency carries them -- an arrival's prefill stalls a
batch that speculation desynchronized, a join reseeds, a switch replays
another record -- and a batch whose speculative steps each look profitable
can still serve more ms per token than plain once those are in. Two kinds
of interval are quarantined, reported and never averaged in: an idle one
(every row the previous step served ended before the next proposal, so the
wall is the gap between requests, not the mode's work) and a clean one
above ``OUTLIER_FACTOR`` times the running median (a wait that is not the
mode's work, such as the lock-step wait on another engine's prefill). The
clean figures (no prefill, no switch, no reseed, no row change) stay on
the stats line as the breakdown. Once a mode has ``GEMMA4_SPEC_CALIB_MIN``
counted samples the decision reads the table scaled by its ratio and the
measured tokens per row instead of the acceptance model; the entry rule
compares that against the measured plain step, with the reseed as served,
over a horizon capped by the measured request length in steps
(``note_request_steps``): a short request cannot amortize a reseed a long
one can. While one mode serves,
after ``GEMMA4_SPEC_CALIB_PROBE_STEPS`` steps without a sample of the other
one a bounded probe of the other mode (one step, overhead-gated like the
retry) keeps both estimates fresh.

Joining rows
------------

A row that would reseed while the batch already speculates can be left out
of the replay: it then commits one token per step from the plain decode the
verify runs beside the replay for it, which costs the batch one plain step
per step (``RecordCandidate.members`` names the rows in the replay, the
candidate's cost carries that plain step). The batch enters the full
record, reseeding the joiners, only when the entry rule pays with the
reseed as served.
"""

from __future__ import annotations

import collections
import dataclasses
import os

SPEC_ADAPTIVE_ENV = "GEMMA4_SPEC_ADAPTIVE"
SPEC_ADAPTIVE_DEFAULT = "1"
SPEC_MIN_GAIN_ENV = "GEMMA4_SPEC_MIN_GAIN"
SPEC_MIN_GAIN_DEFAULT = "1.25"
SPEC_EXIT_GAIN_ENV = "GEMMA4_SPEC_EXIT_GAIN"
SPEC_EXIT_GAIN_DEFAULT = "0.95"
SPEC_ENTRY_HORIZON_ENV = "GEMMA4_SPEC_ENTRY_HORIZON"
SPEC_ENTRY_HORIZON_DEFAULT = "64"
SPEC_RETRY_STEPS_ENV = "GEMMA4_SPEC_RETRY_STEPS"
SPEC_RETRY_STEPS_DEFAULT = "8"
SPEC_RETRY_OVERHEAD_ENV = "GEMMA4_SPEC_RETRY_OVERHEAD"
SPEC_RETRY_OVERHEAD_DEFAULT = "0.03"
SPEC_TRIAL_GAIN_ENV = "GEMMA4_SPEC_TRIAL_GAIN"
SPEC_TRIAL_GAIN_DEFAULT = "0.6"
SPEC_TRIAL_STEPS_ENV = "GEMMA4_SPEC_TRIAL_STEPS"
SPEC_TRIAL_STEPS_DEFAULT = "3"
SPEC_PROBE_STEPS_ENV = "GEMMA4_SPEC_PROBE_STEPS"
SPEC_PROBE_STEPS_DEFAULT = "32"
SPEC_EWMA_ENV = "GEMMA4_SPEC_EWMA"
SPEC_EWMA_DEFAULT = "0.1"
SPEC_PLAIN_STEP_MS_ENV = "GEMMA4_SPEC_PLAIN_STEP_MS"
SPEC_STATS_STEPS_ENV = "GEMMA4_SPEC_STATS_STEPS"
SPEC_STATS_STEPS_DEFAULT = "256"
SPEC_CALIB_MIN_ENV = "GEMMA4_SPEC_CALIB_MIN"
SPEC_CALIB_MIN_DEFAULT = "8"
SPEC_CALIB_PROBE_STEPS_ENV = "GEMMA4_SPEC_CALIB_PROBE_STEPS"
SPEC_CALIB_PROBE_STEPS_DEFAULT = "256"
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
    SPEC_STATS_STEPS_ENV,
    SPEC_CALIB_MIN_ENV,
    SPEC_CALIB_PROBE_STEPS_ENV,
    SPEC_TRIAL_GAIN_ENV,
    SPEC_TRIAL_STEPS_ENV,
)
#: The EWMA weight of the request-length estimate (steps per request) the entry horizon is capped by.
CALIB_EWMA = 0.25
#: Clean samples a mode keeps for its median ratio to the warmup table.
CALIB_WINDOW = 32
#: A clean sample above this multiple of the running median is an outlier (reported, never calibrating)...
OUTLIER_FACTOR = 3.0
#: ...once the mode has this many samples to take the median of.
OUTLIER_MIN_SAMPLES = 4
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
    min_gain: float = 1.25
    exit_gain: float = 0.95
    entry_horizon: int = 64
    retry_steps: int = 8
    retry_overhead: float = 0.03
    probe_steps: int = 32
    ewma: float = 0.1
    plain_step_ms: tuple | None = None
    stats_steps: int = 256
    calib_min: int = 8
    calib_probe_steps: int = 256
    trial_gain: float = 0.6
    trial_steps: int = 3

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
            stats_steps=_env_int(env, SPEC_STATS_STEPS_ENV, SPEC_STATS_STEPS_DEFAULT, low=1),
            calib_min=_env_int(env, SPEC_CALIB_MIN_ENV, SPEC_CALIB_MIN_DEFAULT, low=1),
            calib_probe_steps=_env_int(env, SPEC_CALIB_PROBE_STEPS_ENV, SPEC_CALIB_PROBE_STEPS_DEFAULT, low=0),
            trial_gain=_env_float(env, SPEC_TRIAL_GAIN_ENV, SPEC_TRIAL_GAIN_DEFAULT, low=0.0),
            trial_steps=_env_int(env, SPEC_TRIAL_STEPS_ENV, SPEC_TRIAL_STEPS_DEFAULT, low=1),
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
    object, ``own_sum[j]`` the accepted outcomes among them, and
    ``informed[j]`` says whether the value rests on any observation at all
    (its own, or the prior's when it was opened as a child). At a position a
    row has ``n < min_obs`` own observations of, it reads
    ``n / (n + min_obs)`` of its own mean outcome and the rest of the
    ``prior`` it was opened from (nothing but the prior at ``n = 0``); from
    ``min_obs`` on it reads its own EWMA. ``ewma_accepted`` tracks the
    accepted drafts per speculative step and ``spec_steps`` the steps
    observed.
    """

    __slots__ = (
        "k_max",
        "alpha",
        "cond",
        "observed",
        "own_sum",
        "informed",
        "prior",
        "min_obs",
        "ewma_accepted",
        "spec_steps",
    )

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
        self.own_sum = [0.0] * self.k_max
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
            self.own_sum[j] += outcome
            self.informed[j] = True
        if self.ewma_accepted is None:
            self.ewma_accepted = float(accepted)
        else:
            self.ewma_accepted += self.alpha * (accepted - self.ewma_accepted)
        self.spec_steps += 1

    def _source(self, j):
        """``(cond, informed)`` at position ``j``: the row's own EWMA once it has ``min_obs`` observations, else
        its own mean outcome weighted ``n / (n + min_obs)`` against the prior's value."""
        n = self.observed[j]
        if self.prior is None or n >= self.min_obs:
            return self.cond[j], self.informed[j]
        if n == 0:
            return self.prior.cond[j], self.prior.informed[j]
        weight = n / (n + self.min_obs)
        return weight * (self.own_sum[j] / n) + (1.0 - weight) * self.prior.cond[j], True

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
    one step on it (ms, at the step's position; a partial replay's cost carries the plain step the rows left
    out need), the cost of entering it this step (the reseed, or the dFlash bootstrap replay; 0 for a
    continuing record) and ``members``, the indices of the rows the replay drafts for (None = every row; a
    row left out commits one token per step from the plain decode)."""

    shape: tuple
    width: int
    cost_ms: float
    entry_ms: float = 0.0
    members: tuple | None = None
    tokens_per_row: float | None = None  # measured in serving (a calibrated mode); None = the acceptance model


@dataclasses.dataclass(frozen=True)
class RecordChoice:
    shape: tuple
    width: int
    rows: int
    expected_tokens: float  # summed over the batch's rows (one per row left out of the replay)
    cost_ms: float
    entry_ms: float
    gain: float  # tokens per ms relative to the plain step (1.0 = the same throughput), entry excluded
    reason: str  # "gain", "enter", "hold", "retry" or "probe"
    members: tuple | None = None
    measured: bool = False  # the expected tokens are serving's measured tokens per row, not the acceptance model

    @property
    def tokens_per_row(self):
        return self.expected_tokens / self.rows


def fresh_indices(fresh, n):
    """The rows a step's ``fresh`` names: ``True`` is every row (a batch that is all new), ``False``/``None``
    none, else the indices of the rows that are new requests without a trial behind them."""
    if fresh is True:
        return tuple(range(int(n)))
    if not fresh:
        return ()
    idx = tuple(sorted({int(i) for i in fresh}))
    if idx and (idx[0] < 0 or idx[-1] >= int(n)):
        raise ValueError(f"fresh rows {idx} outside the {n} rows")
    return idx


def rank_records(candidates, rows, plain_ms, optimistic=()):
    """Every candidate scored: ``[RecordChoice]`` sorted best gain first (ties: the cheaper record). Rows in
    ``optimistic`` are credited every draft (``k + 1`` tokens a step) on a record serving has not measured at
    the rung: the trial's hypothesis for new rows. A measured record (``tokens_per_row``) credits them its
    measured tokens like every other row: what its replays delivered is known, and a wave of new rows
    cannot argue with it (the 26B at eight rows: seven new rows of eight re-trialed a record measured at 1.5
    tokens a row every wave)."""
    n = len(rows)
    if n < 1:
        raise ValueError("rank_records needs at least one drafting row")
    optimistic = frozenset(fresh_indices(optimistic, n))
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
        members = None if cand.members is None else tuple(int(i) for i in cand.members)
        if members is not None and (not members or any(i < 0 or i >= n for i in members)):
            raise ValueError(f"record {cand.shape} names members {members} outside the {n} rows")
        inside = range(n) if members is None else members
        if cand.tokens_per_row is not None:
            tokens = float(cand.tokens_per_row) * len(inside) + (n - len(inside))
        else:
            hoped = sum(1 for i in inside if i in optimistic)
            tokens = sum(rows[i].expected_tokens(k_v) for i in inside if i not in optimistic) + (n - len(inside))
            tokens += hoped * (k_v + 1)
        gain = (tokens / n) / (cost / plain_ms)
        ranked.append(
            RecordChoice(
                tuple(cand.shape),
                int(cand.width),
                n,
                tokens,
                cost,
                float(cand.entry_ms),
                gain,
                "gain",
                members,
                cand.tokens_per_row is not None,
            )
        )
    ranked.sort(key=lambda c: (-c.gain, c.cost_ms, c.shape, c.members is not None))
    return ranked


class ModeStats:
    """What one mode (the plain step at a bucket, or a record at a rung) measured in serving.

    ``steps``/``tokens`` count every step served in the mode; ``busy`` the
    samples that calibrate it (every interval with a table that was not idle
    and not an outlier, dirty ones included: the mode in force pays for the
    prefill, switch, reseed or row change inside it) and ``clean`` the
    subset of them with none of those (the stats line's breakdown). A
    counted sample is kept as its ratio to the warmup table at the step's
    own position (so the position stays in the model), in a window of the
    last ``CALIB_WINDOW`` samples whose median calibrates the mode; a clean
    sample above ``OUTLIER_FACTOR`` times the running median (a wait that is
    not this mode's work) goes to ``outliers``, an idle interval (the batch
    emptied inside it) to ``idle``; both are reported, never averaged in.
    ``rows_tokens`` keeps the tokens per live row of the counted samples,
    whose mean is the measured tokens a step of the mode commits per row.
    """

    __slots__ = (
        "steps",
        "tokens",
        "clean",
        "wall_sum",
        "tokens_sum",
        "busy",
        "busy_wall",
        "busy_tokens",
        "ratios",
        "rows_tokens",
        "outliers",
        "idle",
    )

    def __init__(self):
        self.steps = 0
        self.tokens = 0
        self.clean = 0
        self.wall_sum = 0.0
        self.tokens_sum = 0
        self.busy = 0
        self.busy_wall = 0.0
        self.busy_tokens = 0
        self.ratios = collections.deque(maxlen=CALIB_WINDOW)
        self.rows_tokens = collections.deque(maxlen=CALIB_WINDOW)
        self.outliers = [0, 0.0]  # count, ms
        self.idle = [0, 0.0]  # count, ms

    def note(self, wall_ms, tokens, clean, table_ms, live_rows, idle=False, uncounted=False):
        """One step: ``wall_ms`` against ``table_ms`` (the warmup table at its position). Every interval with a
        table calibrates, dirty ones included, except an ``idle`` one (the batch emptied inside it), an
        ``uncounted`` one (the step right after a reseed: its wall holds the entry the policy charges separately,
        so it would double the step's cost) and a clean one far above the running median (an outlier)."""
        self.steps += 1
        self.tokens += int(tokens)
        if idle:
            self.idle[0] += 1
            self.idle[1] += float(wall_ms)
            return False
        if uncounted:
            return False
        if table_ms is None:
            return False
        ratio = float(wall_ms) / max(float(table_ms), 1e-9)
        if clean and len(self.ratios) >= OUTLIER_MIN_SAMPLES and ratio > OUTLIER_FACTOR * self.ratio():
            self.outliers[0] += 1
            self.outliers[1] += float(wall_ms)
            return False
        self.busy += 1
        self.busy_wall += float(wall_ms)
        self.busy_tokens += int(tokens)
        if clean:
            self.clean += 1
            self.wall_sum += float(wall_ms)
            self.tokens_sum += int(tokens)
        self.ratios.append(ratio)
        self.rows_tokens.append(float(tokens) / max(int(live_rows), 1))
        return True

    def ratio(self):
        """The median ratio of the counted walls (dirty included) to the table, or None without a sample."""
        return _median(self.ratios)

    def tokens_per_row(self):
        """The mean tokens one step of the mode committed per live row over the window (a throughput: the
        per-step counts are skewed, so never their median), or None without a sample."""
        if not self.rows_tokens:
            return None
        return sum(self.rows_tokens) / len(self.rows_tokens)

    def describe(self, name):
        """``steps, tokens, clean`` then the clean breakdown (ms/step, tokens/step, ms/token), the total of the
        counted samples (dirty included: the decision's figures) and the ratio; outliers and idle apart."""
        if self.clean:
            mean = self.wall_sum / self.clean
            tokens = self.tokens_sum / self.clean
            text = (
                f"{name}: {self.steps} steps, {self.tokens} tokens, {self.clean} clean, {mean:.1f} ms/step, "
                f"{tokens:.2f} tokens/step, {mean / max(tokens, 1e-9):.1f} ms/token, "
            )
        else:
            text = f"{name}: {self.steps} steps, {self.tokens} tokens, 0 clean, "
        if self.busy:
            text += (
                f"total {self.busy_wall / self.busy:.1f} ms/step "
                f"{self.busy_wall / max(self.busy_tokens, 1):.1f} ms/token, ratio {self.ratio():.2f}x"
            )
        else:
            text = text[:-2]
        if self.outliers[0]:
            text += f", outliers {self.outliers[0]} at {self.outliers[1] / self.outliers[0]:.1f} ms"
        if self.idle[0]:
            text += f", idle {self.idle[0]} at {self.idle[1] / self.idle[0]:.1f} ms"
        return text


def _median(values):
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return (float(values[mid - 1]) + float(values[mid])) / 2.0


def mode_name(mode):
    """``plain[b=8]`` or ``(8,3)@2048`` for the stats line."""
    if mode[0] == "plain":
        return f"plain[b={mode[1]}]"
    shape = mode[1]
    return f"({shape[0]},{shape[1]})@{mode[2]}"


class Calibration:
    """Online cost measurements that scale the warmup table once a mode has enough clean samples.

    ``modes`` keep the engine-life ``ModeStats`` per mode and ``window`` the
    ones since the last stats line; ``reseeds`` the reseed of each shape as
    served (count, ms, ratios to the table), ``dirty`` the steps whose
    interval was not a clean sample by cause (prefill, reseed, switch,
    rows, idle) and ``total`` every step's wall and tokens, both per window
    and engine life, so the figures reconcile with the served latency. The
    decision figures (``step_ms``, ``tokens_per_row``) are the modes' counted
    samples: dirty intervals included, idle ones and outliers apart.
    """

    CAUSES = ("prefill", "reseed", "switch", "rows", "untabled", "idle")

    def __init__(self, min_samples, alpha=CALIB_EWMA):
        self.min_samples = int(min_samples)
        self.alpha = float(alpha)
        if self.min_samples < 1:
            raise ValueError(f"Calibration needs min_samples >= 1, got {min_samples}")
        self.modes = {}
        self.window = {}
        self.reseeds = {}  # shape -> [count, ms_sum, ratios]
        self.window_reseeds = [0, 0.0]
        self.phases = {}  # name -> [calls, ms_sum] (engine life)
        self.window_phases = {}
        self.prefills = 0
        self.window_prefills = 0
        self.dirty = {}  # cause -> [steps, ms]
        self.window_dirty = {}
        self.total = [0, 0.0, 0]  # steps, ms, tokens
        self.window_total = [0, 0.0, 0]
        self.steps_since = {}  # mode kind ("plain" / "step") -> steps since its last clean sample
        # The measured tokens per live row the decision reads, as an EWMA (``alpha``) over the counted samples
        # of a pool: the records of one draft length at one rung share a pool (the drafter's acceptance does
        # not depend on the record's rows), the plain step at a bucket is its own. [ewma, samples]
        self.tokens_pool = {}

    def note_step(self, mode, wall_ms, tokens, clean, table_ms, live_rows=1, cause=None, idle=False):
        """One served step of ``mode`` (``("plain", bucket)`` or ``("step", shape, width)``) whose wall is
        compared to ``table_ms``, the warmup table at the step's position; an unclean step names its ``cause``,
        an ``idle`` one (every row of the previous step ended before this one) is reported and never counted."""
        mode = tuple(mode)
        if idle:
            clean, cause = False, "idle"
        # the step right after a reseed carries the entry (charged as entry_ms): it is reported, never a cost
        # sample, else a run of one-step retries measures the mode at reseed + step and declines every next one
        uncounted = cause == "reseed"
        counted = False
        for table in (self.modes, self.window):
            counted = table.setdefault(mode, ModeStats()).note(
                wall_ms, tokens, clean, table_ms, live_rows, idle, uncounted=uncounted
            )
        if counted and clean:
            # the acceptance pool is steady-state: a dirty step (the one right after a reseed, a switch, a prefill)
            # calibrates the wall but not the tokens per row, else a retry's single first step (reseed + the
            # drafts from a fresh seed) ratchets the pool down and the next retry is declined before it runs
            pool = self.tokens_pool.setdefault(self._tokens_pool_key(mode), [None, 0])
            per_row = float(tokens) / max(int(live_rows), 1)
            pool[0] = per_row if pool[0] is None else pool[0] + self.alpha * (per_row - pool[0])
            pool[1] += 1
        for total in (self.total, self.window_total):
            total[0] += 1
            total[1] += float(wall_ms)
            total[2] += int(tokens)
        if not clean:
            cause = cause if cause in self.CAUSES else "switch"
            for table in (self.dirty, self.window_dirty):
                entry = table.setdefault(cause, [0, 0.0])
                entry[0] += 1
                entry[1] += float(wall_ms)
        for kind in ("plain", "step"):
            if kind == mode[0] and clean:
                self.steps_since[kind] = 0
            else:
                self.steps_since[kind] = self.steps_since.get(kind, 0) + 1

    def note_reseed(self, shape, ms, table_ms=None):
        """A reseed (or bootstrap) of ``shape`` took ``ms``; ``table_ms`` is the warmup line at its position."""
        entry = self.reseeds.setdefault(tuple(shape), [0, 0.0, collections.deque(maxlen=CALIB_WINDOW)])
        entry[0] += 1
        entry[1] += float(ms)
        if table_ms is not None:
            entry[2].append(float(ms) / max(float(table_ms), 1e-9))
        self.window_reseeds[0] += 1
        self.window_reseeds[1] += float(ms)

    def run_length(self, shape):
        """The measured steps one entry of ``shape`` serves before the batch leaves it: the steps served on the
        shape's records (every rung) per reseed of the shape, once two reseeds were measured; else None. The
        entry horizon is capped by it (``SpecPolicy.entry_pays``): a record the batch keeps leaving after ten
        steps cannot amortize a 250 ms reseed over the configured 64 (the Galaxy c32 point: 4-9 reseeds per
        256 steps at eight rows, every entry on a prior or a window mean a few verifies then undid)."""
        shape = tuple(shape)
        entries = self.reseeds.get(shape)
        if entries is None or entries[0] < 2:
            return None
        steps = sum(stats.steps for mode, stats in self.modes.items() if mode[0] == "step" and mode[1] == shape)
        return steps / entries[0]

    def note_phase(self, name, ms):
        for table in (self.phases, self.window_phases):
            entry = table.setdefault(name, [0, 0.0])
            entry[0] += 1
            entry[1] += float(ms)

    def note_prefill(self):
        self.prefills += 1
        self.window_prefills += 1

    def calibrated(self, mode):
        stats = self.modes.get(tuple(mode))
        return stats is not None and len(stats.ratios) >= self.min_samples

    def step_ms(self, mode, prior):
        """``prior`` (the warmup table at the step's position) scaled by the mode's measured ratio once calibrated:
        the median over its counted intervals, dirty ones included (the total wall the mode in force served)."""
        stats = self.modes.get(tuple(mode))
        if stats is not None and len(stats.ratios) >= self.min_samples:
            return float(prior) * stats.ratio()
        return float(prior)

    @staticmethod
    def _tokens_pool_key(mode):
        """The pool a mode's tokens-per-row samples feed: ``("step", k, width)`` for a record (every record of
        the draft length at the rung shares it), the mode itself for a plain bucket."""
        mode = tuple(mode)
        if mode[0] == "step":
            return ("step", int(mode[1][1]), int(mode[2]))
        return mode

    def tokens_per_row(self, mode):
        """The measured tokens per row and step of ``mode`` once its pool holds ``min_samples`` counted samples,
        else None (the model decides): an EWMA (``alpha``) over the samples in the order served, so it follows
        the regime in a handful of steps (a 32-sample mean carried the random-prompt c1 point into the prose
        point that followed and kept it plain, the 12B Galaxy 2026-10-29). A record's samples are pooled over
        the records of its draft length at its rung (``("step", (rows, k), width)`` for every ``rows``): the
        drafter's acceptance does not depend on the record's rows, so a record the serving never replayed
        reads its calibrated sibling's."""
        pool = self.tokens_pool.get(self._tokens_pool_key(mode))
        if pool is not None and pool[1] >= self.min_samples:
            return pool[0]
        return None

    def reseed_ms(self, shape, prior):
        """``prior`` (the warmup line at the position) scaled by the shape's reseed ratio as served, once
        measured ``min_samples`` times."""
        entry = self.reseeds.get(tuple(shape))
        if entry is not None and len(entry[2]) >= self.min_samples:
            return float(prior) * _median(entry[2])
        return float(prior)

    def steps_without(self, kind):
        """Steps since the last clean sample of the ``kind`` ("plain" or "step") of mode."""
        return self.steps_since.get(kind, 0)

    def describe(self, window=True):
        """The stats line's body: every mode, the reseeds, the dirty steps by cause, the totals, the phases,
        the prefills, the calibrated modes."""
        table = self.window if window else self.modes
        parts = [stats.describe(mode_name(mode)) for mode, stats in sorted(table.items(), key=lambda kv: str(kv[0]))]
        if window:
            count, total_ms = self.window_reseeds
            phases, prefills, dirty, total = (
                self.window_phases,
                self.window_prefills,
                self.window_dirty,
                self.window_total,
            )
        else:
            count = sum(entry[0] for entry in self.reseeds.values())
            total_ms = sum(entry[1] for entry in self.reseeds.values())
            phases, prefills, dirty, total = self.phases, self.prefills, self.dirty, self.total
        parts.append(f"reseeds {count}" + (f" at {total_ms / count:.1f} ms" if count else ""))
        if dirty:
            steps = sum(entry[0] for entry in dirty.values())
            ms = sum(entry[1] for entry in dirty.values())
            parts.append(
                f"dirty {steps} steps, {ms:.0f} ms: "
                + ", ".join(
                    f"{cause} {dirty[cause][0]}/{dirty[cause][1]:.0f}" for cause in self.CAUSES if cause in dirty
                )
            )
        if total[0]:
            parts.append(
                f"total {total[0]} steps, {total[1]:.0f} ms, {total[2]} tokens, "
                f"{total[1] / max(total[2], 1):.1f} ms/token"
            )
        if phases:
            parts.append(
                "phases ms/call: "
                + ", ".join(f"{name} {ms / calls:.1f}" for name, (calls, ms) in phases.items() if calls)
            )
        parts.append(f"prefills {prefills}")
        calibrated = [mode_name(mode) for mode in sorted(self.modes, key=str) if self.calibrated(mode)]
        parts.append("calibrated: " + (", ".join(calibrated) if calibrated else "none"))
        return " | ".join(parts)

    def flush_window(self):
        """Close the stats window: its ``ModeStats``, reseeds, dirty steps, totals, phases and prefills start over."""
        self.window = {}
        self.window_reseeds = [0, 0.0]
        self.window_phases = {}
        self.window_prefills = 0
        self.window_dirty = {}
        self.window_total = [0, 0.0, 0]


def _smallest_k(choices, above=None):
    """The choice with the smallest draft length (above ``above`` when given), cheapest first; None if none."""
    pool = [c for c in choices if above is None or int(c.shape[1]) > int(above)]
    if not pool:
        return None
    return min(pool, key=lambda c: (int(c.shape[1]), float(c.cost_ms), tuple(c.shape)))


def _cheapest(ranked):
    """The candidate with the lowest entry plus step cost (ties by shape): what a retry or a trial replays."""
    return min(ranked, key=lambda c: (c.entry_ms + c.cost_ms, c.shape))


class SpecPolicy:
    """The step-level decision with its exploration counters and the batch prior.

    ``decide`` returns the ``RecordChoice`` to replay or ``None`` for a plain
    step. ``declined`` counts the plain steps since the last speculative
    decision (the retry's amortization base), ``below_exit`` the consecutive
    decisions below the exit gain while speculating, ``below_cap`` the
    consecutive speculative steps below the rows-based cap (the probe), and
    ``last_k`` is the draft length of the previous decision (0 for a plain
    step, None before the first), and ``trial`` says that decision was a
    retry or a new request's trial: one observation, continued only when its
    gain clears the entry threshold, never held.
    """

    def __init__(self, config, k_max):
        self.config = config
        self.k_max = int(k_max)
        self.prior = AcceptanceStats(self.k_max, config.ewma)
        self.calibration = Calibration(config.calib_min)
        self.declined = 0
        self.below_exit = 0
        self.below_cap = 0
        self.last_k = None
        self.trial = False
        self.last_reason = None  # why the last decision was what it was ("gain", ..., "exit", "calibrate", "declined")
        self.request_steps = None  # EWMA of the steps a request lived (the entry horizon of a joining row)
        self.trial_age = 0  # steps the current trial / retry has held its record
        self.since_exit = None  # plain steps since the batch left a record (the entry cooldown); None once back in

    def new_row(self):
        return self.prior.child()

    def observe(self, row, accepted, verified):
        """Record one verify's outcome on its row and on the batch prior."""
        row.observe(accepted, verified)
        self.prior.observe(accepted, verified)

    def note_plain_steps(self, n=1):
        """Count ``n`` plain steps the policy was not asked about (no session could speculate) toward the retry."""
        self.declined += int(n)
        if self.since_exit is not None:
            self.since_exit += int(n)
        self.below_cap = 0
        self.below_exit = 0
        self.last_k = 0
        self.trial = False

    def note_overtaken(self):
        """The last decision's replay was never verified: an ordinary decode the runner had in flight committed
        past it (contract s.4d). A trial does not age on such a step; the next decision is its first step."""
        if self.trial and self.trial_age > 0:
            self.trial_age -= 1

    def cooling(self):
        """Whether the batch left a record fewer than ``retry_steps`` plain steps ago: no reseeding entry yet."""
        return self.since_exit is not None and self.since_exit < self.config.retry_steps

    @property
    def speculating(self):
        return self.last_k is not None and self.last_k > 0

    def note_request_steps(self, steps):
        """A request ended after ``steps`` proposals: the entry horizon of a new request follows this EWMA."""
        steps = float(steps)
        self.request_steps = (
            steps
            if self.request_steps is None
            else self.request_steps + self.alpha_steps * (steps - self.request_steps)
        )

    alpha_steps = CALIB_EWMA

    def entry_horizon(self, remaining=None):
        """The steps an entry's saving is counted over: ``GEMMA4_SPEC_ENTRY_HORIZON`` capped by the steps the
        entering rows are expected to live (``remaining``, else the measured request length)."""
        horizon = self.config.entry_horizon
        if remaining is None:
            remaining = self.request_steps
        if remaining is not None:
            horizon = min(horizon, max(1, int(round(remaining))))
        return horizon

    def entry_pays(self, choice, plain_ms, remaining=None):
        """Whether the saving of ``choice`` over the entry horizon covers its entry cost (true when there is none);
        the horizon is capped by the shape's measured run length (``Calibration.run_length``): what an entry of
        it served before the batch left, when serving measured two or more."""
        if choice.entry_ms <= 0.0:
            return True
        saving = choice.tokens_per_row * float(plain_ms) - choice.cost_ms
        horizon = self.entry_horizon(remaining)
        run = self.calibration.run_length(choice.shape)
        if run is not None:
            horizon = min(horizon, max(1, int(round(run))))
        return horizon * saving >= choice.entry_ms

    def retry_choice(self, ranked, plain_ms):
        """The retry for a plain-serving batch, or None: the cheapest candidate (entry plus step) once at least
        ``retry_steps`` plain steps passed and its overhead over them is within ``retry_overhead``."""
        cfg = self.config
        if cfg.retry_steps <= 0 or self.declined < cfg.retry_steps:
            return None
        cheapest = _cheapest(ranked)
        if cheapest.gain < cfg.min_gain * RETRY_MIN_GAIN_FRACTION:
            return None
        overhead = (cheapest.entry_ms + cheapest.cost_ms - float(plain_ms)) / (self.declined * float(plain_ms))
        if overhead > cfg.retry_overhead:
            return None
        return dataclasses.replace(cheapest, reason="retry")

    def trial_choice(self, ranked, plain_ms=None, hypothetical=None, remaining=None):
        """A new request's one trial, or None: the cheapest candidate (entry plus step) when its gain estimate is
        at least ``trial_gain`` (``min_gain`` for a record serving has measured: its measured tokens per row are
        not argued with wave after wave), never while the batch cools after an exit, and, for a batch that is not
        all new, its hypothesis holds: ``hypothetical`` (the
        same candidates ranked with the new rows accepting every draft on a record serving has not measured at
        the rung; a measured record keeps its measured tokens per row) clears ``min_gain`` and its entry pays
        over the horizon. A batch of informed rows a new row cannot lift above the entry bar, or a record whose
        replays already measured below it, has nothing to learn from a trial that reseeds the whole batch;
        independent of the retry cadence, the reseed is its only extra cost."""
        if self.cooling():
            return None
        cheapest = _cheapest(ranked)
        bar = max(self.config.trial_gain, self.config.min_gain) if cheapest.measured else self.config.trial_gain
        if cheapest.gain < bar:
            return None
        if hypothetical is not None:
            same = (cheapest.shape, cheapest.width, cheapest.members)
            hoped = next(c for c in hypothetical if (c.shape, c.width, c.members) == same)
            if hoped.gain < self.config.min_gain or not self.entry_pays(hoped, plain_ms, remaining):
                return None
        return dataclasses.replace(cheapest, reason="trial")

    @staticmethod
    def _hypothetical(candidates, rows, plain_ms, fresh):
        """The trial's hypothesis ranking for a batch some of whose rows are new (None when none or all are)."""
        idx = fresh_indices(fresh, len(rows))
        if not idx or len(idx) == len(rows):
            return None
        return rank_records(candidates, rows, plain_ms, optimistic=idx)

    def plain_probe_due(self, choice, plain_ms):
        """Whether a speculating batch should serve one plain step to refresh the plain estimate: at least
        ``calib_probe_steps`` steps since the last clean plain sample, and the probe's cost (the reseed the
        batch pays to resume plus the tokens one plain step forgoes) within ``retry_overhead`` of them."""
        cfg = self.config
        if cfg.calib_probe_steps <= 0:
            return False
        since = self.calibration.steps_without("plain")
        if since < cfg.calib_probe_steps:
            return False
        forgone = max(0.0, choice.tokens_per_row - 1.0) * float(plain_ms)
        return choice.entry_ms + forgone <= cfg.retry_overhead * since * float(plain_ms)

    def would_speculate(self, candidates, rows, plain_ms, remaining=None, fresh=False):
        """Whether ``decide`` would replay a record now, without changing any counter (the dFlash bootstrap gate);
        ``fresh`` names the rows that are new requests without a trial yet (``True``: every row)."""
        ranked = rank_records(candidates, rows, plain_ms)
        if self.admissible(ranked, plain_ms, remaining) is not None:
            return True
        held = self.holdable(ranked)
        if self.speculating and not self.trial and held is not None and held.gain >= self.config.exit_gain:
            return True
        if self.retry_choice(ranked, plain_ms) is not None:
            return True
        if not fresh_indices(fresh, len(rows)):
            return False
        hypothetical = self._hypothetical(candidates, rows, plain_ms, fresh)
        return self.trial_choice(ranked, plain_ms, hypothetical, remaining) is not None

    def admissible(self, ranked, plain_ms, remaining=None, cooling=None):
        """The best candidate that clears the entry gain and whose entry (if any) pays over the horizon; one that
        reseeds is passed over while the batch cools after an exit (``cooling``, None = as the policy stands)."""
        if cooling is None:
            cooling = self.cooling()
        for choice in ranked:
            if choice.gain >= self.config.min_gain and self.entry_pays(choice, plain_ms, remaining):
                if cooling and choice.entry_ms > 0.0:
                    continue
                return choice
        return None

    @staticmethod
    def holdable(ranked):
        """The best candidate a speculating batch can hold without entering (no reseed this step)."""
        for choice in ranked:
            if choice.entry_ms <= 0.0:
                return choice
        return None

    def decide(self, candidates, rows, plain_ms, k_cap, remaining=None, fresh=False):
        """The record for a step of ``rows`` (their statistics) among ``candidates``, or None for plain.

        ``k_cap`` is the largest draft length the step may use (the rows-based
        cap); a choice below it counts toward the probe. ``remaining`` caps the
        entry horizon by the steps the entering rows are expected to live.
        ``fresh`` names the rows that are new requests without a trial yet
        (``True``: every row): declined otherwise, the step is their trial
        when the prior's gain clears ``trial_gain`` and, with other rows in the
        batch, the hypothesis that the new rows accept every draft clears the
        entry bar (reason "trial", never held). For ``retry_steps`` plain
        steps after an exit no entry reseeds (reason "cooldown").
        """
        cfg = self.config
        candidates = list(candidates)
        if not candidates:
            raise ValueError("decide needs at least one candidate record")
        ranked = rank_records(candidates, rows, plain_ms)
        best = self.admissible(ranked, plain_ms, remaining)
        held = self.holdable(ranked)
        choice = None
        reason = "declined"
        if best is None and self.cooling() and self.admissible(ranked, plain_ms, remaining, cooling=False) is not None:
            reason = "cooldown"
        if best is not None:
            choice = dataclasses.replace(best, reason="enter" if best.entry_ms > 0.0 else "gain")
            self.below_exit = 0
        elif self.speculating and self.trial and self.trial_age < cfg.trial_steps:
            # A trial (a retry, or a new request's first proposal) holds its record for ``trial_steps`` steps
            # before the gain decides: its first step follows the reseed and accepts few drafts, and a one-step
            # trial judged on it never refreshes the acceptance pool it is judged by (the 12B rows sweeps).
            # A trial whose rows must reseed again (its replay was overtaken by a plain step the runner had in
            # flight, or a slot moved) holds the cheapest candidate, entry included: under asynchronous
            # scheduling a retry from the plain regime is ALWAYS overtaken once (the runner drains and asks
            # again), and a trial that could not reseed never got verified.
            choice = dataclasses.replace(held if held is not None else _cheapest(ranked), reason="trial")
        elif self.speculating and not self.trial and held is not None:
            # Hysteresis: a speculating batch holds its best entry-free record above the exit gain, and
            # leaves only after EXIT_STEPS consecutive decisions below it. A retry is a trial and gets no hold.
            if held.gain >= cfg.exit_gain:
                self.below_exit = 0
                choice = dataclasses.replace(held, reason="hold")
            else:
                self.below_exit += 1
                if self.below_exit < EXIT_STEPS:
                    choice = dataclasses.replace(held, reason="hold")
                else:
                    reason = "exit"
        else:
            choice = self.retry_choice(ranked, plain_ms)
            if choice is None and fresh_indices(fresh, len(rows)):
                hypothetical = self._hypothetical(candidates, rows, plain_ms, fresh)
                choice = self.trial_choice(ranked, plain_ms, hypothetical, remaining)
        if choice is not None and choice.reason in ("gain", "hold") and self.plain_probe_due(choice, plain_ms):
            choice, reason = None, "calibrate"
        if choice is None:
            self.declined += 1
            if reason == "exit":
                self.since_exit = 0
            elif self.since_exit is not None:
                self.since_exit += 1
            self.below_cap = 0
            self.below_exit = 0
            self.last_k = 0
            self.trial = False
            self.trial_age = 0
            self.last_reason = reason
            return None
        self.last_reason = choice.reason
        self.declined = 0
        self.since_exit = None
        was_trial = self.trial
        self.trial = choice.reason in ("retry", "trial")
        self.trial_age = self.trial_age + 1 if (self.trial and was_trial) else (1 if self.trial else 0)
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
    """One line for the warmup log: the plain step (per decode bucket when a table) and every record's step and
    reseed cost lines (ms; the step and the reseed timed with every row of the shape live, a bracketed ``1 row``
    line after a step where the warmup found a one-row replay cheaper: the partial replay's cost)."""
    if plain is None:
        parts = ["plain unmeasured"]
    elif isinstance(plain, dict):
        parts = ["plain " + ", ".join(f"b{b} {CostLine.of(line).describe()}" for b, line in sorted(plain.items()))]
    else:
        parts = [f"plain {CostLine.of(plain).describe()}"]
    shapes = sorted({key[1] for key in costs})
    for shape in shapes:
        steps = sorted((key[2], value) for key, value in costs.items() if key[0] == "step" and key[1] == shape)
        reseed = costs.get(("reseed", shape))
        text = ", ".join(
            f"S_k {w} {CostLine.of(value).describe()}"
            + (
                f" [1 row {CostLine.of(costs[('partial', shape, w)]).describe()}]"
                if ("partial", shape, w) in costs
                else ""
            )
            for w, value in steps
        )
        if reseed is not None:
            reseed_text = f"reseed {CostLine.of(reseed).describe()}"
            text = f"{text}; {reseed_text}" if text else reseed_text
        parts.append(f"{shape}: {text}")
    return " | ".join(parts)
