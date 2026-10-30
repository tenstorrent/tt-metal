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
``n / (n + others)`` of its own mean outcome and the rest of the prior,
``others`` being the prior's observations beyond the row's own (which
feed the prior too), at most ``ROW_MIN_OBSERVATIONS`` (one verify at 3-5
accepted moves a row most of the way; with no other evidence the row's
outcome counts alone), and from ``ROW_MIN_OBSERVATIONS`` on its own
EWMA. The prior starts
optimistic (every ``c = 1``), a default and not evidence: the first step of
a server is a one-step trial at the largest record (never an entry), and
its verify decides. A request that stays plain verifies nothing
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

and, unless serving has measured the record, only on informed statistics:
the optimistic default prior credits every draft, and a record it alone
recommends gets a one-step trial instead (the 12B QB2 entered its 256K
record on the prior at an expected 6 tokens a row, served 20 steps at
0.8-0.9x and left). Once speculating (no reseed this step) the best record
is *held* while its gain stays at or above ``GEMMA4_SPEC_EXIT_GAIN``, and
the batch leaves for the plain step only after ``EXIT_STEPS`` consecutive
decisions below it, never on one noisy step. While the batch holds a record
its credit is what the current run delivered (``Calibration``: the mean
tokens per row of the run's own steps, not the pool of earlier runs and not
the rows' per-position model, which is mostly the prior after a few
verifies), so the hold does not end on the noise of its first steps. The
decision is per step for the whole batch: the fused replay evaluates every
row's ``K_v`` columns whatever a row offers, so a zero-draft row inside the
replay saves nothing, and a row left out of it costs an extra plain decode.

Retrying
--------

While the plain step serves, no draft is verified and the statistics do not
move. Every reseeding entry out of the plain regime -- an enter on the
credit, a new request's trial, a retry -- is first priced as served and then
admitted by its rung's *exploration budget*. At a batch's first proposal
(every row at its first step) the entry costs the reseed and its replay is
verified; mid-request, under asynchronous scheduling, the plain step the
runner has in flight overtakes the replay, which takes only when every
row's first draft matches the token it committed, so the entry costs the
retry cycle (the reseed interval as measured per shape, else the table's
entry plus step, and the reload of the plain step after a record step, each
less the plain step it stands in for) times the reseeds expected before it
takes, ``1 / prod_r c_r[1]`` (an eight-row prose wave at 0.8 a row: six;
the 12B QB2 prose c8 re-entered at the tail of every wave and every replay
missed on some row). ``entry_pays`` reads that cost. The budget per rung
(verify width) keeps a *clock*, the plain steps served at the rung since
its last failed run (a new request does not reset it), and a *debt*, the
realized loss of the failed runs there since the rung last paid (a run's
charged costs -- the entry, every step -- less what its tokens would have
cost plain, plus the reload; a paying run clears it). An entry is admitted
when

    debt + expected loss of the entry <= GEMMA4_SPEC_RETRY_OVERHEAD * clock * plain

where the expected loss is the entry cost plus ``GEMMA4_SPEC_TRIAL_STEPS``
steps (one from an uninformed prior) at the record's cost over the credit
the rung has: the rung's measured credit, else the freshest pool of the
draft length at any rung (the acceptance is the text's, not the rung's),
else every draft rejected -- never the per-position prior, which two
11-token coherence requests left at 1.0 on the 12B QB2 and whose 256K
record then served 0.63x for 20 steps. So a rung never measured (the long
rungs' first visit) waits until its trial, priced with every draft
rejected, is within the bound (the 256K rung: ~1000 plain steps at 1 %, so
two 128-token requests cost nothing over plain), a failed trial on random
text charges its loss and the next entry there waits order 1000 plain
tokens a row, and a record once profitable that fails on the current text
re-enters no sooner than any other. Outside the budget: an entry on the
rung's measured credit while the rung has no debt (``measured_pays``:
serving measured the record at break-even or better, its last run did not
fail, its entry as served pays over the horizon -- the prose c1 re-entered
on it every request), a speculating batch's switch, the plain probe's
resume, and an entry-free record. A retry needs at least
``GEMMA4_SPEC_RETRY_STEPS`` plain steps since the last speculative one and
never reseeds while the batch cools after an exit: the best record
``measured_pays`` admits (a re-entry on the pool's credit: the trial that
follows measures it again before the hold), else the cheapest candidate
(entry plus step) as exploration under the budget; a candidate the
statistics already rule out (gain under ``min_gain / 4``) is not worth a
replay. On the MTP rail a declined row reseeds and resumes within the
request; on the dFlash rail a plain step gives the drafter no residual tap,
so a decline ends the request's session and the retry lands on the first
step of a later request (``note_plain_steps`` counts the plain steps in
between, on every rung's clock). A new request gets one *trial* at its
first proposal: the cheapest candidate replays once when its gain estimate
clears ``GEMMA4_SPEC_TRIAL_GAIN`` -- ``GEMMA4_SPEC_MIN_GAIN`` once serving
has ``calib_min`` samples of the record at the rung, fresh or stale: its
measured tokens per row are not argued with wave after wave, and the prior
the model falls back to on a stale pool was learned from the same text --
and its reseed pays over the horizon at that estimate. A prior below the
floor is the random-prompt regime learned on this engine, and the request
stays plain. In a batch that is not all new the trial also needs its
hypothesis (the new rows accept every draft on an unmeasured record) to
clear the entry bar, since it reseeds every row. A trial or retry holds its
record for ``trial_steps`` steps (one from an uninformed prior) and then
continues under the hold rule on what those steps delivered: the entry bar
covered the reseed it already paid, and a record at 1.0-1.25x measured was
dropped for 240 plain steps a request on the 12B QB2 prose c1. A trial whose
replay the runner overtook and whose first draft missed the committed token
ends there (one reseed per trial); a match seeds the next replay without one.
A batch that just left a record waits ``retry_steps`` plain steps before any
reseeding entry (an enter, a trial or a retry): those waves ran
enter-exit-enter cycles a few steps apart. A new batch (every row at its
first proposal) ends its predecessor's run: read on, that run's window held
the predecessor's tail and the new request's first proposal declined on it,
then re-entered a step later through an overtaken reseed. While speculating
below the rows-based cap, after ``GEMMA4_SPEC_PROBE_STEPS`` consecutive such
steps one step is replayed at the next larger draft length (no reseed), so
deeper positions are re-observed and the larger record's pool is fed (``0``
disables the probe).

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
it on the model alone. Every counted step feeds the pool except the one
right after a reseed (a fresh seed accepts few), a switch included: the
replay continues the seed, and the probe of the next draft length is such
a step. While a run serves, its own steps are the credit (the last
``CALIB_WINDOW`` of them); between runs the whole mean of the last run when
it had ``calib_min`` samples, else the last ``CALIB_WINDOW`` samples across
runs; a pool no step fed for ``GEMMA4_SPEC_POOL_STALE_STEPS`` intervals is
stale and the model decides. The decision reads the TOTAL wall of the mode in
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
(``note_request_steps``) and by the record's measured run length (the
longest of its last ``RUN_LENGTH_RUNS`` runs; a declined trial is none): a
short request cannot amortize a reseed a long one can, nor can a record the
batch keeps leaving. While a record serves, after
``GEMMA4_SPEC_CALIB_PROBE_STEPS`` steps without a clean plain sample one
plain step (overhead-gated like the retry) refreshes the plain estimate and
the hold resumes its record, reseeding, under the hold rule: the probe is
not an exit.

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
SPEC_RETRY_OVERHEAD_DEFAULT = "0.01"
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
SPEC_POOL_STALE_ENV = "GEMMA4_SPEC_POOL_STALE_STEPS"
SPEC_POOL_STALE_DEFAULT = "128"
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
    SPEC_POOL_STALE_ENV,
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
#: Runs of a shape (steps between its reseeds) the entry horizon's run-length cap looks back over.
RUN_LENGTH_RUNS = 4


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
    retry_overhead: float = 0.01
    probe_steps: int = 32
    ewma: float = 0.1
    plain_step_ms: tuple | None = None
    stats_steps: int = 256
    calib_min: int = 8
    calib_probe_steps: int = 256
    trial_gain: float = 0.6
    trial_steps: int = 3
    pool_stale: int = 128

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
            pool_stale=_env_int(env, SPEC_POOL_STALE_ENV, SPEC_POOL_STALE_DEFAULT, low=1),
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
    accepted drafts per speculative step, ``spec_steps`` the steps
    observed and ``steps`` the decisions the row has been part of (0 at
    its first proposal: an entry there is not overtaken by a plain step
    the runner has in flight, one later is).
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
        "steps",
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
        self.steps = 0

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
        its own mean outcome weighted ``n / (n + others)`` against the prior's value, where ``others`` is the
        prior's evidence beyond this row's own observations (which feed the prior too), at most ``min_obs``;
        with none the row's outcome counts alone. The prior's value at a position only this row observed is
        the optimistic default, not evidence: weighted three to one it read a rejected first draft as 0.68
        and held a 256K record three more steps (2026-10-30)."""
        n = self.observed[j]
        if self.prior is None or n >= self.min_obs:
            return self.cond[j], self.informed[j]
        if n == 0:
            return self.prior.cond[j], self.prior.informed[j]
        others = min(self.min_obs, max(0, self.prior.observed[j] - n))
        if others == 0:
            return self.own_sum[j] / n, True
        weight = n / (n + others)
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
    run: bool = False  # the credit is the current run's own steps (set by the policy): no per-row bound


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
    tokens a row every wave). A row with verifies of its own is credited at most what its observed acceptance
    predicts: a pool is an EWMA of the steps last served on the record, and the row says what it accepts now.
    """
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
            # the measured credit, bounded per row by what the row's OWN verifies say it accepts; a row without
            # a verify of its own (the batch prior only) takes the measurement as it stands, and so does every
            # row when the credit is the current run's (``run``): the run is the rows' own measurement, and the
            # per-position EWMA that would bound it is still mostly the prior after a few verifies (an eight-row
            # prose wave read 1.4 tokens a row against a run at 3.0 and left, 2026-10-30 replay)
            m = float(cand.tokens_per_row)
            tokens = sum(
                min(m, rows[i].expected_tokens(k_v)) if (rows[i].ewma_accepted is not None and not cand.run) else m
                for i in inside
            ) + (n - len(inside))
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

    def __init__(self, min_samples, alpha=CALIB_EWMA, pool_stale=128):
        self.min_samples = int(min_samples)
        self.alpha = float(alpha)
        self.pool_stale = int(pool_stale)
        self.intervals = 0  # every step noted (the clock a pool's freshness is read against)
        if self.min_samples < 1:
            raise ValueError(f"Calibration needs min_samples >= 1, got {min_samples}")
        self.modes = {}
        self.window = {}
        self.reseeds = {}  # shape -> [count, ms_sum, ratios]
        self.shape_runs = {}  # shape -> [steps since its last reseed, the lengths of its last runs]
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
        # The measured tokens per live row the decision reads: the records of one draft length at one rung
        # share a pool (the drafter's acceptance does not depend on the record's rows), the plain step at a
        # bucket is its own. [samples (the last CALIB_WINDOW, across runs), count, interval of the last sample]:
        # the credit between runs is the samples' mean, the count every clean step ever pooled, and a pool no
        # step fed for ``pool_stale`` intervals is stale and the model decides again.
        self.tokens_pool = {}
        # The current run: per pool the samples since the batch entered a record (the last CALIB_WINDOW of them,
        # their count and their sum); the decision reads the window's mean while the run serves (``end_run``
        # closes it and keeps each pool's whole-run mean in ``last_run``: a run long enough to measure prices
        # the next entry, and one that failed says the pool's older samples describe another text).
        self.run = {}
        self.last_run = {}
        # The walls a retry cycle pays under asynchronous scheduling, as served: per shape the interval that
        # starts with a reseed (the reseed and the replay the in-flight plain step overtakes), and the plain
        # step right after a record step (the runner's reload and the drained pipeline bubble); [count, ms].
        self.reseed_steps = {}
        self.after_record = [0, 0.0]
        # The tokens per live row every noted interval committed (idle ones apart), summed over the engine life:
        # the policy reads its growth over a run to price what the run delivered against the plain step.
        self.tokens_rows_sum = 0.0

    def note_step(self, mode, wall_ms, tokens, clean, table_ms, live_rows=1, cause=None, idle=False):
        """One served step of ``mode`` (``("plain", bucket)`` or ``("step", shape, width)``) whose wall is
        compared to ``table_ms``, the warmup table at the step's position; an unclean step names its ``cause``,
        an ``idle`` one (every row of the previous step ended before this one) is reported and never counted."""
        mode = tuple(mode)
        self.intervals += 1
        if idle:
            clean, cause = False, "idle"
        else:
            self.tokens_rows_sum += float(tokens) / max(int(live_rows), 1)
        probe = mode[0] == "plain" and cause == "switch"
        # A plain step right after a record step (the probe a speculating batch serves, or the first step after an
        # exit) carries the runner's input reload and the drained pipeline bubble: 44-49 ms against a 35 ms plain
        # step on the 31B QB2. Counted, it inflated the plain estimate in every enter/exit churn and the record then
        # read 1.75x instead of 2.5x (9 spurious entries in a 40 s window, 2026-10-30). It is reported, never a cost
        # sample; it does end the probe cadence below (a probe did run), else the probe recurred at the overhead
        # bound (the 26B QB2 served 16 in one 30 s window instead of one per 256 steps).
        # the step right after a reseed carries the entry (charged as entry_ms): it is reported, never a cost
        # sample, else a run of one-step retries measures the mode at reseed + step and declines every next one
        uncounted = cause == "reseed" or probe
        if mode[0] == "step":
            self.shape_runs.setdefault(tuple(mode[1]), [0, collections.deque(maxlen=RUN_LENGTH_RUNS)])[0] += 1
        if cause == "reseed" and mode[0] == "step":
            entry = self.reseed_steps.setdefault(tuple(mode[1]), [0, 0.0])
            entry[0] += 1
            entry[1] += float(wall_ms)
        if probe:
            self.after_record[0] += 1
            self.after_record[1] += float(wall_ms)
        counted = False
        for table in (self.modes, self.window):
            counted = table.setdefault(mode, ModeStats()).note(
                wall_ms, tokens, clean, table_ms, live_rows, idle, uncounted=uncounted
            )
        if counted:
            # the acceptance pool is steady-state: the step right after a reseed (uncounted above) drafts from a
            # fresh seed and accepts few, and a retry's single first step ratcheted the pool down so the next
            # retry was declined before it ran; a switch or a prefill inside the interval moves the wall, not
            # the acceptance (the replay continues the seed), so those steps feed the pool -- the probe of the
            # next draft length is one such step, and a pool it never fed kept its low reading from an earlier
            # run and lost the ranking to its sibling for the rest of the request (2026-10-30 replay)
            key = self._tokens_pool_key(mode)
            per_row = float(tokens) / max(int(live_rows), 1)
            pool = self.tokens_pool.setdefault(key, [collections.deque(maxlen=CALIB_WINDOW), 0, 0])
            pool[0].append(per_row)
            pool[1] += 1
            pool[2] = self.intervals
            if mode[0] == "step":
                run = self.run.setdefault(key, [collections.deque(maxlen=CALIB_WINDOW), 0, 0.0])
                run[0].append(per_row)
                run[1] += 1
                run[2] += per_row
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
            if kind == mode[0] and (clean or probe):
                self.steps_since[kind] = 0
            else:
                self.steps_since[kind] = self.steps_since.get(kind, 0) + 1

    def end_run(self, trial=False):
        """The batch left its record (or reseeds into another one): the run's samples stop deciding, each pool
        the run fed keeps the run's mean as its last run, and the steps served on each shape since the last run
        ended are one run of the shape for ``run_length`` -- unless the run was a ``trial`` that failed: a
        declined trial is exploration, not a record the batch kept leaving."""
        for key, (_window, count, total) in self.run.items():
            self.last_run[key] = (total / count, count)
        self.run = {}
        for runs in self.shape_runs.values():
            if runs[0] > 0 and not trial:
                runs[1].append(runs[0])
            runs[0] = 0

    def last_run_mean(self, mode):
        """The mean tokens per row over the whole of the last run that fed ``mode``'s pool, or None."""
        last = self.last_run.get(self._tokens_pool_key(mode))
        return None if last is None else last[0]

    def pool_credit(self, mode):
        """``(credit, samples)`` of ``mode``'s pool as the stats line shows it: the mean of its last samples
        (None without one) and the clean steps ever pooled."""
        pool = self.tokens_pool.get(self._tokens_pool_key(mode))
        if pool is None or not pool[0]:
            return None, 0 if pool is None else pool[1]
        return sum(pool[0]) / len(pool[0]), pool[1]

    def run_samples(self, mode):
        """Clean samples the current run has of ``mode``'s pool."""
        run = self.run.get(self._tokens_pool_key(mode))
        return 0 if run is None else run[1]

    def pool_known(self, mode):
        """Whether serving has ``min_samples`` clean samples of ``mode``'s pool at all, fresh or stale: a record
        serving has measured gets no trial below the entry bar, however long ago."""
        pool = self.tokens_pool.get(self._tokens_pool_key(mode))
        return pool is not None and pool[1] >= self.min_samples

    def reseed_step_ms(self, shape):
        """The mean wall of the intervals that started with a reseed of ``shape`` (the reseed and its replay,
        overtaken by the in-flight plain step under asynchronous scheduling), or None before one was served."""
        entry = self.reseed_steps.get(tuple(shape))
        return entry[1] / entry[0] if entry and entry[0] else None

    def after_record_ms(self):
        """The mean wall of the plain step right after a record step, or None before one was served."""
        return self.after_record[1] / self.after_record[0] if self.after_record[0] else None

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
        """The measured steps one entry of ``shape`` serves before the batch leaves it: the longest of the
        shape's last ``RUN_LENGTH_RUNS`` runs (the steps served on its records, every rung, from one
        ``end_run`` to the next; a declined trial is none), or None before one ended. The entry horizon is
        capped by it (``SpecPolicy.entry_pays``): a record the batch keeps leaving after ten steps cannot
        amortize a 250 ms reseed over the configured 64 (the Galaxy c32 point: 4-9 reseeds per 256 steps at
        eight rows, every entry on a prior or a window mean a few verifies then undid). The last runs, not the
        engine-life steps per reseed: the one-step trials of a random-prompt point pinned the 12B QB2's (8,3)
        at one step a reseed, and the prose point after it could not enter a record measured at 1.4x because
        no entry could lengthen the run (2026-10-30 replay)."""
        runs = self.shape_runs.get(tuple(shape))
        if runs is None or not runs[1]:
            return None
        return float(max(runs[1]))

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
        the median over its counted intervals, dirty ones included (the total wall the mode in force served). A
        record not yet calibrated at a rung where sibling records are reads their median ratio: the serving
        path's overhead over the table is the rung's, not the shape's, and an uncalibrated sibling at the table
        (1.0x) outranked the calibrated record (1.3x) on cost alone -- the 12B QB2 random c1 switched (1,3) to
        (5,3) to (8,3) at one row and served 16 steps at 1.33 tokens a step on them (2026-10-30)."""
        mode = tuple(mode)
        stats = self.modes.get(mode)
        if stats is not None and len(stats.ratios) >= self.min_samples:
            return float(prior) * stats.ratio()
        if mode[0] == "step":
            siblings = [
                s.ratio()
                for m, s in self.modes.items()
                if m[0] == "step" and m[2] == mode[2] and len(s.ratios) >= self.min_samples
            ]
            if siblings:
                return float(prior) * _median(siblings)
        return float(prior)

    def pool_credit_any(self, k):
        """The credit of the freshest known pool of draft length ``k`` at any rung (the last run's mean when it
        had ``min_samples`` samples, else the pool's window mean), or None: a rung never measured prices its first
        trial on what the drafter delivered on this text elsewhere, since its acceptance does not depend on the
        rung."""
        best = None
        for key, pool in self.tokens_pool.items():
            if key[0] != "step" or int(key[1]) != int(k) or not pool[0] or pool[1] < self.min_samples:
                continue
            if self.intervals - pool[2] > self.pool_stale:
                continue
            if best is None or pool[2] > best[1][2]:
                best = (key, pool)
        if best is None:
            return None
        key, pool = best
        last = self.last_run.get(key)
        if last is not None and last[1] >= self.min_samples:
            return last[0]
        return sum(pool[0]) / len(pool[0])

    @staticmethod
    def _tokens_pool_key(mode):
        """The pool a mode's tokens-per-row samples feed: ``("step", k, width)`` for a record (every record of
        the draft length at the rung shares it), the mode itself for a plain bucket."""
        mode = tuple(mode)
        if mode[0] == "step":
            return ("step", int(mode[1][1]), int(mode[2]))
        return mode

    def tokens_per_row(self, mode):
        """The measured tokens per row and step of ``mode``, or None (the model decides). While the batch serves
        a record, the mean of the last ``CALIB_WINDOW`` counted steps of the current run on ``mode``'s pool (one
        sample is enough: the run says what the record delivers now, and the window bounds how long a text
        that stopped accepting is held); between runs the pool's credit, once it holds ``min_samples`` samples
        in all. A record's samples are pooled over the
        records of its draft length at its rung (``("step", (rows, k), width)`` for every ``rows``): the
        drafter's acceptance does not depend on the record's rows, so a record the serving never replayed
        reads its calibrated sibling's. Means, not an EWMA over the last samples: the steps before an exit
        are the low ones that caused it, and an EWMA of them read the 12B QB2's (1,5) record at 2.2-2.8 tokens
        a row (0.86-1.1x) after runs that averaged 3.3-4.2 (2026-10-30 prose c1), so no request after the
        first re-entered. Between runs the credit is the whole mean of the last run when that run had
        ``min_samples`` samples (a run long enough to measure speaks for itself), else the mean of the pool's
        last ``CALIB_WINDOW`` samples across runs: one short run at the tail of a request (a single one-token
        step) does not price the next entry, and an eight-row wave's tail, which is what a window of the last
        samples holds when the wave ends, does not either (2026-10-30 replay). A pool no step
        fed for ``pool_stale`` intervals is stale (None): a stale reading ranked the 31B QB2's K=5 record
        (1.54 a row, frozen) under its K=3 sibling (3.55) for the rest of the server's life while the rows'
        observed acceptance said 3.3 (2026-10-29 sweep)."""
        key = self._tokens_pool_key(mode)
        run = self.run.get(key)
        if run is not None and run[1] > 0:
            return sum(run[0]) / len(run[0])
        pool = self.tokens_pool.get(key)
        if pool is not None and pool[0] and pool[1] >= self.min_samples:
            if self.intervals - pool[2] <= self.pool_stale:
                last = self.last_run.get(key)
                if last is not None and last[1] >= self.min_samples:
                    return last[0]  # a run long enough to measure speaks for itself
                return sum(pool[0]) / len(pool[0])
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
    retry or a new request's trial: held for ``trial_limit`` steps, then
    continued under the hold rule on what its own steps delivered.
    """

    def __init__(self, config, k_max):
        self.config = config
        self.k_max = int(k_max)
        self.prior = AcceptanceStats(self.k_max, config.ewma)
        self.calibration = Calibration(config.calib_min, pool_stale=config.pool_stale)
        self.declined = 0
        self.below_exit = 0
        self.below_cap = 0
        self.last_k = None
        self.trial = False
        self.last_reason = None  # why the last decision was what it was ("gain", ..., "exit", "calibrate", "declined")
        self.request_steps = None  # EWMA of the steps a request lived (the entry horizon of a joining row)
        self.trial_age = 0  # steps the current trial / retry has held its record
        self.trial_limit = int(config.trial_steps)  # steps the current trial holds (one from an uninformed prior)
        self.since_exit = None  # plain steps since the batch left a record (the entry cooldown); None once back in
        self.probing = False  # the last decision was the plain probe of a hold: the next resumes the record
        # The exploration budget per rung (verify width): ``rung_clock`` counts the plain steps served at the rung
        # since its last failed run (never reset by a request), ``rung_debt`` the realized loss (ms) of the failed
        # runs there since its last paying one. A run is the steps from an entry out of the plain regime to the
        # exit: ``run_cost`` sums the costs the decisions charged (the entry, every step), ``run_tokens_at`` is the
        # calibration's tokens-per-row sum when it began.
        self.rung_clock = {}
        self.rung_debt = {}
        self.run_open = False
        self.run_width = None
        self.run_cost = 0.0
        self.run_tokens_at = 0.0
        self.run_intervals_at = 0
        self.pending_entry_ms = 0.0  # the entry would_speculate admitted, charged to the run decide opens
        self._at_start = True  # the step being decided is every row's first proposal (no plain step in flight)
        self._attempts = 1.0  # expected reseeds before a mid-request entry's first draft matches on every row
        self._plain_ms = None  # the plain step of the last decision (prices a run ``note_plain_steps`` closes)

    def new_row(self):
        return self.prior.child()

    def observe(self, row, accepted, verified):
        """Record one verify's outcome on its row and on the batch prior."""
        row.observe(accepted, verified)
        self.prior.observe(accepted, verified)

    def note_plain_steps(self, n=1):
        """Count ``n`` plain steps the policy was not asked about (no session could speculate) toward the retry
        and every rung's exploration clock."""
        self.declined += int(n)
        if self.since_exit is not None:
            self.since_exit += int(n)
        if self.speculating or self.probing:
            self.calibration.end_run(trial=self.trial)
            self._close_run(self._plain_ms)
        for width in self.rung_clock:
            self.rung_clock[width] += int(n)
        self.below_cap = 0
        self.below_exit = 0
        self.last_k = 0
        self.trial = False
        self.probing = False

    # -- the exploration budget -------------------------------------------------------------------------------
    def _frame(self, rows, fresh):
        """Fix, for the decision at hand, whether it is every row's first proposal (``_at_start``: no plain step
        of the batch is in flight, so a reseed's replay is verified) and otherwise the expected number of reseeds
        a mid-request entry pays before its first draft matches the token the in-flight plain step committed on
        every row (``_attempts``: one over the product of the rows' first-position acceptance; the prose c8 of
        the 12B QB2 re-entered its (8,3) at the tail of every wave and every replay was overtaken and missed on
        some row, 2026-10-30)."""
        n = len(rows)
        idx = fresh_indices(fresh, n)
        self._at_start = len(idx) == n and all(int(getattr(r, "steps", 0)) == 0 for r in rows)
        if self._at_start:
            self._attempts = 1.0
            return
        p = 1.0
        for r in rows:
            p *= float(r.effective_cond(1)[0]) if r.k_max >= 1 else 1.0
        self._attempts = 1.0 / p if p > 1e-6 else 1e6

    def entry_cost(self, choice, plain_ms):
        """What entering ``choice`` from the plain regime costs beyond the steps it serves, as served: at a
        batch's first proposal the reseed (its replay is verified); mid-request the retry cycle (the reseed and
        the replay the in-flight plain step overtakes, the reload after) times the reseeds expected before a
        first draft matches on every row. Nothing for a continuing record."""
        if choice.entry_ms <= 0.0:
            return 0.0
        if self._at_start:
            return float(choice.entry_ms)
        return self._attempts * self.retry_cycle_ms(choice, plain_ms)

    def _credit_for(self, choice):
        """The tokens per row an exploration of ``choice`` is expected to deliver: the measured credit of its
        rung's pool, else of the freshest pool of its draft length at any rung (the acceptance is the text's,
        not the rung's), else one (every draft rejected): the per-position prior is not read here -- the 12B
        QB2's 256K record was entered at 1.7x on a prior two 11-token coherence requests had left at 1.0 and
        served 0.63x (2026-10-30)."""
        if choice.measured:
            return float(choice.tokens_per_row)
        credit = self.calibration.pool_credit_any(int(choice.shape[1]))
        return 1.0 if credit is None else float(credit)

    def explore_cost(self, choice, plain_ms, steps):
        """The expected loss of entering ``choice`` from the plain regime and holding it ``steps`` steps on the
        credit the rung has (``_credit_for``): the entry cost, each step's cost over what its tokens would cost
        plain, and at a first proposal the reload of the plain step after (mid-request the cycle carries it)."""
        plain_ms = float(plain_ms)
        credit = self._credit_for(choice)
        loss = max(0.0, float(choice.cost_ms) - credit * plain_ms)
        cost = self.entry_cost(choice, plain_ms) + int(steps) * loss
        if self._at_start:
            after = self.calibration.after_record_ms()
            cost += max(0.0, after - plain_ms) if after is not None else 0.0
        return cost

    def budget_ok(self, choice, plain_ms, steps):
        """Whether the rung's exploration budget admits entering ``choice`` from the plain regime: its debt (the
        realized loss of the failed runs since the rung last paid) plus this entry's expected loss within
        ``GEMMA4_SPEC_RETRY_OVERHEAD`` of the plain time the rung served since it last failed."""
        width = int(choice.width)
        clock = self.rung_clock.get(width, 0)
        debt = self.rung_debt.get(width, 0.0)
        return debt + self.explore_cost(choice, plain_ms, steps) <= self.config.retry_overhead * clock * float(plain_ms)

    def _tick_clocks(self, candidates):
        """A plain step at the rung(s) of ``candidates``: their exploration clocks advance."""
        for width in {int(c.width) for c in candidates}:
            self.rung_clock[width] = self.rung_clock.get(width, 0) + 1

    def _open_run(self, choice):
        self.run_open = True
        self.run_width = int(choice.width)
        self.run_cost = 0.0
        self.run_tokens_at = self.calibration.tokens_rows_sum
        self.run_intervals_at = self.calibration.intervals

    def _close_run(self, plain_ms):
        """The run ends: its charged costs against what its tokens would have cost plain (plus the reload of the
        plain step after a record step, as measured) is its realized loss. A failed run adds the loss to its
        rung's debt and restarts the rung's clock; a paying run clears the debt. A run the calibration saw no
        interval of (nothing served between the decisions) measures nothing."""
        if not self.run_open:
            return
        self.run_open = False
        if plain_ms is None or self.calibration.intervals == self.run_intervals_at:
            return
        plain_ms = float(plain_ms)
        tokens = self.calibration.tokens_rows_sum - self.run_tokens_at
        after = self.calibration.after_record_ms()
        reload = max(0.0, after - plain_ms) if after is not None else 0.0
        loss = self.run_cost - tokens * plain_ms + reload
        width = self.run_width
        if loss > 0.0:
            self.rung_debt[width] = self.rung_debt.get(width, 0.0) + loss
            self.rung_clock[width] = 0
        else:
            self.rung_debt[width] = 0.0

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

    def entry_pays(self, choice, plain_ms, remaining=None, entry=None):
        """Whether the saving of ``choice`` over the entry horizon covers its entry cost (true when there is none):
        ``entry`` when given, else the entry as served (``entry_cost``: the reseed at a first proposal, the retry
        cycle times the expected reseeds mid-request); the horizon is capped by the shape's measured run length
        (``Calibration.run_length``): what an entry of it served before the batch left, when serving measured
        two or more."""
        if entry is None:
            entry = self.entry_cost(choice, plain_ms)
        if entry <= 0.0:
            return True
        saving = choice.tokens_per_row * float(plain_ms) - choice.cost_ms
        horizon = self.entry_horizon(remaining)
        run = self.calibration.run_length(choice.shape)
        if run is not None:
            horizon = min(horizon, max(1, int(round(run))))
        return horizon * saving >= entry

    @staticmethod
    def informed(rows):
        """Whether any row's statistics rest on an observation (its own or the prior's): the optimistic default
        prior credits every draft, and a record it alone recommends is not entered, only trialed for one step."""
        return any(any(s.informed) for s in rows)

    def retry_cycle_ms(self, choice, plain_ms):
        """What one retry of ``choice`` costs over the plain steps it replaces, as served under asynchronous
        scheduling: the interval that starts with its reseed (the reseed and the replay the in-flight plain
        step overtakes, measured per shape, else the table's entry plus step) and the plain step after a record
        step (the runner's reload; measured, else nothing), each less the plain step they stand in for."""
        plain_ms = float(plain_ms)
        reseed_step = self.calibration.reseed_step_ms(choice.shape)
        if reseed_step is None:
            reseed_step = choice.entry_ms + choice.cost_ms
        after = self.calibration.after_record_ms()
        if after is None:
            after = plain_ms
        return max(0.0, reseed_step - plain_ms) + max(0.0, after - plain_ms)

    def retry_choice(self, ranked, plain_ms):
        """The retry for a plain-serving batch, or None, once at least ``retry_steps`` plain steps passed: the
        best record serving has measured at break-even or better whose retry cycle (times the reseeds expected
        before a mid-request replay takes) pays over the entry horizon (a re-entry on the pool's credit: the
        trial that follows measures it again before the hold), else the cheapest candidate (entry plus step)
        as exploration; either way the rung's budget (``budget_ok``, applied by ``decide``) bounds it."""
        cfg = self.config
        if cfg.retry_steps <= 0 or self.declined < cfg.retry_steps:
            return None
        plain_ms = float(plain_ms)
        cooling = self.cooling()  # a batch that just left a record reseeds into nothing until the cooldown passed
        # A record whose last steps paid for themselves re-enters as soon as its cycle (the reseed, the overtaken
        # replay and the reload) pays over the horizon, capped by its measured run length: marginal random
        # text (runs of a few steps) cannot amortize it, prose (runs of a request) can at once. The 12B QB2
        # prose c1 waited 337 plain steps after a 1.1x run under the overhead bound alone (2026-10-30).
        for choice in ranked:
            if choice.measured and choice.gain >= 1.0:
                # a pool whose last run failed describes the text before it (prose, then random prompts): the
                # exploration below decides, not the older samples
                if self.measured_pays(choice, plain_ms) and not (cooling and choice.entry_ms > 0.0):
                    return dataclasses.replace(choice, reason="retry")
                break
        cheapest = _cheapest(ranked)
        if cheapest.gain < cfg.min_gain * RETRY_MIN_GAIN_FRACTION or (cooling and cheapest.entry_ms > 0.0):
            return None
        # Otherwise a retry is exploration, within the rung's budget: its whole cycle as served (the reseed, the
        # replay the plain step in flight overtakes, the trial steps, the reload), times the reseeds expected
        # before it takes, against GEMMA4_SPEC_RETRY_OVERHEAD of the plain time the rung served since its last
        # failed run. Charged as one table step at 3 % of the plain steps since the last record step, the cadence
        # re-entered four times inside eight 128-token random requests on the 12B QB2 (2026-10-30, -3.5 %).
        return dataclasses.replace(cheapest, reason="retry")

    def trial_choice(self, ranked, plain_ms=None, hypothetical=None, remaining=None):
        """A new request's one trial, or None: the cheapest candidate (entry plus step) when its gain estimate
        clears ``trial_gain``, or ``min_gain`` for a record serving has ever measured ``calib_min`` steps of
        at its rung (fresh or stale: its measured tokens per row are not argued with wave after wave, and the
        prior the model falls back to on a stale pool was learned from the same text); never while the batch
        cools after an exit; and only when its entry pays over the horizon: at the hypothesis for a batch that
        is not all new (``hypothetical``: the same candidates ranked with the new rows accepting every draft on
        a record serving has not measured at the rung; a measured record keeps its measured tokens per row),
        which must clear ``min_gain`` too, else at the candidate's own estimate. A trial below break-even on
        the engine's own prior (the 12B QB2 random c1 ran one per request at 0.64-1.00x, ~150 ms each under
        asynchronous scheduling, 2026-10-30) reseeds for nothing. Independent of the retry cadence, the reseed
        is the trial's only extra cost."""
        if self.cooling():
            return None
        cheapest = _cheapest(ranked)
        mode = ("step", cheapest.shape, cheapest.width)
        known = self.calibration.pool_known(mode)
        bar = (
            max(self.config.trial_gain, self.config.min_gain)
            if (cheapest.measured or known)
            else self.config.trial_gain
        )
        if cheapest.gain < bar:
            return None
        if hypothetical is not None:
            same = (cheapest.shape, cheapest.width, cheapest.members)
            hoped = next(c for c in hypothetical if (c.shape, c.width, c.members) == same)
            if hoped.gain < self.config.min_gain or not self.entry_pays(hoped, plain_ms, remaining):
                return None
        elif plain_ms is not None and not self.entry_pays(cheapest, plain_ms, remaining):
            return None
        return dataclasses.replace(cheapest, reason="trial")

    def _tag_runs(self, candidates):
        """The candidates credited the current run's own steps where the run has any (``run`` set): what the
        adapter's ``Calibration.tokens_per_row`` hands it while a run serves, filled in here so that a caller
        passing the table alone reads the run too."""
        out = []
        for c in candidates:
            mode = ("step", c.shape, c.width)
            if self.calibration.run_samples(mode) > 0:
                c = dataclasses.replace(c, tokens_per_row=self.calibration.tokens_per_row(mode), run=True)
            out.append(c)
        return out

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

    def viable(self, candidates, plain_ms):
        """The candidates that could clear the entry gain if every draft were accepted: ``k + 1`` tokens a step at
        the record's cost against the plain step. A record below that is hopeless at its cost, whatever the
        acceptance, and is never entered, held, retried, trialed or probed: the 31B QB2's (16,1) record (2 tokens
        a step at 2.7x the plain step, 0.75x at best) was retried and trialed at the overhead bound for the whole
        16-row point and cost it 12 % (2026-10-29)."""
        plain_ms = float(plain_ms)
        return [c for c in candidates if (int(c.shape[1]) + 1) * plain_ms / float(c.cost_ms) >= self.config.min_gain]

    def would_speculate(self, candidates, rows, plain_ms, remaining=None, fresh=False):
        """Whether ``decide`` would replay a record now, without changing any counter (the dFlash bootstrap gate);
        ``fresh`` names the rows that are new requests without a trial yet (``True``: every row)."""
        for width in {int(c.width) for c in candidates}:
            self.rung_clock.setdefault(width, 0)  # the rung is known: note_plain_steps advances its clock
        candidates = self.viable(candidates, plain_ms)
        if not candidates:
            return False
        self._frame(rows, fresh)
        ranked = rank_records(self._tag_runs(candidates), rows, plain_ms)
        if self.probing:
            return ranked[0].gain >= self.config.exit_gain
        informed = self.informed(rows)
        best = self.admissible(ranked, plain_ms, remaining, informed=informed)
        if best is not None:
            return self._would_enter(best, plain_ms, informed)
        held = self.holdable(ranked)
        if self.speculating and held is not None:
            if self.trial and self.trial_age < self.trial_limit:
                return True
            return held.gain >= self.config.exit_gain or (not self.trial and self.below_exit + 1 < EXIT_STEPS)
        choice = self.retry_choice(ranked, plain_ms)
        if choice is None and fresh_indices(fresh, len(rows)):
            if not informed:
                choice = self.admissible(ranked, plain_ms, remaining, informed=True)
            if choice is None:
                hypothetical = self._hypothetical(self._tag_runs(candidates), rows, plain_ms, fresh)
                choice = self.trial_choice(ranked, plain_ms, hypothetical, remaining)
        return choice is not None and self._would_enter(choice, plain_ms, informed)

    def _would_enter(self, choice, plain_ms, informed):
        """``would_speculate``'s answer for a reseeding ``choice`` out of the plain regime: within the budget or
        not (``last_reason`` says which for the log line); an admitted entry's cost is carried to the run the
        next ``decide`` opens (the dFlash bootstrap is decided here and charged there)."""
        if not self._within_budget(choice, plain_ms, informed):
            self.last_reason = "budget"
            return False
        if choice.entry_ms > 0.0 and not (self.speculating or self.probing):
            self.pending_entry_ms = float(choice.entry_ms)
        return True

    def _within_budget(self, choice, plain_ms, informed):
        """Whether ``choice``, decided for a batch in the plain regime, is admitted by its rung's exploration
        budget: a reseeding entry is, when the budget covers the steps it will hold (``trial_steps``, one from an
        uninformed prior). Not budgeted: an entry-free record (nothing to lose), a speculating batch's switch or
        the plain probe's resume (the hold goes on), and an entry on the rung's measured credit while the rung
        has no debt (the last run there paid: the credit is evidence, and the prose c1 re-entered on it every
        request; once a run on it fails, the debt gates the next like any exploration)."""
        if choice.entry_ms <= 0.0 or self.speculating or self.probing or choice.reason == "resume":
            return True
        if self.measured_pays(choice, plain_ms) and self.rung_debt.get(int(choice.width), 0.0) <= 0.0:
            return True
        steps = self.config.trial_steps if (informed or choice.measured) else 1
        return self.budget_ok(choice, plain_ms, steps)

    def measured_pays(self, choice, plain_ms):
        """Whether ``choice`` re-enters on evidence: serving measured it at break-even or better, the last run on
        its pool did not fail (its samples would describe the text before it: prose, then random prompts), and
        its entry as served pays over the horizon."""
        return (
            choice.measured
            and choice.gain >= 1.0
            and not self.last_run_failed(choice, plain_ms)
            and self.entry_pays(choice, plain_ms)
        )

    def last_run_failed(self, choice, plain_ms):
        """Whether the last run that fed ``choice``'s pool served below the exit gain."""
        last = self.calibration.last_run_mean(("step", choice.shape, choice.width))
        return last is not None and last * float(plain_ms) / float(choice.cost_ms) < self.config.exit_gain

    def admissible(self, ranked, plain_ms, remaining=None, cooling=None, informed=True):
        """The best candidate that clears the entry gain and whose entry (if any) pays over the horizon; one that
        reseeds is passed over while the batch cools after an exit (``cooling``, None = as the policy stands),
        and, unless serving measured it, while the rows' statistics are the optimistic default (``informed``
        false: the 12B QB2 entered its 256K record on the prior alone at an expected 6 tokens a row, served 20
        steps at 0.8-0.9x and left, 2026-10-30; such a record gets a one-step trial instead)."""
        if cooling is None:
            cooling = self.cooling()
        for choice in ranked:
            if choice.gain >= self.config.min_gain and self.entry_pays(choice, plain_ms, remaining):
                if choice.entry_ms > 0.0 and (cooling or (not informed and not choice.measured)):
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
        when the gain estimate clears the trial bar and the reseed pays and,
        with other rows in the batch, the hypothesis that the new rows accept
        every draft clears the entry bar (reason "trial"; held for
        ``trial_steps`` steps, one from an uninformed prior, then continued
        under the hold rule on what its own steps delivered). For
        ``retry_steps`` plain steps after an exit no entry reseeds (reason
        "cooldown").
        """
        cfg = self.config
        candidates = list(candidates)
        if not candidates:
            raise ValueError("decide needs at least one candidate record")
        self._plain_ms = float(plain_ms)
        self._frame(rows, fresh)
        if self._at_start and (self.speculating or self.probing or self.run_open):
            # every row is at its first proposal: a new batch, whose predecessor's run ended with it. Read on,
            # that run's window held the predecessor's tail (a prose answer ends accepting little) and the new
            # request's first proposal declined on it, then re-entered a step later through an overtaken reseed
            # (2026-10-30 replay); closed here, the credit is the last run's whole mean and the entry is verified.
            self.calibration.end_run(trial=self.trial)
            self._close_run(plain_ms)
            self.last_k = 0
            self.trial = False
            self.trial_age = 0
            self.probing = False
            self.below_exit = 0
            self.below_cap = 0
        for row in rows:
            row.steps = int(getattr(row, "steps", 0)) + 1
        offered = candidates
        candidates = self.viable(candidates, plain_ms)
        choice = None
        reason = "declined"
        if not candidates:
            # every record is hopeless at its cost: the plain step, with none of the retry/trial bookkeeping a
            # declined step keeps (nothing could ever be retried)
            if self.speculating or self.probing:
                self.calibration.end_run(trial=self.trial)
                self._close_run(plain_ms)
            self._tick_clocks(offered)
            self.below_cap = 0
            self.below_exit = 0
            self.last_k = 0
            self.trial = False
            self.trial_age = 0
            self.probing = False
            self.last_reason = "hopeless"
            return None
        ranked = rank_records(self._tag_runs(candidates), rows, plain_ms)
        informed = self.informed(rows)
        best = self.admissible(ranked, plain_ms, remaining, informed=informed)
        held = self.holdable(ranked)
        if best is None and self.cooling() and self.admissible(ranked, plain_ms, remaining, cooling=False) is not None:
            reason = "cooldown"
        if self.probing:
            # The plain probe served: the hold resumes its record (every row reseeds, which the probe's budget
            # charged) under the hold rule, never through the entry or retry gates. A probe was a full exit
            # before (the 26B QB2 served 27 probes in one window and re-entered each by a retry, 2026-10-30).
            self.probing = False
            if ranked[0].gain >= cfg.exit_gain:
                choice = dataclasses.replace(ranked[0], reason="resume")
                self.below_exit = 0
            else:
                reason = "exit"
        elif best is not None:
            choice = dataclasses.replace(best, reason="enter" if best.entry_ms > 0.0 else "gain")
            self.below_exit = 0
        elif self.speculating and self.trial and self.trial_age < self.trial_limit:
            # A trial (a retry, or a new request's first proposal) holds its record for ``trial_limit`` steps
            # before the gain decides: its first step follows the reseed and accepts few drafts, and a one-step
            # trial judged on it never refreshes the acceptance pool it is judged by (the 12B rows sweeps). A
            # trial whose rows must reseed again (its replay was overtaken by a plain step the runner had in
            # flight and the first draft missed the token it committed, or a slot moved) ends: the reseed was
            # its one extra cost, and a second one per request is what the random-prompt points paid for
            # (2026-10-30); a first draft that matched seeds the next replay without one and the trial goes on.
            if held is not None:
                choice = dataclasses.replace(held, reason="trial")
        elif self.speculating and held is not None:
            # Hysteresis: a speculating batch holds its best entry-free record above the exit gain, and leaves
            # only after EXIT_STEPS consecutive decisions below it. A trial that served its steps continues
            # under the same rule on what those steps delivered (the run's own credit): the entry bar covered
            # the reseed it already paid, and a record at 1.0-1.25x measured was dropped for 240 plain steps
            # a request on the 12B QB2 prose c1 (2026-10-30). A trial below the exit gain is declined at once.
            if held.gain >= cfg.exit_gain:
                self.below_exit = 0
                choice = dataclasses.replace(held, reason="hold")
            elif self.trial:
                reason = "declined"
            else:
                self.below_exit += 1
                if self.below_exit < EXIT_STEPS:
                    choice = dataclasses.replace(held, reason="hold")
                else:
                    reason = "exit"
        else:
            choice = self.retry_choice(ranked, plain_ms)
            if choice is None and fresh_indices(fresh, len(rows)):
                if not informed:
                    # the optimistic default prior's one-step trial replays the record it recommends (every
                    # position observed by its verify), not the cheapest one
                    hopeful = self.admissible(ranked, plain_ms, remaining, informed=True)
                    choice = None if hopeful is None else dataclasses.replace(hopeful, reason="trial")
                if choice is None:
                    hypothetical = self._hypothetical(self._tag_runs(candidates), rows, plain_ms, fresh)
                    choice = self.trial_choice(ranked, plain_ms, hypothetical, remaining)
        # Every reseeding entry out of the plain regime (an enter on the credit, a trial, a retry) is within the
        # rung's exploration budget or it waits (reason "budget"): the realized loss of the rung's failed runs
        # since it last paid plus this entry's expected loss, against GEMMA4_SPEC_RETRY_OVERHEAD of the plain time
        # the rung served since its last failed run. A new request does not reset that clock, a record once
        # profitable that fails on the text re-enters no sooner than any other, and a rung never measured (the
        # long rungs' first visit) waits until its trial, priced with every draft rejected, is within the bound.
        if choice is not None and not self._within_budget(choice, plain_ms, informed):
            choice, reason = None, "budget"
        if choice is not None and choice.reason in ("gain", "hold") and self.plain_probe_due(choice, plain_ms):
            choice, reason = None, "calibrate"
        if choice is None:
            self.declined += 1
            if reason == "exit":
                self.since_exit = 0
            elif self.since_exit is not None:
                self.since_exit += 1
            if reason != "calibrate" and (self.speculating or self.probing):
                self.calibration.end_run(trial=self.trial)
                self._close_run(plain_ms)
            elif reason == "calibrate":
                self.run_cost += float(plain_ms)  # the probe is a step of the run
            self._tick_clocks(candidates)
            self.probing = reason == "calibrate"
            self.below_cap = 0
            if reason != "calibrate":
                self.below_exit = 0
            self.last_k = 0
            self.trial = False
            self.trial_age = 0
            self.last_reason = reason
            return None
        if choice.entry_ms > 0.0 and choice.reason != "resume":
            self.calibration.end_run()  # the run the batch leaves (a reseed into another record, or none)
            self._close_run(plain_ms)
            self._open_run(choice)
        elif not self.run_open:
            self._open_run(choice)
            self.run_cost += self.pending_entry_ms  # an entry would_speculate admitted (the dFlash bootstrap)
        self.pending_entry_ms = 0.0
        self.run_cost += float(choice.cost_ms) + float(choice.entry_ms)
        self.last_reason = choice.reason
        self.declined = 0
        self.since_exit = None
        was_trial = self.trial
        self.trial = choice.reason in ("retry", "trial")
        self.trial_age = self.trial_age + 1 if (self.trial and was_trial) else (1 if self.trial else 0)
        if self.trial and not was_trial:
            self.trial_limit = cfg.trial_steps if (informed or choice.measured) else 1
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
