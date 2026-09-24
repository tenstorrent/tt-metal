# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The bring-up's numeric acceptance thresholds, and the code that enforces them.

Every threshold below is quoted verbatim from the bring-up requirements, and every
perf test that produces one of these numbers calls `enforce()` on it, so a regression
fails the suite.

## The rule, in one paragraph

A threshold that one part meets and another does not cannot be a single
unconditional `assert`: on the part that falls short, the suite would be red forever.
So the thresholds are declared once, in `GATES`, and the per-architecture verdict is
declared separately, in `EXPECTATIONS`:

* a threshold recorded as `Meets` is asserted directly -- if the measured value stops
  clearing it, the test fails;
* a threshold recorded as `Misses` is asserted against the recorded measurement, in
  both directions, as `models/perf/device_perf_utils.check_device_perf` does. Slower
  than the band fails, because that is a regression. Faster than the band also fails,
  because it means the recorded number -- which `PERF.md` publishes -- is stale.

Nothing here is `xfail`-ed. An unmet target gets a measured number, a named lever and
a band it has to stay inside, not a marker that hides it from the summary line.

## What is recorded, and from where

Every value in `EXPECTATIONS` comes from the certification run described in
`../../PERF.md` Part I -- one commit, one day, Blackhole `p150a`, Blackhole `p150b`
and Wormhole n300, five configurations each -- except `rtf_synthesize`, which postdates
that run: it was recorded on `p150a` alone, and n300 has no figure for it yet.

A `recorded` value is the centre of a band, not the last run's figure. PERF.md
publishes what a given run measured; this table holds the reference those measurements
have to stay near. Re-centring it after every run would defeat the point — the band
exists because the same board measures a few per cent apart from day to day. What must
hold is that PERF.md's published figures lie inside these bands; when one stops doing
so, the run fails and both are updated together.

Bands are wide on purpose. The flow decoder varies by about 5 % run to run, the two
Blackhole boards differ by another ~5 % through cooling alone, and a host under load
moves the LLM step. A band tight enough to catch a 3 % drift would flake on all three;
these are sized to catch the failures that matter -- a lost trace capture, a dropped
fused-attention path, a cache that started reallocating -- which are 20 % events and
larger.
"""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# the thresholds themselves
# --------------------------------------------------------------------------
AT_LEAST, BELOW = "at_least", "below"


@dataclass(frozen=True)
class Gate:
    """One numeric acceptance threshold."""

    key: str
    label: str
    stage: str
    target: float
    direction: str
    unit: str

    def passes(self, measured: float) -> bool:
        return measured >= self.target if self.direction == AT_LEAST else measured < self.target

    def describe(self) -> str:
        op = ">=" if self.direction == AT_LEAST else "<"
        return f"{self.label} {op} {self.target}{self.unit}"


GATES: dict[str, Gate] = {
    g.key: g
    for g in (
        # Stage 1 -- bring-up baselines.
        Gate("tok_s", "semantic token generation", "Stage 1", 30.0, AT_LEAST, " tok/s"),
        # Two figures for the one requirement. `rtf` is the steady state
        # (`test_device_end_to_end_rtf`): the decode step's time scaled by the token count,
        # and the flow's trace replayed. `rtf_synthesize` is `synthesize` per utterance
        # (`test_device_synthesize_rtf`), with the prefill and the captures a caller pays
        # on every call; the requirement's verdict is that one.
        Gate("rtf", "real-time factor, steady state", "Stage 1", 0.5, BELOW, ""),
        Gate("rtf_synthesize", "real-time factor, typical sentence", "Stage 1", 0.5, BELOW, ""),
        # Stage 3 -- stretch targets.
        Gate("tok_s_stretch", "semantic token generation", "Stage 3 stretch", 60.0, AT_LEAST, " tok/s"),
        Gate("rtf_stretch", "real-time factor", "Stage 3 stretch", 0.2, BELOW, ""),
    )
}


# --------------------------------------------------------------------------
# per-architecture verdicts
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Meets:
    """The gate is cleared on this architecture; assert the gate itself."""


@dataclass(frozen=True)
class Misses:
    """The threshold is not met; assert the recorded measurement instead.

    `recorded` is the published figure, `tol` the fractional half-width of the band
    around it. `lever` names what would close the gap -- it is printed on failure and
    on every run, because an unmet target without a stated lever is just a number.
    """

    recorded: float
    tol: float
    lever: str


@dataclass(frozen=True)
class MissesUnrecorded:
    """The threshold is not met, and no measurement on this architecture is recorded yet.

    Asserted in the one direction that is known without the figure, that the target is
    still not met, and the measured value printed, so the first run on the part supplies
    the figure to record as a `Misses` band. `why` says how the verdict is known.
    """

    why: str
    lever: str


# Recorded on the boards named in PERF.md §1, *The boards*. Blackhole figures are the
# `p150a`/`p150b` pair -- the two differ by ~5 % through cooling, so the bands below
# are the union of both rather than one board's.
BLACKHOLE = {
    # Every configuration clears both throughput thresholds by a wide margin (PERF.md
    # §3.3).
    "tok_s": Meets(),
    "tok_s_stretch": Meets(),
    # PERF.md §3.2.
    "rtf": Meets(),
    # `synthesize` per utterance, zero-shot zh, second call, p150a on 2026-09-24:
    # 0.534-0.538 default, 0.512 with `COSYVOICE_FF2_GRID=8x2`, 0.732-0.736 with
    # `COSYVOICE_KV_INPLACE=1`, whose cache captures 65 decode traces per utterance where
    # the moving one captures one -- a cost the steady state never pays. Centred between
    # them, and wide enough for p150b's ~5 %.
    "rtf_synthesize": Misses(
        0.62,
        0.30,
        "the per-utterance fixed cost: the LLM's prefill and decode-trace capture (about 1.1 s "
        "on p150a, more with the in-place cache) and the flow's capture; keeping traces across "
        "utterances would remove them",
    ),
    # Reaching 0.2 needs the LLM decode step under 1.5 ms on its own, several times
    # below its best measured step (PERF.md §3.4). The band is centred between the two
    # boards' default configurations.
    "rtf_stretch": Misses(
        0.385, 0.35, "no op-level lever left; needs a smaller decoder or multi-chip tensor parallelism"
    ),
}

# These figures are from n300 specifically. A different Wormhole part will trip the
# band below rather than silently inherit n300's verdict, which is the intended
# behaviour -- see the module docstring.
WORMHOLE = {
    # PERF.md §3.3. The explicit in-place row measures the same configuration as the
    # default -- `kv_inplace_default` turns the in-place cache on for Wormhole -- and
    # comes within noise of it, as it should.
    "tok_s": Meets(),
    "tok_s_stretch": Meets(),
    # PERF.md §3.2. The same board moves a few per cent between runs -- the flow
    # decoder's run-to-run variation -- so the band is +/-20 %, centred between two
    # same-day runs. `COSYVOICE_FF2_GRID=8x2` is within noise of the default on this
    # part, which is one reason the flag stays opt-in: its best shape is not portable.
    "rtf": Misses(
        0.55,
        0.20,
        "no flag closes this on n300; it needs the 64-core grid's decode step under "
        "3.2 ms against a measured 10.9, so it is the compute grid rather than tuning",
    ),
    "rtf_stretch": Misses(0.55, 0.20, "same lever as the 0.5 gate, and further from it"),
    "rtf_synthesize": MissesUnrecorded(
        "the steady-state figure above already misses 0.5 on n300, and this one adds the "
        "per-utterance prefill and captures to the same stages",
        "the steady-state figure's lever, plus the per-utterance fixed cost",
    ),
}

EXPECTATIONS = {"blackhole": BLACKHOLE, "wormhole": WORMHOLE}


def arch_key(device) -> str:
    """`'blackhole'` or `'wormhole'` from a live device.

    Keyed on the architecture rather than the board because that is what the code
    branches on everywhere else in this port -- `kv_inplace_default` reads the same
    string -- and because a board name is not available from ttnn at all.
    """
    arch = str(device.arch()).upper()
    if "BLACKHOLE" in arch:
        return "blackhole"
    if "WORMHOLE" in arch:
        return "wormhole"
    raise AssertionError(f"no recorded expectations for architecture {arch!r}")


# --------------------------------------------------------------------------
# enforcement
# --------------------------------------------------------------------------
def enforce(key: str, measured: float, device, *, extra: str = "") -> str:
    """Assert `measured` against gate `key` on `device`'s architecture.

    Returns the one-line verdict it printed, so a caller can collect the lines into a
    summary table. Raises `AssertionError` on any of the three failure modes:

    1. a `Meets` gate no longer cleared -- a real regression against the requirement;
    2. a `Misses` gate that got worse than its recorded band -- a regression against
       the published figure;
    3. a `Misses` gate that got *better* than its recorded band -- the published
       figure is stale, and `PERF.md` plus this table need updating. Promote it to
       `Meets()` once it clears the threshold.
    """
    gate = GATES[key]
    verdict = EXPECTATIONS[arch_key(device)][key]
    arch = arch_key(device)
    suffix = f"  [{extra}]" if extra else ""

    if isinstance(verdict, MissesUnrecorded):
        line = (
            f"{gate.describe():<52} measured {measured:8.3f}   MISS, no band recorded for {arch} yet{suffix}\n"
            f"    known because: {verdict.why}\n    lever: {verdict.lever}"
        )
        assert not gate.passes(measured), (
            f"{gate.stage} gate {gate.describe()} is MET on {arch} (measured {measured:.3f}), but "
            f"tests/perf/gates.py records it as not met. Record the figure: Meets() here, and PERF.md."
        )
        return line

    if isinstance(verdict, Meets):
        line = f"{gate.describe():<52} measured {measured:8.3f}   {'PASS' if gate.passes(measured) else 'FAIL'}{suffix}"
        assert gate.passes(measured), (
            f"{gate.stage} gate not met on {arch}: {gate.describe()}, measured {measured:.3f}. "
            f"This gate is recorded as met in tests/perf/gates.py -- either a regression, "
            f"or the run is not comparable (check trace capture actually happened)."
        )
        return line

    lo = verdict.recorded * (1 - verdict.tol)
    hi = verdict.recorded * (1 + verdict.tol)
    line = (
        f"{gate.describe():<52} measured {measured:8.3f}   MISS, in band "
        f"[{lo:.3f}, {hi:.3f}]{suffix}\n    lever: {verdict.lever}"
    )
    assert not gate.passes(measured), (
        f"{gate.stage} gate {gate.describe()} is now MET on {arch} (measured {measured:.3f}), "
        f"but tests/perf/gates.py records it as missed at {verdict.recorded}. "
        f"Promote it to Meets() and update PERF.md -- a published figure that is worse "
        f"than reality is still a wrong published figure."
    )
    assert lo <= measured <= hi, (
        f"{gate.stage} gate {gate.describe()} on {arch}: measured {measured:.3f}, outside the "
        f"recorded band [{lo:.3f}, {hi:.3f}] around {verdict.recorded}. "
        f"{'Slower than recorded -- a regression.' if measured > hi else 'Faster than recorded -- update PERF.md and this table.'}"
    )
    return line


def report(lines: list[str], title: str) -> None:
    """Print a collected set of `enforce` verdicts as one block."""
    print(f"\n  {title}")
    for line in lines:
        print(f"    {line}")
