# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The bring-up's numeric acceptance thresholds, and the code that enforces them.

Every threshold below is quoted verbatim from the bring-up requirements, and every
perf test that produces one of these numbers calls `enforce()` on it. Before this
module existed the perf suite printed its figures and asserted `total_s > 0` -- a
timing harness, not a gate -- so a regression that halved throughput would still have
run green. That is the gap this closes.

## The rule, in one paragraph

A threshold that one part meets and another misses cannot be a single unconditional
`assert`: on the part that misses it, the suite would simply be red forever, which is
not enforcement so much as a broken build. So the thresholds are declared once, in
`GATES`, and the **per-architecture verdict** is declared separately, in
`EXPECTATIONS`:

* a gate recorded as **`Meets`** is asserted directly -- if the measured value stops
  clearing the threshold, the test fails;
* a gate recorded as **`Misses`** is asserted against the *recorded measurement*, both
  bounds, exactly as `models/perf/device_perf_utils.check_device_perf` does. Slower
  than the band fails, because that is a regression. **Faster than the band also
  fails**, because it means the recorded number -- which `PERF.md` publishes -- is
  stale, and a stale published number is the thing this module exists to prevent.

Nothing here is `xfail`-ed. A missed target gets a measured number, a named lever and
a band it has to stay inside; it does not get a marker that hides it from the summary
line.

## What is recorded, and from where

Every value in `EXPECTATIONS` comes from the three-board re-verification described in
`../../PERF.md` (*Environment*), re-run in full on Blackhole `p150a`, Blackhole
`p150b` and Wormhole n300 at one commit, on one day. Two boards is not a matrix; the
figures below are what makes it one.

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
        Gate("rtf", "real-time factor, typical sentence", "Stage 1", 0.5, BELOW, ""),
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
    """The gate is not cleared; assert the recorded measurement instead.

    `recorded` is the published figure, `tol` the fractional half-width of the band
    around it. `lever` names what would close the gap -- it is printed on failure and
    on every run, because a missed target without a stated lever is just a number.
    """

    recorded: float
    tol: float
    lever: str


# Recorded on the boards named in PERF.md's *Environment*. Blackhole figures are the
# `p150a`/`p150b` pair -- the two differ by ~5 % through cooling, so the bands below
# are the union of both rather than one board's.
BLACKHOLE = {
    # End-to-end traced decode: 175.6 tok/s default on p150a, 171.3 on p150b, 201.0
    # with the in-place KV cache. The standalone traced decode step measures 163.6.
    "tok_s": Meets(),
    "tok_s_stretch": Meets(),
    # 0.378 default on p150a, 0.396 on p150b, 0.342 best (p150a, everything on).
    "rtf": Meets(),
    # Reaching 0.2 needs the LLM decode step under 1.5 ms on its own; the step is
    # 4.98 ms at its best measured and is bandwidth-bound on the AR decoder's weights.
    # Band is centred between the two boards' default configurations.
    "rtf_stretch": Misses(
        0.385, 0.35, "no op-level lever left; needs a smaller decoder or multi-chip tensor parallelism"
    ),
}

# Wormhole figures are n300. A different Wormhole part (N150) has not been measured;
# it will trip the band below rather than silently inherit n300's verdict, which is
# the intended behaviour -- see the module docstring.
WORMHOLE = {
    # End-to-end traced decode: 123.4 tok/s default, 127.5 with the in-place KV cache
    # made explicit. The standalone traced decode step measures 93.4.
    "tok_s": Meets(),
    "tok_s_stretch": Meets(),
    # 0.577 default, 0.550 with COSYVOICE_FF2_GRID=8x2. Missed, and stated as missed.
    "rtf": Misses(
        0.565,
        0.20,
        "COSYVOICE_FF2_GRID=8x2 reaches 0.550; closing the rest needs the 64-core grid's decode step under 3.2 ms",
    ),
    "rtf_stretch": Misses(0.565, 0.20, "same lever as the 0.5 gate, and further from it"),
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
