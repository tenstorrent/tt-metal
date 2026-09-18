# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Derive SFPU accuracy budgets from the sweep the harness already runs.

Budgets must not be hand-guessed. One derived from nothing is either so loose it gates
nothing or so tight it flakes, and either way nobody can tell which by reading it. The
accuracy suite already writes a row per sampled point for every (op, format, approx mode,
fast mode, dest accumulation) variant; this reads those files and prints
``helpers/sfpu_accuracy_budget`` entries, each with the measurement it came from in a
trailing comment. That closes the loop the gate needs: measure, budget, gate, and a kernel
improvement then lands as a visible decrease in a checked-in number.

Run it from ``tests/python_tests``::

    ../.venv/bin/python -m accuracy.emit_budget --arch wh
    ../.venv/bin/python -m accuracy.emit_budget --arch wh --op tanh --op exp --formats fp32

**It measures the gate's own metric, not the sweep's diagnostic one.** The parquet carries
a ``signed_ulp_error`` column, but that is the fractional ``|err| / ulp(golden)`` form,
which returns 0.5 or 2.0 for a single representable step across a power of two and so
cannot define a budget. The files also carry ``golden_result`` and ``hardware_result``, and
those cast back into the output format's dtype losslessly, so this recomputes
:func:`helpers.ulp.ulp_distance` — the integer step count the gate actually applies. A
budget printed here is a budget the gate will honour.

Four things it refuses to paper over, because each one produces a budget that looks
plausible and gates nothing:

* **A cell with a non-finite disagreement gets no budget.** The gate requires ``NaN`` and
  ``Inf`` to agree positionally before it looks at any step count, so a cell where the
  sweep saw an ``Inf`` against a finite reference would fail at *every* budget. Printing a
  number for it would be printing a lie. Those cells are reported and skipped.
* **A budget is never below the measured maximum.** The gate is a max over lanes, so a
  budget under it fails immediately. The proposal's ``p99.9 x headroom`` is applied as a
  *floor* rather than as the answer: it stops one lane from setting the budget when the
  distribution is tight, and the measured maximum wins when the distribution has a tail.
  Both numbers are printed so the choice is auditable.
* **A budget past the point where it stops being tighter than the tolerance it replaces
  is emitted as a tolerance contract.** That bound is :func:`usable_budget_ceiling`, which
  is ``min(rtol * 2**mantissa_bits, MAX_MEANINGFUL_ULP)`` -- the ``rtol`` term, roughly
  20x tighter than ``2**mantissa_bits`` alone: 419,430 rather than 8,388,608 for fp32, and
  6 rather than 128 for bf16. This is where an approximate-mode transcendental in
  bfloat16 belongs, and it is the guard against "the sweep said 15616, so the budget is
  15616".
* **Fast mode is measured per mode and combined by taking the maximum**, because
  ``BudgetKey`` has no fast-mode dimension. Pooling the rows instead would let one mode's
  percentile pull the other's budget down, and would emit a key covering a mode a partial
  sweep never measured. Combining by anything but the max would emit a budget the gate
  cannot meet.

Keys are then collapsed where the measurements agree — over dest accumulation, then
approximation mode — so an op whose budget does not vary across those prints one line
instead of four. Where they disagree the dimension stays in the key, which is the signal
that it mattered.

**Neither format dimension is ever collapsed away**, even where every cell agrees. A key
without the input format would extend a budget to input paths this sweep never measured
(the functional suite runs ``Bfp8_b`` inputs and this does not), and a key without the
output format would match the unmeasured ``Float16`` and ``Bfp8_b`` outputs of a default
fp32+bf16 run. The enrolment model rests on absent formats falling back to tolerance on
their own, and step counts in two formats are not commensurable anyway.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
)
from helpers.sfpu_accuracy_budget import MEASURED_ARCH
from helpers.ulp import (
    MANTISSA_BITS_FOR_ULP,
    MAX_MEANINGFUL_ULP,
    NEAR_ZERO_FRACTION,
    nonfinite_mismatches,
    ulp_distance,
    ulp_dtype,
)
from helpers.utils import tolerances

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = _THIS_DIR / "_csv_output"

#: The abbreviations the harness writes into the files, mapped back to the enums the
#: registry is keyed on. Kept next to the reader rather than imported from the harness so
#: this script can read a file produced by an older revision of it.
FORMAT_BY_ABBR: Dict[str, DataFormat] = {
    "fp32": DataFormat.Float32,
    "fp16": DataFormat.Float16,
    "bf16": DataFormat.Float16_b,
}

#: The reverse of :data:`FORMAT_BY_ABBR`, so the marker lookup and the comment that
#: reports its reason cannot disagree about how a format is spelled.
_ABBR_BY_FORMAT: Dict[DataFormat, str] = {v: k for k, v in FORMAT_BY_ABBR.items()}

#: The order keys are printed in, and the order dimensions are collapsed in.
FORMAT_ORDER = (DataFormat.Float32, DataFormat.Float16_b, DataFormat.Float16)

# An input format outside FORMAT_ORDER would be measured and then never indexed, so it
# would yield no key, no note and no `render_skipped()` line while still being counted in
# `cells=N` -- a whole input pipeline missing from a table that looks complete. `_collapse`
# already guards its own use of FORMAT_ORDER with a second pass for unordered formats;
# this makes the same hazard on the `per_input` side impossible rather than handled.
assert set(FORMAT_BY_ABBR.values()) <= set(FORMAT_ORDER), (
    "FORMAT_ORDER must cover every format the sweep can name: "
    f"{sorted(f.name for f in set(FORMAT_BY_ABBR.values()) - set(FORMAT_ORDER))}"
)

#: Mantissa bits after the implicit leading 1, per measurement dtype. Same numbers as
#: ``sfpu_domains._FORMAT_MANTISSA_BITS``, derived here from the ULP ceiling that
#: ``helpers.ulp`` already publishes so the two cannot disagree.
MANTISSA_BITS: Dict[torch.dtype, int] = {
    dtype: int(math.log2(ceiling)) for dtype, ceiling in MAX_MEANINGFUL_ULP.items()
}

#: A near-zero floor may only cover a minority of a cell's lanes. Past this share the
#: dynamic-range split has not isolated a cancellation region and the floor would be
#: doing the gating; see ``CellMeasurement.near_zero_atol``.
NEAR_ZERO_MAX_SHARE = 0.5

#: The smallest budget a *measurement* may produce. A measured zero means "no error
#: observed on this stimulus", which is not the same as "no error possible" -- and the
#: distinction is not academic: eight cells here measured exactly 0 over the accuracy
#: sweep's deterministic ramp and then found a single step under the functional suite's
#: random draw. An op that is exact by *construction* is a different claim, and those
#: budgets are written by hand in the registry (Abs clears a sign bit; it cannot be one
#: step out). Nothing derived from a finite sample should assert that.
MIN_MEASURED_BUDGET = 1

#: Cells the accuracy sweep does not predict, and which are therefore left on the
#: tolerance metric rather than given a number.
#:
#: The two suites use different stimuli by design: this sweep walks a deterministic ramp
#: across the op's domain, and the functional suite draws a seeded random sample from it.
#: For most ops the ramp bounds the random draw. For an op with a singularity or a
#: near-zero tail it does not -- the random draw lands closer to the pole than any ramp
#: step does -- and the gap is not a headroom multiplier away:
#:
#:   Gelu  fp32->fp32   dest=Yes   sweep 44 steps  ->  functional 19,474,047
#:   Log1p fp16->fp32   dest=Yes   sweep 5,114     ->  functional 938,672,129
#:   Log   fp32->fp32   dest=No    sweep 1         ->  functional 65,536
#:
#: Widening a budget to cover that would blind the gate across the whole domain, and
#: inventing a multiplier that happens to cover it is the hand-guessing this script
#: exists to replace. The honest state is "not enrolled, and here is why". Enrolling them
#: needs the sweep's domain coverage extended toward each singularity first; then these
#: entries come out on their own.
#:
#: **Each marker is scoped to the variants it was measured on.** The key is
#: ``(op, in_abbr, out_abbr, dest_acc)``, with ``dest_acc=None`` meaning "both", because
#: the functional divergence is a property of a variant and not of the format pair: Gelu
#: fp32->fp32 diverges at ``dest_acc=Yes`` and Log fp32->fp32 at ``dest_acc=No``. An
#: unscoped marker forced every approximation and destination variant of the pair onto
#: tolerance -- 8192 points merged into one wildcard contract for Gelu -- and that threw
#: away the cells the sweep does predict. Every ``dest_acc``-scoped entry below was
#: re-measured against the functional suite on WH; the unscoped ones diverge on both.
_NOT_PREDICTED_BY_SWEEP: Dict[Tuple[str, str, str, Optional[DestAccumulation]], str] = {
    ("Log", "fp32", "fp32", DestAccumulation.No): (
        "functional draw reaches 65536 steps where the ramp sees 1"
    ),
    ("Log", "bf16", "fp32", DestAccumulation.No): (
        "functional draw reaches 65536 steps where the ramp sees 1"
    ),
    ("Log", "fp16", "fp32", None): (
        "functional draw reaches 57344 steps, ramp-derived 40960"
    ),
    ("Log1p", "fp16", "fp32", DestAccumulation.Yes): (
        "near-zero tail: functional reaches 938,672,129 steps"
    ),
    ("Log1p", "fp16", "bf16", None): ("near-zero tail: functional reaches 14324 steps"),
    ("Atanh", "fp16", "fp32", None): (
        "functional draw reaches 57344 steps, ramp-derived 51200"
    ),
    ("Gelu", "fp32", "fp32", DestAccumulation.Yes): (
        "near-zero tail: functional reaches 19,474,047 steps"
    ),
    # Same near-zero tail as the fp32-output key above, one format down, and the margin
    # is the point: the ramp's worst absolute error in the band is 5.66e-7, the functional
    # draw's is 7.49e-7, so it needs 1.32x where the emitter applies 1.25x. Raising
    # --headroom to cover it would loosen every budget in the table to fix one cell, which
    # is the hand-tuning this script exists to replace.
    ("Gelu", "fp32", "bf16", DestAccumulation.Yes): (
        "near-zero tail: functional abs error 7.49e-07 exceeds the ramp-derived 7.08e-07 "
        "floor, 148 steps against a 2-step budget"
    ),
}


def not_predicted_reason(cell: "CellMeasurement") -> Optional[str]:
    """Why *cell* is left on the tolerance metric, or ``None`` if it is predicted.

    One lookup for both callers: :func:`_collapse`'s ``budget_of``, which decides, and
    :meth:`EmittedKey.comment`, which reports the reason. They were written out twice,
    so widening what a marker encodes -- as the ``dest_acc`` scoping above does -- would
    otherwise have let the gating decision drift from the reported reason.
    """
    in_abbr = _ABBR_BY_FORMAT.get(cell.input_format)
    out_abbr = _ABBR_BY_FORMAT.get(cell.output_format)
    if in_abbr is None or out_abbr is None:
        return None
    for scope in (cell.dest_acc, None):
        reason = _NOT_PREDICTED_BY_SWEEP.get((cell.op.name, in_abbr, out_abbr, scope))
        if reason is not None:
            return reason
    return None


#: The sweep directory name for each architecture. The sweep writes these, so the
#: mapping lives here rather than on ``ChipArchitecture``.
ARCH_ABBR: Dict[ChipArchitecture, str] = {
    ChipArchitecture.WORMHOLE: "wh",
    ChipArchitecture.BLACKHOLE: "bh",
    ChipArchitecture.QUASAR: "qsr",
}

#: The only architecture whose sweep can be turned into an active contract, because
#: ``EmittedKey`` has no arch dimension and ``accuracy_contract()`` downgrades every other
#: architecture to the tolerance metric. *Derived* from
#: ``sfpu_accuracy_budget.MEASURED_ARCH`` rather than restated: the two were previously
#: hand-maintained literals in different naming schemes with nothing tying them together,
#: so whoever measures a second architecture had to remember both, and the test asserting
#: ``EMITTABLE_ARCH == "wh"`` was asserting it against itself.
EMITTABLE_ARCH = ARCH_ABBR[MEASURED_ARCH]

DEFAULT_HEADROOM = 1.25
DEFAULT_PERCENTILE = 99.9

#: The ops the sweep runs in both fast modes. Mirrors
#: ``accuracy/test_sfpu_accuracy.SUPPORTED_FAST_MODE_OPS``, kept here rather than imported
#: so this script can read a file produced by an older revision of the harness -- the same
#: reason ``FORMAT_BY_ABBR`` is local.
FAST_MODE_CAPABLE_OPS = (MathOperation.Rsqrt, MathOperation.Sqrt)


@dataclass(frozen=True)
class _Resolution:
    """One cell's resolved contract, *and which lane set each number came from*.

    The provenance is not decoration. Two comment lines are built from it and both were
    wrong without it:

    * ``floored`` was inferred key-level from ``near_zero_atol is not None``, but
      :meth:`CellMeasurement.resolve` combines the fast-mode components with
      ``max(budgets)`` and ``max(floors)`` *independently* -- so a key can carry a floor
      from one mode while its budget came from the other mode's unfloored all-lane
      statistics, and the comment then printed bulk statistics beside an all-lane budget.
    * On the ceiling-refusal path ``resolve()`` returns ``(None, None)``, discarding the
      floor, so ``floored`` was always false there and the "budget would be N" line
      reported an all-lane number even when the value the ceiling actually rejected had
      been computed from the bulk lanes. The verdict was never wrong -- the all-lane
      budget is the larger of the two, so also past the ceiling -- but the magnitude was.
    """

    #: The emitted budget, or ``None`` when the ceiling refused it.
    budget: Optional[int]
    #: The emitted near-zero floor, or ``None``.
    floor: Optional[float]
    #: Whether *budget* (or *rejected*) was measured over the bulk lanes rather than all
    #: of them -- i.e. whether a floor was holding the near-zero ones when it was chosen.
    floored: bool
    #: The value the ceiling rejected, when it did. ``None`` otherwise.
    rejected: Optional[int] = None


@dataclass(frozen=True)
class CellMeasurement:
    """What the sweep says about one (op, format, approx, dest) cell."""

    op: MathOperation
    input_format: DataFormat
    output_format: DataFormat
    approx_mode: ApproximationMode
    dest_acc: DestAccumulation
    #: Which fast modes this measurement covers. More than one means the statistics are
    #: the conservative combination across them; see :func:`_combine_fast_modes`.
    fast_modes: Tuple[FastMode, ...]
    points: int
    max_ulp: int
    percentile_ulp: float
    exact_fraction: float
    nonfinite_disagreements: int
    unmeasurable: int
    near_zero_points: int
    near_zero_max_ulp: int
    near_zero_max_abs_err: float
    all_max_ulp: int
    all_percentile_ulp: float
    all_exact_fraction: float
    #: The largest finite ``|golden|`` in this cell, which is what sets the *relative*
    #: half of the near-zero cut. The gate recomputes this from its own tensor, so it is
    #: the one measurement input that does not travel with the contract -- see
    #: :meth:`dynamic_range_margin`.
    dynamic_range: float = 0.0
    #: ``near_zero_fraction * dynamic_range``: the relative cut this cell's band was
    #: measured against. Stored rather than recomputed because ``--near-zero-fraction``
    #: is an argument, and the margin has to be measured against the cut that was used.
    near_zero_relative_cut: float = 0.0
    #: The band's step-count frontier: ``(|golden|, distance)`` in decreasing magnitude,
    #: keeping only the lanes that set a new maximum distance from the top down. Enough
    #: to answer "the largest band magnitude whose distance exceeds B" for any B, which
    #: is all :meth:`dynamic_range_margin` needs, and short -- a handful of entries --
    #: where the band itself can be thousands of lanes.
    near_zero_frontier: Tuple[Tuple[float, int], ...] = ()
    #: The per-fast-mode measurements this cell combines, when it combines more than one.
    #: Kept so :meth:`resolve` can combine the *resolved contracts* rather than the raw
    #: statistics -- see :func:`_combine_fast_modes`.
    components: Tuple["CellMeasurement", ...] = ()

    @property
    def gateable(self) -> bool:
        """Whether a budget can be emitted for this cell at all.

        Three ways it cannot, and :attr:`ungateable_reason` says which:

        * A **non-finite disagreement** fails the gate at *every* budget, so printing a
          number for it would be printing a lie.
        * **No measurable lane** -- every point NaN on one side or both -- supplies no
          finite evidence: the maxima come back 0, ``_budget`` floors them to
          ``MIN_MEASURED_BUDGET`` and the result reads as a measured 1-step contract. The
          ``max(..., 1)`` in :attr:`measurable_points` masks that state rather than
          rejecting it.
        * **Only one fast mode measured**, for an op that runs in both. ``BudgetKey`` has
          no fast-mode dimension, so the key would gate the unmeasured mode too.
        """
        if self.nonfinite_disagreements or self.unmeasurable >= self.points:
            return False
        if self.op in FAST_MODE_CAPABLE_OPS and len(self.fast_modes) < 2:
            # BudgetKey has no fast-mode dimension, so a key derived from one mode would
            # gate the other on a number measured for neither. A partial sweep is a
            # reason to re-run it, not to emit half a measurement.
            return False
        return True

    @property
    def ungateable_reason(self) -> Optional[str]:
        """Why no budget was emitted, for the skipped-cell report."""
        if self.nonfinite_disagreements:
            return (
                f"{self.nonfinite_disagreements} non-finite disagreement(s) in "
                f"{self.points} pts -- no budget can pass this cell; fix the op or the "
                "sweep's domain first"
            )
        if self.unmeasurable >= self.points:
            return (
                f"all {self.points} pts unmeasurable (NaN on one side or both) -- no "
                "finite lane was measured, so there is no measurement to derive a budget "
                "from; extend the sweep's domain first"
            )
        if self.op in FAST_MODE_CAPABLE_OPS and len(self.fast_modes) < 2:
            measured = ", ".join(m.name for m in self.fast_modes) or "none"
            return (
                f"{self.op.name} runs in both fast modes but the sweep measured only "
                f"{measured}; BudgetKey has no fast-mode dimension, so a key from one "
                "mode would gate the other on an unmeasured number -- re-run the sweep"
            )
        return None

    @property
    def measurable_points(self) -> int:
        return max(self.points - self.unmeasurable, 1)

    def dynamic_range_margin(self, headroom: float) -> Optional[float]:
        """How much narrower a judged tile may be before a floored contract stops holding.

        The near-zero band has two bounds and only one of them travels. ``near_zero_atol``
        is in the contract, so the gate applies the same absolute cut; the *relative*
        bound is recomputed by the gate from its own tensor's dynamic range
        (``helpers/ulp.py``), which the functional suite's random draw generally makes
        narrower than this sweep cell's deterministic ramp. Where the relative bound is
        the binding one, a narrower tile shrinks the band, and the lanes that fall out of
        it are charged against ``max_ulp`` -- lanes excluded from ``max_ulp`` precisely
        because they exceeded it.

        Returns ``cut / m``, where ``m`` is the largest band magnitude whose step count
        exceeds the emitted budget: the factor the tile's range may shrink by before that
        lane is uncovered. ``None`` when no band lane exceeds the budget (nothing to
        uncover) or when no budget was emitted. Measured against the *relative* cut,
        which is the bound that moves -- so where the absolute bound is the binding one
        the answer is if anything conservative.
        """
        budget = self._resolve(headroom).budget
        if budget is None or not self.near_zero_frontier:
            return None
        for magnitude, distance in self.near_zero_frontier:
            if distance > budget and magnitude > 0:
                return self.near_zero_relative_cut / magnitude
        return None

    def _resolve(self, headroom: float) -> _Resolution:
        """This cell's resolution: the budget, the floor, and where each came from.

        The ``(budget, near_zero_atol)`` pair is decided *together*.

        The two cannot be chosen independently, and getting that wrong is subtle enough
        to be worth spelling out: the budget measured over the bulk lanes is only valid
        if something else is holding the near-zero ones. When the floor is suppressed --
        because the near-zero split covered too many lanes to be a floor at all -- the
        budget has to cover every lane instead.

        Measured on WH, ``exp`` at bfloat16 output and ``dest_acc=Yes`` is what caught
        this: 6036 of its 6144 points sit under 1% of a dynamic range of 5.5e34, the bulk
        lanes are all exact, and emitting the bulk budget alone produced ``max_ulp=0``
        with no floor. The gate then saw all 6144 lanes, found one a step out, and failed
        a budget that the sweep had apparently justified.
        """
        if not self.gateable:
            return _Resolution(None, None, False)
        if not self.components:
            return self._resolve_measurement(headroom)

        # One key covers every fast mode, because BudgetKey has no fast-mode dimension,
        # so the resolution has to hold for each. Combining the resolved pairs rather
        # than the pooled statistics is what keeps that true: a percentile over pooled
        # rows reaches further into the worse mode's tail than that mode's own percentile
        # does, so the two disagree in both directions, and a ceiling refusal for one
        # mode has to take the whole key to tolerance rather than being averaged away.
        #
        # `_resolve_measurement`, not `resolve`: a component carries one mode, so
        # `gateable`'s completeness clause would reject every one of them. Completeness
        # is a property of this cell, and it was checked above.
        resolved = [c._resolve_measurement(headroom) for c in self.components]
        if any(r.budget is None for r in resolved):
            # A ceiling refusal for one mode takes the whole key to tolerance. The
            # provenance follows the largest rejected value, which is the one the
            # comment reports.
            refused = [r for r in resolved if r.budget is None]
            worst = max(refused, key=lambda r: r.rejected or 0)
            return _Resolution(None, None, worst.floored, worst.rejected)
        floors = [r.floor for r in resolved if r.floor is not None]
        # `max(floors)` independently of the budget, because the floor has to cover every
        # mode -- but the *provenance* follows the winning budget, which is the number the
        # comment's statistics have to match.
        winner = max(resolved, key=lambda r: r.budget or 0)
        return _Resolution(
            winner.budget, max(floors) if floors else None, winner.floored
        )

    def resolve(self, headroom: float) -> Tuple[Optional[int], Optional[float]]:
        """The ``(budget, near_zero_atol)`` pair, for the callers that want only that."""
        resolution = self._resolve(headroom)
        return resolution.budget, resolution.floor

    def _resolve_measurement(self, headroom: float) -> _Resolution:
        """One measurement's resolution, without the gateability checks."""
        floor = self._floor(headroom)
        floored = floor is not None
        if floored:
            budget = self._budget(self.max_ulp, self.percentile_ulp, headroom)
        else:
            budget = self._budget(self.all_max_ulp, self.all_percentile_ulp, headroom)

        if budget > usable_budget_ceiling(self.output_format):
            # The floor is discarded with the budget -- a floor under no budget gates
            # nothing -- but `floored` and `rejected` are kept, so the refusal can be
            # reported against the value and the lane set it was actually decided on.
            return _Resolution(None, None, floored, budget)
        return _Resolution(budget, floor, floored)

    @staticmethod
    def _budget(worst: int, percentile: float, headroom: float) -> int:
        """Never below the measured maximum, because the gate is a max over lanes.

        The percentile term is a floor that keeps a tight distribution from being gated
        at exactly its worst observed lane; it is not the answer on its own.
        """
        return max(worst, math.ceil(percentile * headroom), MIN_MEASURED_BUDGET)

    def budget(self, headroom: float) -> Optional[int]:
        return self.resolve(headroom)[0]

    def near_zero_atol(self, headroom: float) -> Optional[float]:
        return self.resolve(headroom)[1]

    def _floor(self, headroom: float) -> Optional[float]:
        """An absolute floor for the near-zero lanes, or ``None`` if they get none.

        Only when those lanes would otherwise blow the bulk budget *and* they are a
        minority. Where the reference is a small non-zero value and the hardware returns
        exactly zero -- measured on WH for ``gelu(-4.18)``, ``erfinv(0.0005)`` and
        approximate ``exp(-9.68)``, every one of them ``hw = 0`` against a golden of order
        1e-4 -- the step count from zero to that golden is five figures and says nothing
        about the kernel's accuracy anywhere else. Raising the budget to swallow it would
        blind the gate across the whole domain, which is the hole this mechanism exists to
        close.
        """
        if self.near_zero_points == 0:
            return None
        if self.near_zero_max_ulp <= self._budget(
            self.max_ulp, self.percentile_ulp, headroom
        ):
            return None
        if self.near_zero_points > NEAR_ZERO_MAX_SHARE * self.measurable_points:
            # The split has failed to isolate anything. "Below 1% of the dynamic range"
            # assumes a roughly linear-scale tensor; for an op whose output spans decades
            # it swallows almost every lane -- measured on WH, exp over its swept domain
            # reaches ~5e34, so 98% of its points sit under 1% of that and the derived
            # floor comes out at 8e29. A floor that covers the majority of lanes is not a
            # floor, it is the gate, and an absolute tolerance is the magnitude-blind gate
            # the step count exists to replace. ttnn's heuristic was written for
            # reductions and normalization, where cancellation really does produce a small
            # minority of tiny residuals; this is the guard for everything else.
            return None
        floor = self.near_zero_max_abs_err * headroom
        if floor <= 0:
            return None
        # Rounded *up* to three significant figures, not to nearest. `.3g` is
        # round-to-nearest, so it can land up to ~0.5% under the value it was derived
        # from -- and the gate's own band membership cut (`near_zero_atol /
        # near_zero_fraction`) mirrors the emitter's exactly, so a narrower floor evicts
        # a lane the emitter deliberately kept out of `bulk` and `max_ulp` therefore never
        # covered. `max(..., near_zero_max_abs_err)` does not prevent that: at
        # DEFAULT_HEADROOM that term is 0.8 * floor while the rounded value is at least
        # 0.995 * floor, so it binds only for headroom below ~1.005, and --headroom is
        # unvalidated. Widening is always safe here, so the tidy literals are kept by
        # rounding up rather than by rounding to nearest.
        rounded = max(_round_up_3sig(floor), self.near_zero_max_abs_err)
        # Bounded from above as well as from below. Nothing else bounds the floor:
        # `_floor` gates only on lane and step counts, `AccuracyContract.__post_init__`
        # checks only mutual exclusion, and `passed_test` warns only on `max_ulp`. A
        # floor above the `atol` of the `isclose` the ULP arm *returns ahead of* makes
        # the emitted contract looser than the gate it replaces, for every band lane with
        # `|golden| < (floor - atol) / rtol` -- reachable through a mid-magnitude
        # collapse, where a kernel that goes to ~0 at golden 0.1 emits 1.25 * 0.1 = 0.125
        # against an atol of 0.05. The 50% share guard does not stop that shape; it
        # suppresses the wide-range one. Refusing the floor sends the budget back to the
        # all-lane statistics, which is normally past the ceiling and therefore a
        # tolerance contract -- the honest answer for a cell that needs a floor looser
        # than the tolerance.
        gate_atol = tolerances[self.output_format].atol
        if rounded > gate_atol:
            return None
        return rounded


def _round_up_3sig(value: float) -> float:
    """*value* rounded **up** to three significant figures.

    Three figures keeps the emitted literals readable; rounding up rather than to nearest
    keeps them from landing under the measurement they were derived from. See
    :meth:`CellMeasurement._floor`.
    """
    if value <= 0:
        return value
    # Through Decimal and back, so the result is the *same* clean literal `.3g` would
    # have printed. `ceil(value / scale) * scale` in binary floating point lands on
    # 5.590000000000001e-07 instead of 5.59e-07, which would put fifteen-digit noise in
    # a checked-in table.
    quantized = Decimal(value).quantize(
        Decimal(1).scaleb(math.floor(math.log10(value)) - 2), rounding=ROUND_CEILING
    )
    return float(quantized)


def usable_budget_ceiling(output_format: DataFormat) -> float:
    """The largest budget that is still *stronger* than the gate it replaces.

    ``MAX_MEANINGFUL_ULP`` is the wrong ceiling to emit against: ``2**mantissa_bits`` is
    roughly 100% relative error, so it admits budgets that gate nothing. Because
    ``passed_test`` returns on the ULP verdict and skips both ``isclose`` and PCC, such a
    budget *is* the whole gate — measured on this sweep, approximate tanh on an fp32
    output came out at 2,949,120 steps, about 35% relative error, and tanh is bounded in
    (-1, 1), so a kernel returning 1.35 against a golden of 1.0 would have passed where
    the tolerance gate failed it.

    The real bound is the ``rtol`` half of the ``isclose`` this replaces, which is itself a
    step budget at large magnitude: ``rtol * 2**mantissa_bits`` — about 419,430 steps for
    fp32, 51 for fp16, 6 for bf16. Past it the budget is looser than what it displaced
    with no PCC behind it, and the op belongs on the tolerance metric. ``passed_test``
    warns on the same line at runtime; this refuses to *emit* past it.
    """
    dtype = ulp_dtype(output_format)
    by_rtol = tolerances[output_format].rtol * (1 << MANTISSA_BITS_FOR_ULP[dtype])
    return min(by_rtol, float(MAX_MEANINGFUL_ULP[dtype]))


def agreement_bits(max_ulp: int, output_format: DataFormat) -> float:
    """How many mantissa bits the hardware and the reference actually agree to.

    A step count only means something relative to how fine the format's steps are, and
    this is what makes two budgets in different formats comparable. ``mantissa_bits -
    log2(max_ulp)``: a 1-step disagreement in bfloat16 and a 65536-step disagreement in
    float32 are the *same* physical accuracy, 7 mantissa bits, because float32 counts
    finer steps. Printing it stops a large float32 budget from reading as a loose gate
    when it is really a narrow result in a wide container -- and stops a small one from
    reading as tight when the format simply has few bits.

    It is also the figure that decides whether a step budget is a gate at all. Where the
    agreement is at or below the *next* format's mantissa width, the result is not using
    the resolution it was asked for, and the interesting question is why, not what number
    to write in the table.
    """
    bits = MANTISSA_BITS[ulp_dtype(output_format)]
    if max_ulp <= 0:
        return float(bits)
    return bits - math.log2(max_ulp)


def _max_and_percentile(
    values: torch.Tensor, percentile: float
) -> Tuple[int, float, float]:
    """``(max, percentile, exact_fraction)`` over *values*, or zeros for an empty tensor.

    One shape for the two step-count blocks in :func:`measure_cell`, which differed only
    in whether they kept the exact fraction. The near-zero block stays separate: it
    measures an absolute error, not a step count.
    """
    if values.numel() == 0:
        return 0, 0.0, float("nan")
    as_float = values.to(torch.float64)
    return (
        int(values.max()),
        float(torch.quantile(as_float, percentile / 100.0)),
        float((values == 0).sum()) / values.numel(),
    )


def _to_enum_flag(value: object, enum_cls):
    """The harness writes these as ``"0"``/``"1"``; accept the enum or a bool too."""
    if isinstance(value, enum_cls):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes"):
        return enum_cls.Yes
    if text in ("0", "false", "no"):
        return enum_cls.No
    raise ValueError(f"cannot read {value!r} as {enum_cls.__name__}")


def load_sweep(source: Path, arch: str, ops: Optional[Sequence[str]] = None):
    """Every sweep row for *arch*, from parquet if present and csv otherwise."""
    arch_dir = source / arch
    if not arch_dir.is_dir():
        available = sorted(p.name for p in source.iterdir() if p.is_dir())
        raise SystemExit(
            f"no sweep output for arch {arch!r} under {source}. Available: "
            f"{', '.join(available) or 'none'}. Run the accuracy suite first."
        )

    wanted = {name.lower() for name in ops} if ops else None
    by_suffix: Dict[str, List[Path]] = {}
    for suffix in (".parquet", ".csv"):
        found = [
            p
            for p in sorted(arch_dir.glob(f"*{suffix}"))
            if wanted is None or p.stem.lower() in wanted
        ]
        if found:
            by_suffix[suffix] = found
    if not by_suffix:
        raise SystemExit(f"no sweep files matching {ops or 'anything'} in {arch_dir}")
    if len(by_suffix) > 1:
        # Refused rather than resolved by preference. merge_shards() rewrites only the
        # ops from the current run and leaves older per-op files in place, so after a
        # format change or a partial run one stale .parquet made every fresh .csv
        # invisible -- and the emitted budgets would carry today's stamp over another
        # run's measurement. There is no run-level manifest to tell them apart, so the
        # author has to say which set they mean.
        listing = "; ".join(
            f"{suffix}: {', '.join(p.name for p in files)}"
            for suffix, files in sorted(by_suffix.items())
        )
        raise SystemExit(
            f"{arch_dir} holds both parquet and csv sweep output ({listing}). "
            "merge_shards() leaves older per-op files in place, so these may come from "
            "different runs and there is no provenance to tell them apart. Delete the "
            "stale set. (--op does not help for the likeliest collision -- `to_csv.py` "
            "writing <name>.csv beside <name>.parquet, or a format-flipped re-run -- "
            "because both files share the stem it filters on, so the same refusal comes "
            "straight back.)"
        )
    paths = next(iter(by_suffix.values()))

    frames = [
        pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p) for p in paths
    ]
    return pd.concat(frames, ignore_index=True), [p.name for p in paths]


def measure_cell(
    rows: pd.DataFrame,
    op: MathOperation,
    input_format: DataFormat,
    output_format: DataFormat,
    approx_mode: ApproximationMode,
    dest_acc: DestAccumulation,
    fast_mode: FastMode,
    percentile: float,
    near_zero_fraction: float,
    headroom: float = DEFAULT_HEADROOM,
) -> CellMeasurement:
    """Recompute the gate's integer step count over one cell's rows."""
    dtype = ulp_dtype(output_format)
    golden = torch.tensor(rows["golden_result"].to_numpy(), dtype=torch.float64).to(
        dtype
    )
    hardware = torch.tensor(rows["hardware_result"].to_numpy(), dtype=torch.float64).to(
        dtype
    )

    disagreements = int(nonfinite_mismatches(golden, hardware).sum())
    distance = ulp_distance(golden, hardware)
    measurable = distance >= 0
    unmeasurable = int((~measurable).sum())

    # Split the lanes the way the gate splits them. The gate bounds "near zero" two ways
    # (helpers.ulp.ulp_elementwise_valid): below a fraction of the tensor's own dynamic
    # range, *and* below `near_zero_atol / near_zero_fraction` in absolute terms. Modelling
    # only the relative half is what let this script derive an atol from a lane the gate
    # then refused to rescue -- measured: hardsigmoid fp32->fp32 dest_acc=Yes emitted
    # max_ulp=10 with near_zero_atol=9.31e-09 from a lane at |golden|=7.7e-4, whose 7.45e-9
    # error is inside that atol but whose magnitude is 800x the absolute cut, so the gate
    # charged it 128 steps against a 10-step budget.
    #
    # The absolute bound depends on the atol, which is derived from the lanes the bound
    # selects, so it is solved by iteration rather than in one pass. The set only ever
    # shrinks -- dropping lanes can only lower the max error, which lowers the atol, which
    # lowers the cut -- so this terminates, and the iteration cap is belt and braces.
    absolute_error = (hardware.to(torch.float64) - golden.to(torch.float64)).abs()
    # In float32, the way the gate compares them (helpers.ulp hoists the same value):
    # both cuts are Python floats, so comparing against a 16-bit `golden.abs()` would
    # promote them onto that lattice and round each edge, and the emitter has to model the
    # split the gate performs rather than a neighbouring one.
    magnitude = golden.abs().to(torch.float32)
    finite_golden = magnitude[torch.isfinite(golden)]
    dynamic_range = float(finite_golden.max()) if finite_golden.numel() else 0.0
    if dynamic_range > 0:
        within_relative = magnitude < near_zero_fraction * dynamic_range
    else:
        within_relative = torch.ones_like(measurable)

    near_zero = within_relative
    # Until the mask stabilises, not for a fixed number of rounds. The set shrinks
    # monotonically -- dropping lanes can only lower the max error, which lowers the atol,
    # which lowers the cut, and `narrowed` is always a subset of the current mask -- so it
    # reaches a fixed point in at most one round per lane. A 64-round cap was wrong for a
    # cell with more than 64 progressively shrinking error levels, where the mask can lose
    # a single lane per round: exiting at the cap left the emitted budget derived from a
    # split the gate would not reproduce, which is exactly the contract-fails-on-its-own-
    # rows defect this modelling exists to prevent.
    for _ in range(int(golden.numel()) + 1):
        selected = measurable & near_zero
        if not bool(selected.any()):
            break
        atol = float(absolute_error[selected].max()) * headroom
        if atol <= 0:
            break
        narrowed = within_relative & (magnitude <= atol / near_zero_fraction)
        if bool(torch.equal(narrowed, near_zero)):
            break
        near_zero = narrowed
    else:  # pragma: no cover - the monotone shrink makes this unreachable
        raise AssertionError(
            "near-zero split did not converge in one round per lane; the mask is not "
            "shrinking monotonically, which breaks the termination argument above"
        )

    bulk = measurable & ~near_zero
    edge = measurable & near_zero

    max_ulp, pct, exact = _max_and_percentile(distance[bulk], percentile)
    all_max, all_pct, all_exact = _max_and_percentile(distance[measurable], percentile)

    edge_values = distance[edge]
    if edge_values.numel() == 0:
        near_zero_max_ulp, near_zero_abs = 0, 0.0
    else:
        near_zero_max_ulp = int(edge_values.max())
        near_zero_abs = float(absolute_error[edge].max())

    # The band's step-count frontier, largest magnitude first: only the lanes that set a
    # new maximum distance as we walk down. The first frontier entry whose distance
    # exceeds a budget is the largest band magnitude that does, which is what
    # `dynamic_range_margin` asks -- see there for why the emitter records it at all.
    frontier: List[Tuple[float, int]] = []
    if bool(edge.any()):
        band_magnitudes = magnitude[edge].tolist()
        band_distances = distance[edge].tolist()
        record = -1
        for mag, dist in sorted(
            zip(band_magnitudes, band_distances), key=lambda pair: -pair[0]
        ):
            if int(dist) > record:
                record = int(dist)
                frontier.append((float(mag), record))

    return CellMeasurement(
        op=op,
        input_format=input_format,
        output_format=output_format,
        approx_mode=approx_mode,
        dest_acc=dest_acc,
        fast_modes=(fast_mode,),
        points=int(len(rows)),
        max_ulp=max_ulp,
        percentile_ulp=pct,
        exact_fraction=exact,
        nonfinite_disagreements=disagreements,
        unmeasurable=unmeasurable,
        near_zero_points=int(edge.sum()),
        near_zero_max_ulp=near_zero_max_ulp,
        near_zero_max_abs_err=near_zero_abs,
        all_max_ulp=all_max,
        all_percentile_ulp=all_pct,
        all_exact_fraction=all_exact,
        dynamic_range=dynamic_range,
        near_zero_relative_cut=near_zero_fraction * dynamic_range,
        near_zero_frontier=tuple(frontier),
    )


def _combine_fast_modes(cells: Sequence[CellMeasurement]) -> CellMeasurement:
    """One measurement covering every fast mode, combined so no mode is understated.

    ``BudgetKey`` has no fast-mode dimension, so the emitted key covers both and the
    budget has to hold for both. Maxima combine by ``max`` -- which pooling the rows also
    gets right -- but a **percentile over pooled rows is not the max of the per-mode
    percentiles**: the pooled quantile reaches further into the worse mode's tail than
    that mode's own quantile does, so the two disagree in both directions, and a ceiling
    refusal for one mode would be averaged away rather than taking the whole key to
    tolerance.

    So each mode is measured on its own, the statistics here are the worst of each for
    the comment, and ``components`` carries the per-mode measurements so
    :meth:`CellMeasurement.resolve` can combine the *resolved contracts*.
    """
    if len(cells) == 1:
        return cells[0]
    first = cells[0]
    # The statistics below are for the comment; `components` is what resolve() uses.
    finite_exact = [
        c.exact_fraction for c in cells if c.exact_fraction == c.exact_fraction
    ]
    finite_all_exact = [
        c.all_exact_fraction
        for c in cells
        if c.all_exact_fraction == c.all_exact_fraction
    ]
    return CellMeasurement(
        op=first.op,
        input_format=first.input_format,
        output_format=first.output_format,
        approx_mode=first.approx_mode,
        dest_acc=first.dest_acc,
        fast_modes=tuple(
            sorted((m for c in cells for m in c.fast_modes), key=lambda m: m.name)
        ),
        points=sum(c.points for c in cells),
        max_ulp=max(c.max_ulp for c in cells),
        percentile_ulp=max(c.percentile_ulp for c in cells),
        exact_fraction=min(finite_exact, default=float("nan")),
        nonfinite_disagreements=sum(c.nonfinite_disagreements for c in cells),
        unmeasurable=sum(c.unmeasurable for c in cells),
        near_zero_points=sum(c.near_zero_points for c in cells),
        near_zero_max_ulp=max(c.near_zero_max_ulp for c in cells),
        near_zero_max_abs_err=max(c.near_zero_max_abs_err for c in cells),
        all_max_ulp=max(c.all_max_ulp for c in cells),
        all_percentile_ulp=max(c.all_percentile_ulp for c in cells),
        all_exact_fraction=min(finite_all_exact, default=float("nan")),
        components=tuple(cells),
    )


def measure_all(
    df: pd.DataFrame,
    formats: Optional[Sequence[DataFormat]],
    percentile: float,
    near_zero_fraction: float,
    headroom: float = DEFAULT_HEADROOM,
) -> Tuple[List[CellMeasurement], List[str]]:
    """One measurement per (op, format, approx, dest), fast mode measured per mode.

    *headroom* is needed here and not only at render: the gate's near-zero band is bounded
    by ``near_zero_atol / near_zero_fraction``, and the atol is ``headroom`` times the
    measured error, so the lane split depends on it.
    :meth:`CellMeasurement.resolve` must be called with the same value, which is what
    :func:`main` does.
    """
    ops_by_name = {op.name.lower(): op for op in MathOperation}
    measurements: List[CellMeasurement] = []
    notes: List[str] = []

    for op_name, per_op in df.groupby("op", sort=True):
        op = ops_by_name.get(str(op_name).lower())
        if op is None:
            notes.append(f"{op_name}: no MathOperation with that name; skipped")
            continue

        for fmt_abbr, per_fmt in per_op.groupby("output_format", sort=False):
            output_format = FORMAT_BY_ABBR.get(str(fmt_abbr))
            if output_format is None:
                notes.append(f"{op_name}: unknown output_format {fmt_abbr!r}; skipped")
                continue
            if formats is not None and output_format not in formats:
                continue

            for in_abbr, per_in in per_fmt.groupby("input_format", sort=False):
                input_format = FORMAT_BY_ABBR.get(str(in_abbr))
                if input_format is None:
                    notes.append(
                        f"{op_name}: unknown input_format {in_abbr!r}; skipped"
                    )
                    continue
                for approx_raw, per_approx in per_in.groupby("approx_mode", sort=False):
                    approx_mode = _to_enum_flag(approx_raw, ApproximationMode)
                    for dest_raw, per_dest in per_approx.groupby(
                        "dest_acc", sort=False
                    ):
                        dest_acc = _to_enum_flag(dest_raw, DestAccumulation)
                        # One measurement per fast mode, combined by the worst of each.
                        # Pooling the rows would dilute the slower mode's percentile.
                        per_mode = [
                            measure_cell(
                                rows,
                                op,
                                input_format,
                                output_format,
                                approx_mode,
                                dest_acc,
                                _to_enum_flag(fast_raw, FastMode),
                                percentile,
                                near_zero_fraction,
                                headroom,
                            )
                            for fast_raw, rows in per_dest.groupby(
                                "fast_mode", sort=True
                            )
                        ]
                        if not per_mode:
                            continue
                        measurements.append(_combine_fast_modes(per_mode))
    return measurements, notes


@dataclass(frozen=True)
class EmittedKey:
    """One registry line: which variants it covers and what it says about them."""

    input_format: Optional[DataFormat]
    output_format: Optional[DataFormat]
    approx_mode: Optional[ApproximationMode]
    dest_acc: Optional[DestAccumulation]
    budget: Optional[int]  # None => emit a tolerance contract
    near_zero_atol: Optional[float]
    cells: Tuple[CellMeasurement, ...]

    def key_source(self) -> str:
        parts = []
        if self.input_format is not None:
            parts.append(f"input_format=DataFormat.{self.input_format.name}")
        if self.output_format is not None:
            parts.append(f"output_format=DataFormat.{self.output_format.name}")
        if self.approx_mode is not None:
            parts.append(f"approx_mode=ApproximationMode.{self.approx_mode.name}")
        if self.dest_acc is not None:
            parts.append(f"dest_acc=DestAccumulation.{self.dest_acc.name}")
        return f"BudgetKey({', '.join(parts)})" if parts else "DEFAULT"

    def contract_source(self) -> str:
        if self.budget is None:
            return "AccuracyContract(metric=Metric.TOLERANCE)"
        if self.near_zero_atol is None:
            return f"AccuracyContract(max_ulp={self.budget})"
        return (
            f"AccuracyContract(max_ulp={self.budget}, "
            f"near_zero_atol={self.near_zero_atol!r})"
        )

    def comment(
        self,
        arch: str,
        stamp: str,
        percentile: float,
        headroom: float = DEFAULT_HEADROOM,
    ) -> str:
        points = sum(c.points for c in self.cells)
        # Per cell, from the cell's own resolution, not one key-level boolean derived
        # from `near_zero_atol is not None`. `resolve()` combines fast-mode components
        # with `max(budgets)` and `max(floors)` independently, so a key can carry a floor
        # from one mode while its budget came from the other mode's unfloored all-lane
        # statistics -- and on the ceiling-refusal path the floor is discarded entirely,
        # which made the key-level boolean always false there. Both printed statistics
        # from a different lane set than the number beside them.
        resolutions = [c._resolve(headroom) for c in self.cells]
        floored_cells = [r.floored for r in resolutions]
        worst = max(
            (c.max_ulp if f else c.all_max_ulp)
            for c, f in zip(self.cells, floored_cells)
        )
        pct = max(
            (c.percentile_ulp if f else c.all_percentile_ulp)
            for c, f in zip(self.cells, floored_cells)
        )
        # The widest format the key covers, deliberately: cells[0] is whichever group
        # came first out of a sort=False groupby, so a key spanning fp32 and bf16 printed
        # either ~23 or ~7 bits purely by row order. Hoisted, because the ceiling clause
        # and the mantissa-bits suffix below both need it and had grown their own copies
        # of the same expression.
        widest = max(
            (c.output_format for c in self.cells),
            key=lambda f: MAX_MEANINGFUL_ULP[ulp_dtype(f)],
        )
        # From the same lane set as max and percentile. Reporting the bulk fraction
        # beside an all-lane maximum produced contradictions like Exp2's "max N ULP,
        # 100% exact".
        exact = min(
            (
                value
                for value in (
                    (c.exact_fraction if f else c.all_exact_fraction)
                    for c, f in zip(self.cells, floored_cells)
                )
                if value == value
            ),
            default=float("nan"),
        )
        exact_text = "n/a" if exact != exact else f"{100.0 * exact:.0f}%"

        # Accumulated, not reassigned. The near-zero clause used to overwrite the
        # "measured 0, floored to 1" one, and the two are not exclusive: whenever a floor
        # is emitted `worst` is the bulk max, so a cell whose bulk lanes are all exact
        # gives worst == 0 and budget == MIN_MEASURED_BUDGET at the same time. That left
        # `max_ulp=1` under a "max 0 ULP, 100% exact" comment with no explanation --
        # which is the reading MIN_MEASURED_BUDGET's note was added to fix.
        notes: List[str] = []
        if self.budget is None:
            # Every cell the key covers, not cells[0] -- that is whichever group came
            # first out of a sort=False groupby, so a key spanning two output formats
            # printed one reason and silently dropped the other. Shipped once as a merged
            # Log1p key reporting 14324 steps while hiding 938,672,129.
            reasons: List[str] = []
            for cell in self.cells:
                reason = not_predicted_reason(cell)
                if reason is not None and reason not in reasons:
                    reasons.append(reason)
            if reasons:
                # Accumulated, not returned. A key can cover a marker cell *and* a
                # ceiling-refused sibling -- `budget_of` gives both (None, None), so the
                # two collapse together -- and returning here attributed the whole
                # refusal to the marker, dropping the ceiling clause and the measurement
                # line with it. Two shipped entries carried a `dest_acc=Yes` marker
                # reason while the `dest_acc=No` half's measurement appeared nowhere.
                notes.append("not enrolled -- " + "; ".join(reasons))
            # Report against the bound that actually rejected it: the point past which a
            # budget stops being tighter than the tolerance it replaces -- and against the
            # *value* that crossed it, which is the headroom-adjusted budget rather than
            # the measured maximum printed above. The two differ whenever the percentile
            # term wins, so attributing the refusal to `worst` produced lines that were
            # numerically false: Exp read "max 393216 ULP ... past the 419430-step point".
            #
            # From the resolution, so the value and the lane set it came from are the
            # ones the ceiling actually compared: whenever a floor existed,
            # `_resolve_measurement` rejected a *bulk* budget, and recomputing it here
            # from the key-level `floored` reported an all-lane number instead.
            refused = [r.rejected for r in resolutions if r.rejected is not None]
            if refused:
                notes.append(
                    f"budget would be {max(refused)}, past the "
                    f"{usable_budget_ceiling(widest):.0f}-step point where a budget "
                    "stops being tighter than the tolerance it replaces, so tolerance"
                )
        if self.budget == MIN_MEASURED_BUDGET and worst == 0:
            notes.append(
                f"measured 0, floored to {MIN_MEASURED_BUDGET} (a finite sample cannot "
                "assert exactness)"
            )
        if self.near_zero_atol is not None:
            edge_points = sum(c.near_zero_points for c in self.cells)
            edge_worst = max(c.near_zero_max_ulp for c in self.cells)
            notes.append(
                f"{edge_points} near-zero pts reach {edge_worst} steps and are held by "
                "the atol floor instead"
            )
            # The precondition a floored contract carries into the functional suite, in
            # the one number that says how much slack it has. The band has two bounds and
            # only the absolute one travels: `near_zero_atol` is in the contract, while
            # the relative bound is recomputed by the gate from *its own* tensor's
            # dynamic range (`helpers/ulp.py`), which the functional suite's random draw
            # makes narrower than this sweep cell's. Where the relative bound is the
            # binding one, a narrower tile shrinks the band and the lanes that fall out
            # are charged against `max_ulp` -- lanes excluded from it precisely because
            # they exceeded it. `dynamic_range_margin` is how far that can go before it
            # bites.
            margin = min(
                (
                    m
                    for m in (c.dynamic_range_margin(headroom) for c in self.cells)
                    if m is not None
                ),
                default=None,
            )
            dr = max(c.dynamic_range for c in self.cells)
            if margin is None:
                notes.append(
                    f"floor measured at dynamic range {dr:.3g}; no band lane exceeds "
                    "the budget, so a narrower tile cannot uncover one"
                )
            else:
                notes.append(
                    f"floor measured at dynamic range {dr:.3g}; holds while the judged "
                    f"tile's range stays above {dr / margin:.3g} ({margin:.1f}x narrower)"
                )
        note = "".join(f"; {text}" for text in notes)
        return (
            f"#   {arch}: max {worst} ULP, p{percentile:g} {pct:.1f}, {exact_text} "
            f"exact, ~{agreement_bits(worst, widest):.0f} mantissa bits, "
            f"{points} pts, {stamp}{note}"
        )


def _collapse(
    cells: Sequence[CellMeasurement], headroom: float, input_format: DataFormat
) -> List[EmittedKey]:
    """Group one input format's cells into the fewest keys whose budgets agree.

    Collapse over dest accumulation first, then approximation mode. Those are the only
    two stages: ``by_format_approx`` partitions on the output format and ``per_format`` is
    never merged across keys, which is the same statement the module docstring makes
    ("Neither format dimension is ever collapsed away"). A dimension only disappears when
    every cell under it wants the same budget, so a key that keeps a dimension is itself
    the statement that the dimension mattered.

    The **input** format is never collapsed away, even where every input path agrees. A
    key without it would cover input formats this sweep never measured -- the functional
    suite runs ``Bfp8_b`` inputs and the accuracy sweep does not -- and a budget silently
    extended to an unmeasured, much coarser path is exactly the kind of number this
    script exists to avoid printing.
    """

    def budget_of(cell: CellMeasurement) -> Tuple[Optional[int], Optional[float]]:
        """The (budget, floor) pair for one cell, or (None, None) for tolerance."""
        if not_predicted_reason(cell) is not None:
            return None, None
        return cell.resolve(headroom)

    # (format, approx) -> {dest: budget}
    by_format_approx: Dict[
        Tuple[DataFormat, ApproximationMode], List[CellMeasurement]
    ] = {}
    for cell in cells:
        by_format_approx.setdefault((cell.output_format, cell.approx_mode), []).append(
            cell
        )

    # Collapse dest_acc where the budgets agree.
    stage: Dict[
        DataFormat,
        List[
            Tuple[
                ApproximationMode,
                Tuple[Optional[int], Optional[float]],
                Tuple[CellMeasurement, ...],
                Optional[DestAccumulation],
            ]
        ],
    ] = {}
    for (fmt, approx), group in by_format_approx.items():
        budgets = {c.dest_acc: budget_of(c) for c in group}
        # `len(budgets) > 1` matters: a single-dest_acc group makes the set trivially
        # one-valued, and groups *are* single-dest -- main() passes only gateable cells to
        # render, and a sweep can skip a dest_acc for some input format -- so collapsing on
        # that would drop a pin the sweep never justified removing.
        if len(budgets) > 1 and len(set(budgets.values())) == 1:
            stage.setdefault(fmt, []).append(
                (approx, next(iter(budgets.values())), tuple(group), None)
            )
        else:
            for cell in group:
                stage.setdefault(fmt, []).append(
                    (approx, budget_of(cell), (cell,), cell.dest_acc)
                )

    # Collapse approx_mode where the budgets agree and dest_acc already collapsed.
    per_format: Dict[DataFormat, List[EmittedKey]] = {}
    for fmt, entries in stage.items():
        collapsible = all(dest is None for _, _, _, dest in entries)
        budgets = {approx: budget for approx, budget, _, _ in entries}
        if collapsible and len(set(budgets.values())) == 1 and len(entries) > 1:
            merged = tuple(c for _, _, group, _ in entries for c in group)
            budget, floor = next(iter(budgets.values()))
            per_format[fmt] = [
                EmittedKey(
                    input_format=input_format,
                    output_format=fmt,
                    approx_mode=None,
                    dest_acc=None,
                    budget=budget,
                    near_zero_atol=floor,
                    cells=merged,
                )
            ]
        else:
            per_format[fmt] = [
                EmittedKey(input_format, fmt, approx, dest, budget[0], budget[1], group)
                for approx, budget, group, dest in entries
            ]

    # The **output** format is never collapsed away, for the same reason as the input
    # format. A default run measures fp32 and bf16 only, so a key that dropped
    # output_format because those two happened to agree would also match the unmeasured
    # Float16 and Bfp8_b outputs -- and the whole enrolment model rests on absent formats
    # falling back to tolerance on their own. Collapsing it also left comment() picking an
    # arbitrary format to report the agreement bits against, since step counts are not
    # commensurable between fp32 and bf16.

    ordered: List[EmittedKey] = []
    for fmt in FORMAT_ORDER:
        ordered.extend(per_format.get(fmt, []))
    for fmt, keys in per_format.items():
        if fmt not in FORMAT_ORDER:
            ordered.extend(keys)
    return ordered


def render(
    measurements: Sequence[CellMeasurement],
    arch: str,
    headroom: float,
    stamp: str,
    percentile: float = DEFAULT_PERCENTILE,
) -> str:
    by_op: Dict[MathOperation, Dict[DataFormat, List[CellMeasurement]]] = {}
    for cell in measurements:
        by_op.setdefault(cell.op, {}).setdefault(cell.input_format, []).append(cell)

    lines: List[str] = []
    for op in sorted(by_op, key=lambda o: o.name):
        # budget_table(), not a dict literal: BudgetKey is frozen, so two identical keys
        # in a literal are equal and Python silently keeps the later contract -- which
        # left validate_registry() looking at an already-deduplicated table and the
        # tie-raise in resolve_contract unable to fire. The emitter cannot currently
        # produce a duplicate, since _collapse groups by key, but a hand-edit of generated
        # text can, and that is the edit this whole file exists to make safe.
        lines.append(f"    MathOperation.{op.name}: budget_table(")
        per_input = by_op[op]
        for input_format in FORMAT_ORDER:
            if input_format not in per_input:
                continue
            for key in _collapse(per_input[input_format], headroom, input_format):
                lines.append(f"    {key.comment(arch, stamp, percentile, headroom)}")
                lines.append(f"        ({key.key_source()}, {key.contract_source()}),")
        lines.append("    ),")
    return "\n".join(lines)


def render_skipped(measurements: Sequence[CellMeasurement]) -> List[str]:
    out = []
    for cell in sorted(
        (c for c in measurements if not c.gateable),
        key=lambda c: (c.op.name, c.output_format.name),
    ):
        out.append(
            f"#   {cell.op.name} {cell.input_format.name}->{cell.output_format.name} "
            f"approx={cell.approx_mode.name} dest_acc={cell.dest_acc.name}: "
            f"{cell.ungateable_reason}"
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print sfpu_accuracy_budget entries derived from the accuracy sweep.",
    )
    parser.add_argument(
        "--arch",
        default="wh",
        help="arch subdirectory to read (only 'wh' can be emitted; see below)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"sweep output root (default {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--op",
        action="append",
        dest="ops",
        help="restrict to this op (repeatable); defaults to every file present",
    )
    parser.add_argument(
        "--formats",
        default="fp32,bf16",
        help="comma-separated output formats to emit, or 'all' (default fp32,bf16 -- "
        "the order the proposal enrols them in)",
    )
    parser.add_argument(
        "--headroom",
        type=float,
        default=DEFAULT_HEADROOM,
        help=f"multiplier on the percentile floor (default {DEFAULT_HEADROOM})",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help=f"percentile used as the floor (default {DEFAULT_PERCENTILE})",
    )
    parser.add_argument(
        "--near-zero-fraction",
        type=float,
        default=NEAR_ZERO_FRACTION,
        help="lanes below this fraction of the tensor's dynamic range are judged by an "
        f"absolute floor rather than a step count (default {NEAR_ZERO_FRACTION})",
    )
    parser.add_argument(
        "--stamp",
        default=None,
        help="date written into each comment (default: today)",
    )
    args = parser.parse_args(argv)

    if args.arch.strip().lower() != EMITTABLE_ARCH:
        # EmittedKey has no arch dimension and accuracy_contract() downgrades every
        # architecture but MEASURED_ARCH to the tolerance metric before it resolves a
        # key, so text emitted from another arch's sweep cannot become an active
        # contract: it would be plausible, measured and silently ignored. Emitting
        # arch-qualified keys needs BudgetKey(arch=...) on the ULP entries, which ties
        # specificity with the per-format keys and makes validate_registry() raise, and
        # resolution support for more than one measured architecture. Until that exists
        # this refuses rather than prints.
        raise SystemExit(
            f"--arch {args.arch!r} cannot be emitted. The registry has no arch dimension "
            f"on its ULP keys and resolves every architecture but {EMITTABLE_ARCH!r} to "
            "the tolerance metric, so these budgets would be inert. Re-run with --arch "
            f"{EMITTABLE_ARCH}, or add multi-arch resolution first."
        )

    if args.near_zero_fraction <= 0 or args.near_zero_fraction >= 1:
        raise SystemExit(
            f"--near-zero-fraction must be in (0, 1); got {args.near_zero_fraction}"
        )

    if args.formats.strip().lower() == "all":
        formats: Optional[List[DataFormat]] = None
    else:
        formats = []
        for token in args.formats.split(","):
            token = token.strip()
            if token not in FORMAT_BY_ABBR:
                raise SystemExit(
                    f"unknown format {token!r}; expected from "
                    f"{', '.join(FORMAT_BY_ABBR)} or 'all'"
                )
            formats.append(FORMAT_BY_ABBR[token])

    df, files = load_sweep(args.source, args.arch, args.ops)
    measurements, notes = measure_all(
        df, formats, args.percentile, args.near_zero_fraction, args.headroom
    )
    if not measurements:
        raise SystemExit("no measurable cells; check --op and --formats")

    stamp = args.stamp or date.today().isoformat()
    gateable = [c for c in measurements if c.gateable]

    print(f"# Emitted by accuracy/emit_budget.py from {len(files)} sweep file(s)")
    print(
        f"# arch={args.arch} rows={len(df)} cells={len(measurements)} "
        f"headroom={args.headroom} percentile=p{args.percentile:g} "
        f"near_zero_fraction={args.near_zero_fraction:g}"
    )
    if args.near_zero_fraction != NEAR_ZERO_FRACTION:
        # The gate always applies NEAR_ZERO_FRACTION, so a different value here splits
        # the lanes differently from the verdict it is deriving: the emitted max_ulp
        # omits lanes the gate still charges against it. Stamped above and called out
        # here, since AccuracyContract has nowhere to carry it.
        print(
            f"# WARNING: the gate splits near-zero lanes at {NEAR_ZERO_FRACTION:g}, not "
            f"{args.near_zero_fraction:g}. Every max_ulp below omits lanes the gate will "
            "still charge against it. Re-run at the default before pasting."
        )
    print("# Paste into _SFPU_ACCURACY_BUDGET; every number below is measured.")
    for note in notes:
        print(f"# note: {note}")
    skipped = render_skipped(measurements)
    if skipped:
        print("#")
        print("# Cells deliberately given no budget; each line says why:")
        for line in skipped:
            print(line)
    print()
    print(render(gateable, args.arch, args.headroom, stamp, args.percentile))
    return 0


if __name__ == "__main__":
    sys.exit(main())
