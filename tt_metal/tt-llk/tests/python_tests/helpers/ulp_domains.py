# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where an op holds its figure: the domains of a swept cell, and the typed errors the
budget table is refused with.

A row of ``sfpu_accuracy_budget.yaml`` the sweep writes carries a list of
:class:`Domain` entries. An entry holds the lanes whose input, as the kernel received
it, satisfies ``low <= x < high``; ``ulp`` is the worst ranked lane there, in steps,
outside the row's near-zero floor if it has one -- or ``None`` where too few lanes are
ranked to draw an edge on (fewer than :data:`DOMAIN_MIN_BIN_LANES` in each of its
binades): past the op's claim (``cos`` beyond 65504), undefined (``log`` below 0), or an
answer the Dest cannot hold. The sweep holds the few ranked lanes a null entry has to
the row's largest figure, so the emitter writes ``None`` only while they fit it
(:func:`entries_of`). An entry whose figure is within the output format's
:func:`usable_budget_ceiling` is gated at it by the sweep; one past the ceiling is
measured and held to its figure by the headroom report. The whole-cell contract is
derived (:func:`contract_of`).

The list is a contiguous partition of ``(-inf, inf)`` whose interior edges are ``0`` or
powers of two (:func:`check_domains`). The emitter draws it from the sweep's per-lane
distances (:func:`partition`); nothing else writes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple, Union

from .format_config import DataFormat


class Problem(Enum):
    """Why the budget table, a row of it or a ``domains`` list was refused. Tests assert
    on these, never on a message."""

    UNKNOWN_OP = auto()
    UNKNOWN_FIELD = auto()
    NOT_A_MEMBER = auto()
    BAD_NUMBER = auto()
    WRONG_SHAPE = auto()
    UNREADABLE = auto()
    MISSING_FIELD = auto()
    NO_ROWS = auto()
    DUPLICATE_YAML_KEY = auto()
    REPEATED_KEY = auto()
    AMBIGUOUS = auto()
    NOT_A_COMPUTE_FORMAT = auto()
    MISSING_FIGURE = auto()
    MIXED_METRIC = auto()
    NONFINITE_ON_BUDGET = auto()
    BESIDE_DOMAINS = auto()
    UNPINNED_DOMAINS = auto()
    EMPTY_DOMAINS = auto()
    EMPTY_RANGE = auto()
    NOT_COVERING = auto()
    GAP = auto()
    EDGE_NOT_POWER_OF_TWO = auto()
    NO_FIGURE = auto()


class TableError(ValueError):
    """The table, or something built for it, is refused: :attr:`problem` says why, and
    the message names where, for the person fixing it."""

    def __init__(self, problem: Problem, where: object = ""):
        self.problem = problem
        super().__init__(f"{problem.name}: {where}" if where != "" else problem.name)


def usable_budget_ceiling(output_format: DataFormat) -> float:
    """The largest budget that is still *stronger* than the gate it replaces.

    ``MAX_MEANINGFUL_ULP`` is the wrong bound: ``2**mantissa_bits`` is roughly 100%
    relative error, so it admits budgets that gate nothing -- and since ``passed_test``
    returns on the ULP verdict and skips both ``isclose`` and PCC, such a budget *is* the
    whole gate. The real bound is the ``rtol`` half of the ``isclose`` this replaces,
    itself a step budget at large magnitude, rounded *down* to a whole step: 419,430
    steps for fp32, 51 for fp16, 6 for bf16. Down, not up: bf16's 6.4 lets ``isclose``
    accept ``128 -> 134`` and refuse ``128 -> 135``, so a 7-step budget would pass what
    the tolerance it displaced rejected, with no PCC behind it.
    """
    from .ulp import MANTISSA_BITS_FOR_ULP, MAX_MEANINGFUL_ULP, ulp_dtype
    from .utils import tolerances

    dtype = ulp_dtype(output_format)
    by_rtol = tolerances[output_format].rtol * (1 << MANTISSA_BITS_FOR_ULP[dtype])
    return float(math.floor(min(by_rtol, float(MAX_MEANINGFUL_ULP[dtype]))))


#: How finely the gateable range is split. ``None`` is the binary rule: a binade is
#: either within the ceiling or past it, and adjacent binades on the same side merge.
#: An integer *n* would split the gateable range into step classes a factor of ``2**n``
#: apart as well (0 and 1 merged). Drawn from one Wormhole sweep's recorded profiles,
#: per cell and before the emitter's collapse (the 3,041 cells with a profile),
#: factor-2 classes gave 8,452 entries and the binary rule 5,677; the extra ones mostly
#: part a 0-1-step binade from a 2-7-step neighbour or, on a Float32 output, an exact
#: binade from one at 2**13-2**17 steps. The binary rule is the question asked: where
#: is the op safe.
DOMAIN_CLASS_BITS: Optional[int] = None

#: A binade with fewer lanes *present* than this (the single zero lane; a format's last
#: partial binade) joins its neighbour; one with that many lanes but fewer *ranked* is
#: unranked -- the stray ranked lane at x = 1 in erfinv's [1, 2) binade is noise, and
#: the list ends its ranked domain at 1.
DOMAIN_MIN_BIN_LANES = 16

#: A gateable domain with fewer ranked lanes than this (1.6% of a 16-bit format) is
#: folded into the neighbour at least as loose as itself, so a one-binade island does
#: not become an entry; a domain past the ceiling is never folded into a gateable one,
#: so a narrow bad region is never hidden.
DOMAIN_MIN_LANES = 1024

#: Makes every float32 binade index positive, so the sign of the index is the sign of
#: x: the binade of the smallest subnormal, -2**-149, is ``[-2**-149, -2**-150)``, k =
#: -150 (see :func:`binade`), so ``k + 151`` is at least 1.
BINADE_OFFSET = 151

_INF = float("inf")


@dataclass(frozen=True)
class Domain:
    """One entry: the lanes with ``low <= x < high``, and the worst of them in steps,
    or ``None`` where none is ranked."""

    low: float
    high: float
    ulp: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise TableError(Problem.EMPTY_RANGE, self)
        if self.ulp is not None and self.ulp < 0:
            raise TableError(Problem.BAD_NUMBER, self)

    def gated(self, ceiling: float) -> bool:
        """Whether the sweep holds this entry's lanes to its figure."""
        return self.ulp is not None and self.ulp <= ceiling


def binade(x: float) -> int:
    """The signed binade index of a finite input: 0 for zero, otherwise
    ``sign(x) * (k + BINADE_OFFSET)`` for the *k* whose binade (:func:`binade_edges`)
    holds x under the lists' own ``low <= x < high``: ``2**k <= x < 2**(k+1)``, or
    ``-2**(k+1) <= x < -2**k``. A negative power of two is the *low* edge of the binade
    above it, as -1 is of ``[-1, -0.5)``, so its k is one less than its magnitude's;
    otherwise a binade's lanes would straddle an entry's edge. Orders binades as the
    values do."""
    if x == 0:
        return 0
    mantissa, exponent = math.frexp(abs(x))
    k = exponent - 1
    if x < 0 and mantissa == 0.5:
        k -= 1
    index = k + BINADE_OFFSET
    return index if x > 0 else -index


def binade_edges(index: int) -> Tuple[float, float]:
    """``[low, high)`` of the binade *index* names: ``[2**k, 2**(k+1))`` on the positive
    side, ``[-2**(k+1), -2**k)`` on the negative, ``[0, 0)`` for zero's own."""
    if index == 0:
        return (0.0, 0.0)
    k = abs(index) - BINADE_OFFSET
    return (2.0**k, 2.0 ** (k + 1)) if index > 0 else (-(2.0 ** (k + 1)), -(2.0**k))


#: One binade of a cell: lanes present, lanes ranked, worst ranked lane (-1 if none).
Bin = Tuple[int, int, int]


class BinClass(Enum):
    """What :func:`partition` makes of one binade: too sparse to say (it takes a
    neighbour's class), too few lanes ranked, within the ceiling, or past it. With
    :data:`DOMAIN_CLASS_BITS` set, a gateable binade is an integer step class."""

    PENDING = auto()
    UNRANKED = auto()
    GATEABLE = auto()
    PAST = auto()


#: A binade's class: a :class:`BinClass`, or a step class under ``DOMAIN_CLASS_BITS``.
Class = Union[BinClass, int]


def _step_class(worst: int, ceiling: float) -> Class:
    if worst > ceiling:
        return BinClass.PAST
    if DOMAIN_CLASS_BITS is None:
        return BinClass.GATEABLE
    return 1 + int(math.floor(math.log2(max(worst, 1)) / DOMAIN_CLASS_BITS))


def _looseness(cls: Class) -> float:
    if cls is BinClass.PAST:
        return _INF
    return 0.0 if cls is BinClass.GATEABLE else float(cls)


def partition(
    bins: Dict[int, Bin], ceiling: float
) -> Tuple[Tuple[float, float, bool], ...]:
    """The domains the rule draws on a cell's binade profile, as ``(low, high, ranked)``
    triples covering ``(-inf, inf)``; the caller measures each ranked domain's figure over
    its own lanes.

    Per binade: fewer than :data:`DOMAIN_MIN_BIN_LANES` lanes present means it takes the
    class of the binade before it (or after, at the start); fewer ranked means it is
    unranked; otherwise its worst lane is within the ceiling or past it (or, with
    :data:`DOMAIN_CLASS_BITS`, in a step class). Adjacent binades of one class merge; a
    gateable run under :data:`DOMAIN_MIN_LANES` ranked lanes folds into the neighbour at
    least as loose as itself, the loosest first, and never into an unranked one. The
    outer edges are infinite; an edge with no swept lane between it and zero is
    written as 0.
    """
    if not bins:
        return ((-_INF, _INF, False),)
    labelled: List[list] = []
    for index in sorted(bins):
        present, ranked, worst = bins[index]
        if present < DOMAIN_MIN_BIN_LANES:
            cls = None
        elif ranked < DOMAIN_MIN_BIN_LANES:
            cls = BinClass.UNRANKED
        else:
            cls = _step_class(worst, ceiling)
        labelled.append([index, ranked, cls])
    last = None
    for row in labelled:
        if row[2] is None:
            row[2] = last if last is not None else BinClass.PENDING
        else:
            last = row[2]
    nxt = BinClass.UNRANKED
    for row in reversed(labelled):
        if row[2] is BinClass.PENDING:
            row[2] = nxt
        else:
            nxt = row[2]
    runs: List[dict] = []
    for index, ranked, cls in labelled:
        if runs and runs[-1]["cls"] == cls:
            runs[-1]["last"] = index
            runs[-1]["ranked"] += ranked
        else:
            runs.append({"first": index, "last": index, "cls": cls, "ranked": ranked})
    changed = True
    while changed:
        changed = False
        narrow = sorted(
            (
                i
                for i, r in enumerate(runs)
                if r["cls"] not in (BinClass.UNRANKED, BinClass.PAST)
                and r["ranked"] < DOMAIN_MIN_LANES
            ),
            key=lambda i: runs[i]["ranked"],
        )
        for i in narrow:
            own = _looseness(runs[i]["cls"])
            candidates = [
                j
                for j in (i - 1, i + 1)
                if 0 <= j < len(runs)
                and runs[j]["cls"] is not BinClass.UNRANKED
                and _looseness(runs[j]["cls"]) >= own
            ]
            if not candidates:
                continue
            j = max(candidates, key=lambda j: _looseness(runs[j]["cls"]))
            lo, hi = min(i, j), max(i, j)
            runs[lo : hi + 1] = [
                {
                    "first": runs[lo]["first"],
                    "last": runs[hi]["last"],
                    "cls": runs[j]["cls"],
                    "ranked": runs[lo]["ranked"] + runs[hi]["ranked"],
                }
            ]
            changed = True
            break
    merged: List[dict] = []
    for r in runs:
        if merged and merged[-1]["cls"] == r["cls"]:
            merged[-1]["last"] = r["last"]
            merged[-1]["ranked"] += r["ranked"]
        else:
            merged.append(dict(r))
    # The smallest magnitude any swept lane has: an edge at or below it has nothing
    # between it and zero, and reads better as 0.
    smallest = min(
        (
            abs(edge)
            for index, (present, _, _) in bins.items()
            if index != 0 and present > 0
            for edge in binade_edges(index)
        ),
        default=0.0,
    )
    out: List[Tuple[float, float, bool]] = []
    for k, r in enumerate(merged):
        low = -_INF if k == 0 else binade_edges(r["first"])[0]
        high = _INF if k == len(merged) - 1 else binade_edges(r["last"])[1]
        if 0 < abs(low) <= smallest:
            low = 0.0
        if 0 < abs(high) <= smallest:
            high = 0.0
        ranked = r["cls"] is not BinClass.UNRANKED
        if out and not low < high:
            # A run of nothing but the zero lane, snapped to [0, 0): dropped rather than
            # written empty. The next entry's `low` is pulled down to 0 below, so under
            # `low <= x < high` the zero lane lands in the domain after it.
            continue
        if out and out[-1][1] != low:
            low = out[-1][1]
        out.append((low, high, ranked))
    return tuple(out)


def entries_of(
    spans: Tuple[Tuple[float, float, bool], ...],
    worst: Callable[[float, float], Optional[int]],
) -> Tuple[Domain, ...]:
    """The entries the table writes for :func:`partition`'s *spans*, given *worst(low,
    high)*: the worst ranked lane in ``[low, high)`` outside the floor (0 where the floor
    covers every one), ``None`` where none is ranked.

    A ranked span carries its worst lane. An unranked one is ``null``, but it can still
    hold a few ranked lanes, and the sweep holds those to the row's largest figure; so
    if any are worse than every figure in the list, the span whose strays are worst
    carries its own figure instead, and the next run on the same output passes the list
    it was written with. ``()`` -- no list, a plain row -- when no ranked span measured
    anything: ``check_domains`` refuses a list without a figure."""
    figures = [worst(low, high) if ranked else None for low, high, ranked in spans]
    if all(f is None for f in figures):
        return ()
    largest = max(f for f in figures if f is not None)
    strays = [
        (stray, i)
        for i, (low, high, ranked) in enumerate(spans)
        if not ranked
        for stray in (worst(low, high),)
        if stray is not None
    ]
    if strays:
        stray, i = max(strays)
        if stray > largest:
            figures[i] = stray
    return tuple(Domain(low, high, f) for (low, high, _), f in zip(spans, figures))


def _is_power_of_two(value: float) -> bool:
    return value != 0 and math.isfinite(value) and math.frexp(abs(value))[0] == 0.5


def check_domains(domains: Tuple[Domain, ...]) -> None:
    """Refuse a list that is not a contiguous partition of ``(-inf, inf)`` with interior
    edges at 0 or a power of two, or that holds no figure at all."""
    if not domains:
        raise TableError(Problem.EMPTY_DOMAINS)
    if domains[0].low != -_INF or domains[-1].high != _INF:
        raise TableError(Problem.NOT_COVERING, domains)
    for before, after in zip(domains, domains[1:]):
        if before.high != after.low:
            raise TableError(Problem.GAP, (before, after))
        if before.high != 0 and not _is_power_of_two(before.high):
            raise TableError(Problem.EDGE_NOT_POWER_OF_TWO, before)
    if all(d.ulp is None for d in domains):
        raise TableError(Problem.NO_FIGURE, domains)


def largest_figure(domains: Tuple[Domain, ...]) -> int:
    """The largest figure of a checked list (:func:`check_domains` guarantees one): what
    the ranked lanes of its null entries are held to, by the sweep, ``diff`` and
    ``headroom`` alike, and the whole cell's figure."""
    return max(d.ulp for d in domains if d.ulp is not None)


def contract_of(domains: Tuple[Domain, ...], ceiling: float) -> Tuple[bool, int]:
    """``(gated, figure)``: the whole-cell contract a checked list implies -- gated at
    its largest figure when every figure is within *ceiling*, tolerance otherwise."""
    figure = largest_figure(domains)
    return figure <= ceiling, figure
