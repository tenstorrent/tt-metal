# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where a swept op holds its figure, and the typed errors the budget table is refused
with.

A :class:`Domain` entry holds the lanes with ``low <= x < high`` (x as the kernel
received it) and the worst ranked lane there in steps, or no figure where too few lanes
are ranked. A row's list is a contiguous partition of ``(-inf, inf)`` with interior
edges at 0 or a power of two (:func:`check_domains`). Only the emitter writes it, from
the sweep's per-binade profile (:func:`partition`, :func:`entries_of`).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

from .format_config import DataFormat
from .ulp import mantissa_bits, max_meaningful_ulp, ulp_dtype
from .utils import tolerances


class Problem(Enum):
    """Why the table, a row, a list or a measurement record was refused."""

    BAD_YAML = auto()
    UNKNOWN_OP = auto()
    UNKNOWN_FIELD = auto()
    MISSING_FIELD = auto()
    NOT_A_MEMBER = auto()
    BAD_NUMBER = auto()
    WRONG_SHAPE = auto()
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
    """Refused for a typed reason; *where* names what, for the person fixing it."""

    def __init__(self, problem: Problem, *where: object):
        self.problem = problem
        self.where = where
        super().__init__(": ".join([problem.name, *map(str, where)]))


def usable_budget_ceiling(output_format: DataFormat) -> float:
    """The largest step budget still stronger than the ``rtol`` it replaces, rounded down
    to a whole step: 419,430 for fp32, 51 for fp16, 6 for bf16. ``max_meaningful_ulp``
    (``2**mantissa_bits``, about 100% relative error) would admit budgets that gate
    nothing."""
    dtype = ulp_dtype(output_format)
    by_rtol = tolerances[output_format].rtol * (1 << mantissa_bits(dtype))
    return float(math.floor(min(by_rtol, float(max_meaningful_ulp(dtype)))))


# A binade with fewer lanes present joins its neighbour; with that many present but
# fewer ranked it is unranked (a stray ranked lane is noise, not a domain).
DOMAIN_MIN_BIN_LANES = 16

# A gateable run with fewer ranked lanes folds into a neighbour at least as loose; a run
# past the ceiling never folds into a gateable one, so a bad region is never hidden.
DOMAIN_MIN_LANES = 1024

# Added to a binade's exponent so every float32 binade index is nonzero and signed like x.
BINADE_OFFSET = 151

_INF = float("inf")


@dataclass(frozen=True)
class Domain:
    """The lanes with ``low <= x < high`` and the worst of them in steps, or ``None``
    where none is ranked."""

    low: float
    high: float
    ulp: int | None = None

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise TableError(Problem.EMPTY_RANGE, self)
        if self.ulp is not None and self.ulp < 0:
            raise TableError(Problem.BAD_NUMBER, self)


def binade(x: float) -> int:
    """The signed binade index of a finite input: 0 for zero, else ``sign(x) * (k +
    BINADE_OFFSET)`` with ``2**k <= |x| < 2**(k+1)``. A negative power of two is the
    *low* edge of the binade above it (-1 lies in ``[-1, -0.5)``), so a binade's lanes
    never straddle an entry's ``low <= x < high`` edge. Orders as the values do."""
    if x == 0:
        return 0
    mantissa, exponent = math.frexp(abs(x))
    k = exponent - 1
    if x < 0 and mantissa == 0.5:
        k -= 1
    index = k + BINADE_OFFSET
    return index if x > 0 else -index


def binade_edges(index: int) -> tuple[float, float]:
    """``[low, high)`` of the binade *index* names; ``[0, 0)`` for zero's own."""
    if index == 0:
        return (0.0, 0.0)
    k = abs(index) - BINADE_OFFSET
    return (2.0**k, 2.0 ** (k + 1)) if index > 0 else (-(2.0 ** (k + 1)), -(2.0**k))


# One binade of a cell: lanes present, lanes ranked, worst ranked lane (-1 if none).
Bin = tuple[int, int, int]


@dataclass
class _Run:
    first: int
    last: int
    ranked: bool
    past: bool
    lanes: int


def partition(
    bins: dict[int, Bin], ceiling: float
) -> tuple[tuple[float, float, bool], ...]:
    """The domains the rule draws on a cell's binade profile, as ``(low, high, ranked)``
    triples covering ``(-inf, inf)``.

    A binade with fewer than :data:`DOMAIN_MIN_BIN_LANES` lanes present takes its
    neighbour's class; with fewer ranked it is unranked; otherwise it is within the
    ceiling or past it. Adjacent binades of one class merge, a gateable run under
    :data:`DOMAIN_MIN_LANES` ranked lanes folds into the looser neighbour (never an
    unranked one), the outer edges are infinite, and an edge with no swept lane between
    it and zero is written as 0."""
    if not bins:
        return ((-_INF, _INF, False),)
    labelled: list[list] = []
    for index in sorted(bins):
        present, ranked, worst = bins[index]
        if present < DOMAIN_MIN_BIN_LANES:
            cls = None
        elif ranked < DOMAIN_MIN_BIN_LANES:
            cls = (False, False)
        else:
            cls = (True, worst > ceiling)
        labelled.append([index, ranked, cls])
    last = None
    for row in labelled:
        if row[2] is None:
            row[2] = last
        else:
            last = row[2]
    nxt = (False, False)
    for row in reversed(labelled):
        if row[2] is None:
            row[2] = nxt
        else:
            nxt = row[2]
    runs: list[_Run] = []
    for index, lanes, (ranked, past) in labelled:
        if runs and (runs[-1].ranked, runs[-1].past) == (ranked, past):
            runs[-1].last = index
            runs[-1].lanes += lanes
        else:
            runs.append(
                _Run(first=index, last=index, ranked=ranked, past=past, lanes=lanes)
            )
    changed = True
    while changed:
        changed = False
        narrow = sorted(
            (
                i
                for i, r in enumerate(runs)
                if r.ranked and not r.past and r.lanes < DOMAIN_MIN_LANES
            ),
            key=lambda i: runs[i].lanes,
        )
        for i in narrow:
            candidates = [
                j for j in (i - 1, i + 1) if 0 <= j < len(runs) and runs[j].ranked
            ]
            if not candidates:
                continue
            j = max(candidates, key=lambda j: runs[j].past)
            lo, hi = min(i, j), max(i, j)
            runs[lo : hi + 1] = [
                _Run(
                    first=runs[lo].first,
                    last=runs[hi].last,
                    ranked=True,
                    past=runs[j].past,
                    lanes=runs[lo].lanes + runs[hi].lanes,
                )
            ]
            changed = True
            break
    merged: list[_Run] = []
    for r in runs:
        if merged and (merged[-1].ranked, merged[-1].past) == (r.ranked, r.past):
            merged[-1].last = r.last
            merged[-1].lanes += r.lanes
        else:
            merged.append(_Run(**vars(r)))
    # The smallest magnitude any swept lane has: an edge at or below it has nothing
    # between it and zero, and reads as 0.
    smallest = min(
        (
            abs(edge)
            for index, (present, _, _) in bins.items()
            if index != 0 and present > 0
            for edge in binade_edges(index)
        ),
        default=0.0,
    )
    out: list[tuple[float, float, bool]] = []
    for k, r in enumerate(merged):
        low = -_INF if k == 0 else binade_edges(r.first)[0]
        high = _INF if k == len(merged) - 1 else binade_edges(r.last)[1]
        if 0 < abs(low) <= smallest:
            low = 0.0
        if 0 < abs(high) <= smallest:
            high = 0.0
        if out and not low < high:
            # Only the zero lane, snapped to [0, 0): dropped; the next entry's `low` is
            # pulled down to 0 below, so the zero lane lands in the domain after it.
            continue
        if out and out[-1][1] != low:
            low = out[-1][1]
        out.append((low, high, r.ranked))
    return tuple(out)


def entries_of(
    spans: tuple[tuple[float, float, bool], ...],
    worst: Callable[[float, float], int | None],
) -> tuple[Domain, ...]:
    """The entries for :func:`partition`'s *spans*, given ``worst(low, high)``: the worst
    ranked lane in ``[low, high)`` outside the floor, or ``None``. A ranked span carries
    its figure; an unranked one is null unless its strays are worse than every figure,
    when the span whose strays are worst takes its own. ``()`` (a plain row) when no
    ranked span measured anything."""
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


def check_domains(domains: tuple[Domain, ...]) -> None:
    """Refuse a list that is not a contiguous partition of ``(-inf, inf)`` with interior
    edges at 0 or a power of two, or that holds no figure."""
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


def largest_figure(domains: tuple[Domain, ...]) -> int:
    """The largest figure of a checked list (:func:`check_domains` guarantees one): what
    its null entries' ranked lanes are held to, by the sweep, ``diff`` and ``headroom``
    alike, and the whole row's figure."""
    return max(d.ulp for d in domains if d.ulp is not None)
