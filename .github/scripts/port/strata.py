#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Choose which cases a performance band measures, and name the classes it measured them in.

The band budget is much smaller than the sweep -- two dozen measurements against a grid of a couple
hundred points -- so *which* cases get measured decides what the gates are capable of noticing. The
first version took a flat prefix, and `ledger.py` emits cases from `itertools.product` over sorted
grid keys, so the leading key varies slowest: a prefix of 24 out of 184 pinned one dtype and one
layout. Every gate downstream was working correctly on a sample that could not contain the failure.

So the selection is stratified. Cases are partitioned along whichever of their own fields behave
like discrete axes, the budget is spread across the partitions, and within a partition the picks are
spread across problem size with both extremes always included. Nothing here knows what op it is
looking at: the axes are discovered from the ledger, which derives them from the manifest's
`vector_map`, so an op with different parameters gets different axes for free.

Two properties are worth stating because the gates depend on them:

  Deterministic. The wall band and the device band are separate `measure.py` invocations, and
  gate.py joins their results by case. Same ledger and same budget must give the same picks, so
  there is no sampling here in the random sense -- only a fixed rule.

  Honest about its own limits. A budget smaller than the number of strata cannot cover them all,
  and a grid with many axes cannot be partitioned along all of them and still leave anything in
  each partition. Both situations are reported rather than silently accepted, because a coverage
  claim the sample cannot support is how the previous version misled us.
"""

from __future__ import annotations

import json

# A field has to repeat this many times per distinct value to count as a class rather than a
# continuum. A per-case tuple that is nearly unique would otherwise put one case in every stratum
# and leave the size spread nothing to spread over.
MIN_CASES_PER_VALUE = 2

STRATUM_ALL = "all"


def _canonical(value) -> str:
    """A hashable, order-independent rendering of a ledger value.

    Ledger cases arrive via JSON, so values are lists and dicts as often as scalars, and `default`
    covers anything the ledger stringified on its way out.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _axis_value(case: dict, axis: str):
    if axis.startswith("kwargs."):
        return (case.get("kwargs") or {}).get(axis.split(".", 1)[1])
    return case.get(axis)


def _element_count(case: dict) -> int:
    """Input element count, as a size proxy for spreading picks within a stratum.

    The input shape rather than anything op-specific: an op whose output size is driven by a kwarg
    still has its cost scale with the tensor it reads, and this only has to order cases, not
    predict their cost.
    """
    shape = case.get("shape") or []
    total = 1
    for dim in shape:
        try:
            total *= int(dim)
        except (TypeError, ValueError):
            return 0
    return total


def _ceiling(cases: list[dict], budget: int) -> int:
    """How many classes it is worth splitting these cases into.

    Two limits, and the tighter one wins. The budget, because a class that never gets measured is
    not a class the gates can say anything about. And the case count over `MIN_CASES_PER_VALUE`,
    because a partition holding one case each is per-case grading wearing a coverage report -- which
    also keeps an uncapped run, where the budget limit is vacuous, from producing 156 strata.
    """
    return max(2, min(budget or len(cases), len(cases) // MIN_CASES_PER_VALUE))


def discover_axes(cases: list[dict], budget: int = 0) -> list[str]:
    """Ledger fields that behave like discrete axes, coarsest first.

    `shape` is deliberately not a candidate. It is the one field that is always a continuum, and it
    is the field the within-stratum spread orders by, so stratifying on it would both explode the
    partition count and leave the spread nothing to do.

    Everything else is bounded by two things rather than by a magic number. A field cannot be an
    axis if it has more values than the budget has measurements, because a class that never gets
    measured is not a class the gates can say anything about; and it cannot be an axis if its values
    barely repeat, which is what separates a class from a continuum for an uncapped run where the
    first test is vacuous. Ordering by ascending cardinality matters because `choose_axes` adds axes
    only while they still fit, so the coarsest partitions -- the ones leaving the most cases per
    stratum -- get in first.
    """
    names = ["dtype", "layout"]
    names += [f"kwargs.{k}" for k in sorted({k for c in cases for k in (c.get("kwargs") or {})})]

    ceiling = _ceiling(cases, budget)
    ranked = []
    for axis in names:
        distinct = {_canonical(_axis_value(c, axis)) for c in cases}
        if 2 <= len(distinct) <= ceiling:
            ranked.append((len(distinct), axis))
    return [axis for _, axis in sorted(ranked)]


def _label(value) -> str:
    """`_canonical` with the quotes off plain strings, since these labels reach the PR body."""
    text = _canonical(value)
    return text[1:-1] if isinstance(value, str) else text


def stratum_key(case: dict, axes: list[str]) -> str:
    """Name the class a case belongs to. Readable on purpose: it reaches the PR body."""
    if not axes:
        return STRATUM_ALL
    return "|".join(f"{axis.rsplit('.', 1)[-1]}={_label(_axis_value(case, axis))}" for axis in axes)


def choose_axes(cases: list[dict], budget: int) -> tuple[list[str], list[str]]:
    """Take as many axes as the budget can still cover one case per stratum along.

    Dropping an axis is a real loss of resolution, so the dropped ones are returned rather than
    discarded -- a report that describes full coverage along two axes while a third went
    unpartitioned is overstating what it measured, and the PR body has to be able to say so.
    """
    chosen: list[str] = []
    dropped: list[str] = []
    ceiling = _ceiling(cases, budget)
    for axis in discover_axes(cases, budget):
        trial = chosen + [axis]
        if len({stratum_key(c, trial) for c in cases}) <= ceiling:
            chosen = trial
        else:
            dropped.append(axis)
    return chosen, dropped


def _allocate(sizes: list[int], budget: int) -> list[int]:
    """Split `budget` across strata proportionally, one each before anything is proportional.

    The floor of one is the whole point of stratifying: a stratum holding 3 of 184 cases would get
    nothing from a purely proportional split, and a stratum that is never measured is exactly the
    blind spot this module exists to close.
    """
    n = len(sizes)
    if n == 0 or budget <= 0:
        return [0] * n
    if budget <= n:
        # The caller has already chosen which strata survive a budget this tight.
        return [1] * budget + [0] * (n - budget)

    total = sum(sizes)
    quota = [1.0 + (budget - n) * (size / total) for size in sizes]
    alloc = [min(int(q), sizes[i]) for i, q in enumerate(quota)]

    # Hand out the remainder by largest fractional part, skipping strata already exhausted. Looping
    # until no stratum can take another case also absorbs a budget larger than the case count.
    by_remainder = sorted(range(n), key=lambda i: (-(quota[i] - int(quota[i])), i))
    left = budget - sum(alloc)
    while left > 0:
        progressed = False
        for i in by_remainder:
            if left == 0:
                break
            if alloc[i] < sizes[i]:
                alloc[i] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    return alloc


def _spread(cases: list[dict], count: int) -> list[dict]:
    """Pick `count` cases spread across problem size, both extremes included.

    The extremes are not decoration. A defect that does work per element where it should do work per
    page or per tile costs almost nothing on the smallest case in a stratum and dominates the
    largest, so a sample that omits the largest case cannot distinguish the two. That is a property
    of how scaling defects present, not a rule about any particular op.
    """
    ordered = sorted(cases, key=lambda c: (_element_count(c), c.get("case_id", "")))
    n = len(ordered)
    if count >= n:
        return ordered
    if count <= 1:
        # Room for exactly one: take the largest, where a scaling defect is loudest.
        return [ordered[-1]]
    picks = sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})
    return [ordered[i] for i in picks]


def plan_selection(cases: list[dict], budget: int) -> dict:
    """Choose the cases to measure and describe the coverage the choice achieves."""
    if not cases:
        return {
            "select": "stratified",
            "axes": [],
            "axes_dropped": [],
            "strata": {},
            "strata_unmeasured": [],
            "coverage_complete": False,
            "cases": [],
        }

    axes, dropped = choose_axes(cases, budget)
    groups: dict[str, list[dict]] = {}
    for case in cases:
        groups.setdefault(stratum_key(case, axes), []).append(case)

    # Largest strata first, so a budget too small to reach every stratum spends itself on the
    # classes holding the most cases rather than on whichever one sorts first alphabetically.
    ranked = sorted(groups, key=lambda k: (-len(groups[k]), k))
    effective = budget if budget else sum(len(v) for v in groups.values())
    survivors = ranked[:effective] if effective < len(ranked) else ranked
    alloc = _allocate([len(groups[k]) for k in survivors], effective)

    measured = dict(zip(survivors, alloc))
    selected: list[dict] = []
    for key in survivors:
        selected.extend(_spread(groups[key], measured[key]))
    ledger_order = {case.get("case_id"): i for i, case in enumerate(cases)}

    strata = {key: {"total": len(groups[key]), "measured": measured.get(key, 0)} for key in sorted(groups)}
    unmeasured = [key for key, entry in strata.items() if not entry["measured"]]
    # Complete means every class the harness chose to track got measured. A dropped axis is reported
    # beside it but does not falsify it: dropping a continuum is the correct call, so folding it in
    # here would leave the flag permanently false for any op with a per-case kwarg, and a flag that
    # is always false teaches the reader to skip it.
    return {
        "select": "stratified",
        "axes": axes,
        "axes_dropped": dropped,
        "strata": strata,
        "strata_unmeasured": unmeasured,
        "coverage_complete": not unmeasured,
        # Ledger order, not stratum order: the device band's positional profiler attribution is
        # easier to read when both bands walk the cases the same way the ledger lists them.
        "cases": sorted(selected, key=lambda c: ledger_order.get(c.get("case_id"), 0)),
    }
