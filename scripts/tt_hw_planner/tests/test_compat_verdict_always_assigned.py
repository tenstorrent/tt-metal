"""`_aggregate_overall` must always assign a verdict.

Regression guard for the bug that aborted the FLUX.2 `text_encoder` fan-out with
``ERROR: unexpected compat verdict 'UNKNOWN'; refusing to scaffold``.

`CompatReport.overall` defaults to ``"UNKNOWN"``. `_aggregate_overall` walks
`_OVERALL_FROM_STATUSES` in order and assigns the first matching label. The final
entry is meant to be a terminal catch-all ("nothing MISSING, nothing PARTIAL, so
READY"), but its predicate returned ``[]`` -- an empty list of offending blocks,
which is falsy. So a model whose needed blocks were *all* SUPPORTED matched no
predicate, fell out of the loop, and kept the ``"UNKNOWN"`` default, which
`scaffold` refuses.

The perverse consequence: the *better* a model's compat result, the worse its
verdict. A partially-supported model got "FEASIBLE WITH WORK" and proceeded; a
fully-supported one was refused outright.
"""

from scripts.tt_hw_planner.compatibility import (
    BUILDING_BLOCKS,
    CheckResult,
    CompatReport,
    Effort,
    Status,
    _aggregate_overall,
)

_DEFAULT_VERDICT = CompatReport.__dataclass_fields__["overall"].default


def _report(*statuses: Status) -> CompatReport:
    """A report whose needed blocks carry `statuses` (real blocks, no fixtures)."""
    results = [
        CheckResult(
            block=BUILDING_BLOCKS[i % len(BUILDING_BLOCKS)],
            needed=True,
            status=st,
            effort=Effort.NONE,
            notes="",
        )
        for i, st in enumerate(statuses)
    ]
    return CompatReport(
        model_id="test",
        architecture_family="test",
        similar_supported_model=None,
        results=results,
    )


def test_all_supported_yields_ready_not_unknown():
    """The regression: every needed block SUPPORTED must be READY, never UNKNOWN."""
    report = _report(Status.SUPPORTED, Status.SUPPORTED, Status.SUPPORTED)
    _aggregate_overall(report)
    assert report.overall != _DEFAULT_VERDICT
    assert report.overall == "READY"


def test_verdict_is_never_left_at_the_dataclass_default():
    """No combination of block statuses may leave `overall` unassigned."""
    every = list(Status)
    combos = [(a,) for a in every] + [(a, b) for a in every for b in every]
    for combo in combos:
        report = _report(*combo)
        _aggregate_overall(report)
        assert report.overall != _DEFAULT_VERDICT, f"{combo} left the default verdict"


def test_catch_all_does_not_mask_blocked_or_partial():
    """Ordering still wins: MISSING -> BLOCKED, PARTIAL -> FEASIBLE WITH WORK."""
    blocked = _report(Status.SUPPORTED, Status.MISSING, Status.PARTIAL)
    _aggregate_overall(blocked)
    assert blocked.overall == "BLOCKED"

    partial = _report(Status.SUPPORTED, Status.PARTIAL)
    _aggregate_overall(partial)
    assert partial.overall == "FEASIBLE WITH WORK"


def test_unneeded_blocks_do_not_suppress_ready():
    """`by_status` filters on `needed`, so UNNEEDED blocks never gate the verdict."""
    report = _report(Status.SUPPORTED)
    report.results.append(
        CheckResult(
            block=BUILDING_BLOCKS[0],
            needed=False,
            status=Status.MISSING,
            effort=Effort.NONE,
            notes="",
        )
    )
    _aggregate_overall(report)
    assert report.overall == "READY"


def test_scaffold_accepts_the_catch_all_verdict():
    """The catch-all label must be one `scaffold` actually allows, or the fix is moot."""
    import inspect

    from scripts.tt_hw_planner import scaffold

    src = inspect.getsource(scaffold)
    assert '_allowed_verdicts = ("READY", "FEASIBLE WITH WORK")' in src
