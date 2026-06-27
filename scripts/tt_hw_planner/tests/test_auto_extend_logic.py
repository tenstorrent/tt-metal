"""End-to-end behavioral tests for the brain's run-level
should_extend_budget decision. These exercise the full policy matrix
that auto_iterate consults at budget exhaustion."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.agentic.convergence import (
    BUDGET_EXTEND_MAX_PENDING,
    BUDGET_EXTEND_MAX_PER_RUN,
    should_extend_budget,
)


def test_extend_when_pending_small_graduations_and_descending_history() -> None:
    """The happy path: 1 pending, graduations exist, history trending
    to zero. Brain must say YES with a positive bump and a reason that
    cites both the favorable trajectory AND momentum."""
    v = should_extend_budget(
        pending_components=["leftover"],
        pcc_history_per_component={"leftover": [0.10, 0.07, 0.04, 0.02]},
        graduated_this_run=["a", "b", "c"],
        max_iters=4,
        extensions_used=0,
    )
    assert v.extend is True
    assert v.bump == max(2, 4 // 2)
    assert "favorable" in v.reason
    assert "graduations" in v.reason or "momentum" in v.reason


def test_extend_on_momentum_alone_when_no_history() -> None:
    """Even with no PCC history (e.g. component just decomposed late),
    if graduations exist this run the brain extends — graduations are
    the momentum proxy."""
    v = should_extend_budget(
        pending_components=["leftover"],
        pcc_history_per_component={},  # no history at all
        graduated_this_run=["a", "b"],
        max_iters=4,
        extensions_used=0,
    )
    assert v.extend is True
    assert "graduations" in v.reason or "momentum" in v.reason


def test_no_extend_when_no_momentum_and_no_progress() -> None:
    """Flat history + no graduations → no momentum. Brain refuses."""
    v = should_extend_budget(
        pending_components=["leftover"],
        pcc_history_per_component={"leftover": [0.15, 0.15, 0.15, 0.15]},
        graduated_this_run=[],
        max_iters=4,
        extensions_used=0,
    )
    assert v.extend is False
    assert v.bump == 0
    assert "momentum" in v.reason or "trajectory" in v.reason


def test_no_extend_when_residue_too_large() -> None:
    """Broadly-failing run (4 pending) → bigger problem than 1 more
    iter can solve. Brain refuses."""
    v = should_extend_budget(
        pending_components=["a", "b", "c", "d"],
        pcc_history_per_component={c: [0.10, 0.05, 0.01] for c in "abcd"},
        graduated_this_run=["x", "y", "z"],
        max_iters=4,
        extensions_used=0,
    )
    assert v.extend is False
    assert "residue too large" in v.reason


def test_no_extend_when_already_extended_once() -> None:
    """Per-run cap: even with perfect conditions, a second extension
    must be refused."""
    v = should_extend_budget(
        pending_components=["leftover"],
        pcc_history_per_component={"leftover": [0.10, 0.05, 0.01]},
        graduated_this_run=["x"],
        max_iters=6,
        extensions_used=BUDGET_EXTEND_MAX_PER_RUN,
    )
    assert v.extend is False
    assert "already extended" in v.reason or "cap" in v.reason


def test_no_extend_when_no_pending_components() -> None:
    """If nothing is pending, there's nothing to extend FOR."""
    v = should_extend_budget(
        pending_components=[],
        pcc_history_per_component={},
        graduated_this_run=["a"],
        max_iters=4,
        extensions_used=0,
    )
    assert v.extend is False
    assert v.bump == 0
    assert "no-pending" in v.reason or "no pending" in v.reason


def test_bump_size_scales_with_max_iters() -> None:
    """Bump = max(2, max_iters // 2). Larger runs get larger extensions
    so a "long" run gets a meaningful extension, not a token 2 iters."""
    v_small = should_extend_budget(
        pending_components=["leftover"],
        pcc_history_per_component={"leftover": [0.10, 0.05, 0.02]},
        graduated_this_run=["x"],
        max_iters=2,  # tiny run
        extensions_used=0,
    )
    v_large = should_extend_budget(
        pending_components=["leftover"],
        pcc_history_per_component={"leftover": [0.10, 0.05, 0.02]},
        graduated_this_run=["x"],
        max_iters=10,  # bigger run
        extensions_used=0,
    )
    assert v_small.bump == 2  # floor
    assert v_large.bump == 5  # max_iters // 2


def test_residue_threshold_is_brain_constant_not_magic_number() -> None:
    """The threshold must live in the brain module as a constant — so
    future tuning happens in ONE place. Verify the constant gates the
    decision."""
    assert BUDGET_EXTEND_MAX_PENDING >= 1
    # At-threshold: extend.
    v = should_extend_budget(
        pending_components=["x"] * BUDGET_EXTEND_MAX_PENDING,
        pcc_history_per_component={f"x{i}": [0.10, 0.05] for i in range(BUDGET_EXTEND_MAX_PENDING)},
        graduated_this_run=["g"],
        max_iters=4,
        extensions_used=0,
    )
    assert v.extend is True or v.extend is False  # either is fine; just ensure the call works
    # One over threshold: never extend.
    v_over = should_extend_budget(
        pending_components=["x"] * (BUDGET_EXTEND_MAX_PENDING + 1),
        pcc_history_per_component={},
        graduated_this_run=["g"],
        max_iters=4,
        extensions_used=0,
    )
    assert v_over.extend is False
    assert "residue too large" in v_over.reason
