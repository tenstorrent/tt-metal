"""I-5 tests: counters & exit policy with a directional metric block (PLAN section 5)."""

import pytest

from agent.exit_policy import check_exit


def _base(metric=None, **over):
    """State with a min-metric (wall_ms) not yet at target."""
    state = {
        "metric": metric
        or {
            "name": "wall_ms",
            "unit": "ms",
            "direction": "min",
            "baseline": 20.0,
            "current": 14.0,
            "target": 12.0,
        },
        "max_iter": 25,
        "iteration": 4,
        "candidates": ["a", "b"],
        "tried": ["a"],
    }
    state.update(over)
    return state


def _metric(direction, current, target, name="wall_ms"):
    return {
        "name": name,
        "unit": "ms",
        "direction": direction,
        "baseline": current,
        "current": current,
        "target": target,
    }


def test_check_exit_target_met_min_metric():
    # min: 14 <= 12? no -> continue; 11.9 <= 12 -> DONE.
    assert check_exit(_base(metric=_metric("min", 14.0, 12.0))) == "continue"
    assert check_exit(_base(metric=_metric("min", 11.9, 12.0))) == "DONE"


def test_check_exit_target_met_max_metric():
    # max (fps): current 6.5 >= target 6.45 -> DONE; 6.4 -> continue.
    assert check_exit(_base(metric=_metric("max", 6.5, 6.45, name="fps"))) == "DONE"
    assert check_exit(_base(metric=_metric("max", 6.4, 6.45, name="fps"))) == "continue"


def test_check_exit_max_iter():
    assert check_exit(_base(iteration=25)) == "STOPPED"
    assert check_exit(_base(iteration=30)) == "STOPPED"


def test_check_exit_no_untried_levers():
    assert check_exit(_base(candidates=["a", "b"], tried=["a", "b"])) == "STOPPED"


def test_check_exit_otherwise_continue():
    assert check_exit(_base()) == "continue"


def test_target_takes_precedence_over_budget():
    # Target met AND budget blown -> DONE wins.
    assert check_exit(_base(metric=_metric("min", 10.0, 12.0), cost_usd=99.0)) == "DONE"


def test_no_candidates_does_not_stop():
    assert check_exit(_base(candidates=[], tried=[])) == "continue"


def test_missing_metric_does_not_crash():
    # Before baseline is recorded there may be no usable current/target.
    assert check_exit(_base(metric={})) == "continue"


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        check_exit(_base(metric=_metric("sideways", 10.0, 12.0)))
