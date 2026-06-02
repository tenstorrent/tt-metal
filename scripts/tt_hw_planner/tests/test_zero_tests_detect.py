"""Unit tests for the zero-tests-collected short-circuit detector."""

from __future__ import annotations

import pytest

from scripts.tt_hw_planner._cli_helpers.error_patterns import (
    ZeroTestsCollected,
    detect_zero_tests_collected,
    format_zero_tests_message,
)


# ─── detect_zero_tests_collected ───────────────────────────────────


def test_phi35_style_all_deselected_line_matches() -> None:
    # Real output from the Phi-3.5 run that motivated this work.
    text = (
        "collected 1 item / 1 deselected / 0 selected\n"
        "============= 1 deselected, 1 warning in 0.01s ===============\n"
    )
    info = detect_zero_tests_collected(rc=5, text=text)
    assert info is not None
    assert info.flavor == "all_deselected"
    assert "deselected" in info.excerpt or "selected" in info.excerpt


def test_pytest_rc5_alone_is_enough() -> None:
    """rc=5 with no matching stdout marker still short-circuits."""
    info = detect_zero_tests_collected(rc=5, text="")
    assert info is not None
    assert info.flavor == "none_collected"


def test_rc0_is_not_matched() -> None:
    info = detect_zero_tests_collected(rc=0, text="passed in 1.2s")
    assert info is None


def test_rc1_with_actual_failure_is_not_matched() -> None:
    text = "FAILED test_demo.py::test_perf - AssertionError\n1 failed in 30s"
    info = detect_zero_tests_collected(rc=1, text=text)
    assert info is None


def test_collected_zero_phrase_matches() -> None:
    info = detect_zero_tests_collected(rc=5, text="no tests collected")
    assert info is not None
    assert info.flavor == "none_collected"


def test_pytest_not_found_error_matches() -> None:
    text = "ERROR: not found: models/foo/demo.py::test_bar\n"
    info = detect_zero_tests_collected(rc=5, text=text)
    assert info is not None
    assert info.flavor == "none_collected"


def test_collected_with_explicit_zero_selected() -> None:
    text = "collected 3 items / 3 deselected / 0 selected"
    info = detect_zero_tests_collected(rc=1, text=text)  # rc could be anything
    assert info is not None
    assert info.flavor == "all_deselected"


def test_summary_line_only_deselected_no_passed() -> None:
    """Pytest's summary `==== N deselected in 0.01s ====` shape."""
    text = "==== 5 deselected in 0.02s ===="
    info = detect_zero_tests_collected(rc=1, text=text)
    assert info is not None
    assert info.flavor == "all_deselected"


def test_summary_with_warnings_still_matches() -> None:
    text = "============ 1 deselected, 1 warning in 0.01s ============="
    info = detect_zero_tests_collected(rc=1, text=text)
    assert info is not None
    assert info.flavor == "all_deselected"


def test_normal_passed_summary_not_matched() -> None:
    text = "============ 3 passed, 1 warning in 12.34s ============="
    info = detect_zero_tests_collected(rc=0, text=text)
    assert info is None


def test_empty_text_with_non_5_rc_returns_none() -> None:
    assert detect_zero_tests_collected(rc=1, text="") is None
    assert detect_zero_tests_collected(rc=0, text="") is None


def test_some_deselected_but_others_passed_not_matched() -> None:
    """Realistic case: -k filter trims some, others run and pass.
    This is NOT zero-tests; don't short-circuit."""
    text = "============ 3 passed, 2 deselected in 12.34s ============="
    info = detect_zero_tests_collected(rc=0, text=text)
    assert info is None


# ─── format_zero_tests_message ─────────────────────────────────────


def test_format_all_deselected_mentions_selector() -> None:
    info = ZeroTestsCollected(flavor="all_deselected", excerpt="1 deselected, 1 warning in 0.01s")
    msg = format_zero_tests_message("microsoft/Phi-3.5-mini-instruct", info)
    assert "microsoft/Phi-3.5-mini-instruct" in msg
    assert "-k" in msg or "selector" in msg
    assert "parametrize" in msg
    assert "1 deselected" in msg


def test_format_none_collected_mentions_path() -> None:
    info = ZeroTestsCollected(flavor="none_collected", excerpt="rc=5")
    msg = format_zero_tests_message("foo/bar", info)
    assert "foo/bar" in msg
    assert "path" in msg.lower() or "test" in msg.lower()


def test_format_always_says_no_iteration_will_help() -> None:
    for flavor in ("all_deselected", "none_collected"):
        info = ZeroTestsCollected(flavor=flavor, excerpt="x")
        msg = format_zero_tests_message("m", info)
        # Must signal this is NOT a PCC fail (no iteration helps)
        assert "iteration" in msg.lower() or "will help" in msg.lower()
