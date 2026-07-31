"""Unit tests for ``_run_strict_pcc_gate``.

The 2026-06-02 audit found that the strict end-to-end PCC gate was
called from three duplicated blocks (Path 2, Path B, and a missing
Path A) with identical category-probing + dispatcher-invocation code.
The helper extracts that logic into a single, testable function so
all three paths share one gate-routing implementation.

These tests pin:

  * Auto-mode + strict_pcc + captured_output ⇒ the dispatcher is
    invoked with the right kwargs and the result is returned.
  * Any of (auto_mode=False, strict_pcc=False, empty captured_output)
    short-circuits to (None, None). Caller treats this as "gate
    didn't run for this invocation" — preserves legacy non-auto
    behavior and lets the new UNVERIFIED outcome at Path A surface
    correctly.
  * The category probe is best-effort: probe failures fall back to
    "LLM" rather than skipping the gate.
"""

from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.tt_hw_planner.cli import _run_strict_pcc_gate


def _args(strict_pcc: bool = True) -> argparse.Namespace:
    """Build a minimal Namespace with the attributes the helper reads."""
    return argparse.Namespace(
        strict_pcc=strict_pcc,
        pcc_engine="legacy",
        strict_pcc_tokens=None,
        no_instruct=False,
    )


def test_short_circuits_when_auto_mode_disabled() -> None:
    """Non-auto invocations should not gate (legacy behavior).
    Returns (None, None) without touching the dispatcher."""
    args = _args()
    with patch("scripts.tt_hw_planner.correctness.run_gate") as mock_gate:
        result, prompt = _run_strict_pcc_gate(args, "test/m", "captured", auto_mode=False)
    assert result is None
    assert prompt is None
    mock_gate.assert_not_called()


def test_short_circuits_when_strict_pcc_disabled() -> None:
    """``--no-strict-pcc`` operator opt-out must skip the gate even
    in auto mode. The gate is not the only correctness mechanism;
    operators may disable it for benchmark-only runs."""
    args = _args(strict_pcc=False)
    with patch("scripts.tt_hw_planner.correctness.run_gate") as mock_gate:
        result, prompt = _run_strict_pcc_gate(args, "test/m", "captured", auto_mode=True)
    assert result is None
    assert prompt is None
    mock_gate.assert_not_called()


def test_short_circuits_when_captured_output_empty() -> None:
    """No demo output ⇒ nothing to gate on. Returning (None, None)
    here is what makes Path A's caller correctly mark UNVERIFIED
    instead of false-greening on an empty gate run."""
    args = _args()
    with patch("scripts.tt_hw_planner.correctness.run_gate") as mock_gate:
        result, prompt = _run_strict_pcc_gate(args, "test/m", "", auto_mode=True)
    assert result is None
    assert prompt is None
    mock_gate.assert_not_called()


def test_invokes_dispatcher_with_correct_kwargs() -> None:
    """Happy path: when all preconditions hold, the helper forwards
    every gate-input to the dispatcher. Pin the kwarg shape so a
    dispatcher-signature change doesn't silently drop a field."""
    args = _args()
    args.pcc_engine = "evidence"
    args.strict_pcc_tokens = 64

    fake_result = MagicMock(ok=True, reason="test pass")
    with patch("scripts.tt_hw_planner.correctness.run_gate") as mock_gate, patch(
        "scripts.tt_hw_planner.probe.probe_model"
    ) as mock_probe:
        mock_probe.return_value = MagicMock(category="LLM")
        mock_gate.return_value = (fake_result, "repair prompt body")
        result, prompt = _run_strict_pcc_gate(args, "test/m", "USER OUTPUT", auto_mode=True)

    assert result is fake_result
    assert prompt == "repair prompt body"
    mock_gate.assert_called_once()
    kwargs = mock_gate.call_args.kwargs
    assert kwargs["category"] == "LLM"
    assert kwargs["model_id"] == "test/m"
    assert kwargs["captured_output"] == "USER OUTPUT"
    assert kwargs["engine"] == "evidence"
    assert kwargs["compare_tokens"] == 64
    assert kwargs["instruct"] is True


def test_probe_failure_falls_back_to_LLM_category() -> None:
    """Probe is best-effort. A probe crash must NOT skip the gate —
    LLM is the broad-cover comparator and the right default for
    text-generation models, which is what auto-up is built around."""
    args = _args()
    with patch("scripts.tt_hw_planner.correctness.run_gate") as mock_gate, patch(
        "scripts.tt_hw_planner.probe.probe_model", side_effect=RuntimeError("probe boom")
    ):
        mock_gate.return_value = (MagicMock(ok=True), None)
        _run_strict_pcc_gate(args, "test/m", "captured", auto_mode=True)

    mock_gate.assert_called_once()
    assert mock_gate.call_args.kwargs["category"] == "LLM"


def test_unknown_category_normalized_to_LLM() -> None:
    """Probe sometimes returns 'Unknown' for models that don't match
    a known category. Treat 'Unknown' the same as 'no probe result' —
    fall back to LLM so the gate still runs."""
    args = _args()
    with patch("scripts.tt_hw_planner.correctness.run_gate") as mock_gate, patch(
        "scripts.tt_hw_planner.probe.probe_model"
    ) as mock_probe:
        mock_probe.return_value = MagicMock(category="Unknown")
        mock_gate.return_value = (MagicMock(ok=True), None)
        _run_strict_pcc_gate(args, "test/m", "captured", auto_mode=True)

    assert mock_gate.call_args.kwargs["category"] == "LLM"


def test_returns_gate_result_unchanged() -> None:
    """The helper is a routing layer, not a verdict layer. It must
    return the dispatcher's result unchanged so the caller's escalation
    decision (escalate-to-Path-A vs mark-UNVERIFIED) sees the exact
    ValidationResult the dispatcher produced."""
    args = _args()
    fake_fail = MagicMock(ok=False, reason="LOGIT-PCC FAIL: 0.42")
    with patch("scripts.tt_hw_planner.correctness.run_gate") as mock_gate, patch(
        "scripts.tt_hw_planner.probe.probe_model"
    ) as mock_probe:
        mock_probe.return_value = MagicMock(category="LLM")
        mock_gate.return_value = (fake_fail, "fix the rope code")
        result, prompt = _run_strict_pcc_gate(args, "test/m", "captured", auto_mode=True)
    assert result is fake_fail
    assert result.ok is False
    assert prompt == "fix the rope code"
