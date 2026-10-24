"""Unit tests for the strict logit-PCC gate in correctness/text.py.

Pins the behavior introduced by the 2026-06-02 audit:

  * ``load_reference`` always requests ``return_logits=True`` from HF
    (Gap 2 — decoupled from TT-side capture so a missing dump doesn't
    cascade into HF skipping logits too).
  * ``compare`` fail-CLOSES on every gap (Gap 3 — replaces the
    silent skip that let token-overlap stand in for the strict gate).

The strict gate's contract: SUCCESS verdict requires BOTH the token-
overlap heuristic AND the numerical logit-PCC ≥ 0.99 to fire AND pass.
Any missing input, any unverified output, any sub-threshold PCC = fail
(``result.ok=False``) so the caller escalates to Path A's per-component
loop.

These tests use lightweight stand-ins for ``Evidence``,
``ValidationResult``, and the HF reference object so the strict gate
logic can be exercised without importing torch/ttnn or running an
actual HF model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional
from unittest.mock import patch

import pytest

from scripts.tt_hw_planner.correctness.base import Evidence
from scripts.tt_hw_planner.correctness.text import TextComparator, _LOGIT_PCC_MIN
from scripts.tt_hw_planner.output_validation import ValidationResult


@dataclass
class _MockReference:
    """Stand-in for output_validation._HFRefOutput; only carries the
    attributes TextComparator.compare reads."""

    _model_id: str = "test/m"
    token_ids: List[int] = field(default_factory=list)
    text: str = ""
    step0_logits: Optional[Any] = None


def _passing_token_overlap_result() -> ValidationResult:
    """Build a ValidationResult that the token-overlap heuristic would
    have stamped as a pass. The strict gate's job is to either keep
    this verdict (PCC ≥ 0.99) or override it (any gap)."""
    return ValidationResult(
        ok=True,
        reason="OK: 20/32 tokens match (mismatch 38% < tolerance 70%)",
        compared_tokens=32,
        mismatch_count=12,
        mismatch_ratio=0.38,
    )


def _make_comparator() -> TextComparator:
    """TextComparator has no constructor args; one factory keeps the
    tests symmetric if the constructor changes."""
    return TextComparator()


# ─── Gap 2: load_reference always asks HF for logits ─────────────────


def test_load_reference_always_requests_logits_when_tt_path_present() -> None:
    """Sanity baseline: when TT side captured logits, HF must too.
    (Old behavior already did this; pinning to prevent regression.)"""
    ev = Evidence(payload="hello world", input_hint="hello")
    setattr(ev, "_tt_logits_path", "/tmp/tt_logits.npy")
    with patch("scripts.tt_hw_planner.output_validation.generate_hf_reference") as mock_gen:
        mock_gen.return_value = _MockReference()
        _make_comparator().load_reference(ev, "test/m")
    mock_gen.assert_called_once()
    assert mock_gen.call_args.kwargs.get("return_logits") is True


def test_load_reference_always_requests_logits_when_tt_path_missing() -> None:
    """Gap 2: even when TT side did NOT capture (no _tt_logits_path
    set on Evidence), HF must still be asked for logits. Decouples
    the HF capture from the TT capture so they each fail/succeed
    independently rather than cascading."""
    ev = Evidence(payload="hello world", input_hint="hello")
    # Deliberately NOT setting _tt_logits_path
    with patch("scripts.tt_hw_planner.output_validation.generate_hf_reference") as mock_gen:
        mock_gen.return_value = _MockReference()
        _make_comparator().load_reference(ev, "test/m")
    mock_gen.assert_called_once()
    assert mock_gen.call_args.kwargs.get("return_logits") is True, (
        "HF reference must always capture step-0 logits, even when TT-side dump "
        "is missing — the strict gate fail-closes on the missing TT side, but the "
        "decision lives in compare(), not load_reference()."
    )


def test_load_reference_raises_without_input_hint() -> None:
    """Sanity: empty prompt is a setup error, not a soft skip."""
    ev = Evidence(payload="x", input_hint=None)
    with pytest.raises(RuntimeError, match="input_hint"):
        _make_comparator().load_reference(ev, "test/m")


# ─── Gap 3: compare() fail-closes on every kind of missing input ──────


def _patch_tokenize_and_overlap(monkeypatch, overlap_result: ValidationResult):
    """Stub out the token-overlap path so tests focus on the strict
    gate. Returns the patched ``compare_token_sequences`` mock for any
    test that wants to assert on its call."""
    monkeypatch.setattr(
        "scripts.tt_hw_planner.output_validation.tokenize_text_for_compare",
        lambda _model_id, _text: [1, 2, 3],
    )
    monkeypatch.setattr(
        "scripts.tt_hw_planner.output_validation.compare_token_sequences",
        lambda *a, **kw: overlap_result,
    )


def test_compare_fail_closes_when_tt_logits_path_missing(monkeypatch) -> None:
    """Gap 3a: TT-side did not emit ``==LOGITS PATH:`` → result.ok flips
    to False even though token-overlap would have passed."""
    _patch_tokenize_and_overlap(monkeypatch, _passing_token_overlap_result())
    ev = Evidence(payload="text", input_hint="hello")
    # _tt_logits_path NOT set
    ref = _MockReference(step0_logits=[0.1, 0.2, 0.3])
    result = _make_comparator().compare(ev, ref)
    assert result.ok is False
    assert "UNVERIFIED" in result.reason
    assert "TT-side step-0 logits not captured" in result.reason


def test_compare_fail_closes_when_hf_logits_missing(monkeypatch) -> None:
    """Gap 3b: TT side captured but HF did not return logits → fail."""
    _patch_tokenize_and_overlap(monkeypatch, _passing_token_overlap_result())
    ev = Evidence(payload="text", input_hint="hello")
    setattr(ev, "_tt_logits_path", "/tmp/tt_logits.npy")
    ref = _MockReference(step0_logits=None)  # HF reference is None
    result = _make_comparator().compare(ev, ref)
    assert result.ok is False
    assert "UNVERIFIED" in result.reason
    assert "HF reference did not return step-0 logits" in result.reason


def test_compare_fail_closes_when_pcc_computation_returns_none(monkeypatch) -> None:
    """Gap 3c: both inputs present but the comparison itself fails
    (file unreadable, shape mismatch, torch not available, …) → fail."""
    _patch_tokenize_and_overlap(monkeypatch, _passing_token_overlap_result())
    monkeypatch.setattr(
        "scripts.tt_hw_planner.correctness.text._compute_step0_logit_pcc",
        lambda _path, _hf: None,
    )
    ev = Evidence(payload="text", input_hint="hello")
    setattr(ev, "_tt_logits_path", "/tmp/tt_logits.npy")
    ref = _MockReference(step0_logits=[0.1, 0.2, 0.3])
    result = _make_comparator().compare(ev, ref)
    assert result.ok is False
    assert "UNVERIFIED" in result.reason
    assert "could not compute PCC" in result.reason


def test_compare_fails_when_pcc_below_threshold(monkeypatch) -> None:
    """Gap 3 baseline: PCC computed but < 0.99 → existing fail behavior
    preserved (pinning to make sure the refactor didn't break it)."""
    _patch_tokenize_and_overlap(monkeypatch, _passing_token_overlap_result())
    monkeypatch.setattr(
        "scripts.tt_hw_planner.correctness.text._compute_step0_logit_pcc",
        lambda _path, _hf: 0.85,
    )
    ev = Evidence(payload="text", input_hint="hello")
    setattr(ev, "_tt_logits_path", "/tmp/tt_logits.npy")
    ref = _MockReference(step0_logits=[0.1, 0.2, 0.3])
    result = _make_comparator().compare(ev, ref)
    assert result.ok is False
    assert "LOGIT-PCC FAIL" in result.reason
    assert "0.8500" in result.reason


def test_compare_passes_when_pcc_at_or_above_threshold(monkeypatch) -> None:
    """The happy path: token-overlap pass + PCC ≥ 0.99 → SUCCESS.
    Verifies both gates passing is the ONLY way to keep result.ok=True."""
    _patch_tokenize_and_overlap(monkeypatch, _passing_token_overlap_result())
    monkeypatch.setattr(
        "scripts.tt_hw_planner.correctness.text._compute_step0_logit_pcc",
        lambda _path, _hf: 0.995,
    )
    ev = Evidence(payload="text", input_hint="hello")
    setattr(ev, "_tt_logits_path", "/tmp/tt_logits.npy")
    ref = _MockReference(step0_logits=[0.1, 0.2, 0.3])
    result = _make_comparator().compare(ev, ref)
    assert result.ok is True
    assert "logit-PCC: 0.9950" in result.reason


def test_compare_passes_at_exact_threshold(monkeypatch) -> None:
    """Boundary: PCC exactly at _LOGIT_PCC_MIN should pass (≥, not >).
    Pins the inequality choice so future edits don't accidentally make
    the threshold exclusive."""
    _patch_tokenize_and_overlap(monkeypatch, _passing_token_overlap_result())
    monkeypatch.setattr(
        "scripts.tt_hw_planner.correctness.text._compute_step0_logit_pcc",
        lambda _path, _hf: _LOGIT_PCC_MIN,
    )
    ev = Evidence(payload="text", input_hint="hello")
    setattr(ev, "_tt_logits_path", "/tmp/tt_logits.npy")
    ref = _MockReference(step0_logits=[0.1, 0.2, 0.3])
    result = _make_comparator().compare(ev, ref)
    assert result.ok is True


def test_compare_token_overlap_fail_still_runs_strict_gate(monkeypatch) -> None:
    """Even if token-overlap already failed, the strict gate must still
    run and report its number — both gates are independent contributors
    to SUCCESS. The reason should include both signals."""
    fail_overlap = ValidationResult(
        ok=False,
        reason="REPEAT: tokens repeat 87% > tolerance 50%",
        compared_tokens=32,
        mismatch_count=20,
    )
    _patch_tokenize_and_overlap(monkeypatch, fail_overlap)
    monkeypatch.setattr(
        "scripts.tt_hw_planner.correctness.text._compute_step0_logit_pcc",
        lambda _path, _hf: 0.50,
    )
    ev = Evidence(payload="text", input_hint="hello")
    setattr(ev, "_tt_logits_path", "/tmp/tt_logits.npy")
    ref = _MockReference(step0_logits=[0.1, 0.2, 0.3])
    result = _make_comparator().compare(ev, ref)
    assert result.ok is False
    # Strict gate's failure should be visible — the LOGIT-PCC line is the
    # canonical signal for "TT diverged from HF numerically", which the
    # repair loop needs to focus the next iteration on.
    assert "LOGIT-PCC FAIL" in result.reason
