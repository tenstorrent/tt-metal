"""Pin: the strict logit-PCC ≥ 0.99 gate must fire on the LLM/VLM
auto-up path.

The Phi-3.5 run on 2026-06-02 stamped SUCCESS with a 38%-token-mismatch
verdict because ``run_evidence_gate`` (the path the ``agentic`` engine
takes for LLM/VLM models) skipped the strict numerical-PCC check
entirely. The strict gate code IS implemented in
:class:`TextComparator.compare` but the engine routing in
``run_gate`` excludes LLM/VLM from the comparator path, so the
comparator's strict logic never ran.

The fix extracts the strict-gate logic into
:func:`apply_strict_logit_pcc_gate` and calls it from BOTH paths.
These tests guard the wiring on both ends.
"""

from __future__ import annotations

from typing import Optional

import pytest


# ─── apply_strict_logit_pcc_gate behavior ──────────────────────────


class _FakeResult:
    """Stand-in for ValidationResult — just needs .ok and .reason."""

    def __init__(self, ok: bool = True, reason: str = "20/32 token match"):
        self.ok = ok
        self.reason = reason


def test_strict_gate_missing_tt_logits_fails_closed():
    from scripts.tt_hw_planner.correctness.text import apply_strict_logit_pcc_gate

    r = _FakeResult(ok=True, reason="20/32 match")
    apply_strict_logit_pcc_gate(r, tt_logits_path=None, hf_logits=object())
    assert r.ok is False
    assert "LOGIT-PCC UNVERIFIED" in r.reason
    assert "no `==LOGITS PATH:` marker" in r.reason


def test_strict_gate_missing_hf_logits_fails_closed():
    from scripts.tt_hw_planner.correctness.text import apply_strict_logit_pcc_gate

    r = _FakeResult(ok=True, reason="20/32 match")
    apply_strict_logit_pcc_gate(r, tt_logits_path="/tmp/x.npy", hf_logits=None)
    assert r.ok is False
    assert "LOGIT-PCC UNVERIFIED" in r.reason
    assert "HF reference did not return step-0 logits" in r.reason


def test_strict_gate_passes_through_when_pcc_above_threshold(monkeypatch):
    from scripts.tt_hw_planner.correctness import text as _text_mod

    monkeypatch.setattr(_text_mod, "_compute_step0_logit_pcc", lambda *_a, **_k: 0.995)
    r = _FakeResult(ok=True, reason="20/32 match")
    _text_mod.apply_strict_logit_pcc_gate(r, tt_logits_path="/tmp/x.npy", hf_logits=[1.0])
    assert r.ok is True
    assert "logit-PCC: 0.9950" in r.reason


def test_strict_gate_fails_when_pcc_below_threshold(monkeypatch):
    from scripts.tt_hw_planner.correctness import text as _text_mod

    monkeypatch.setattr(_text_mod, "_compute_step0_logit_pcc", lambda *_a, **_k: 0.85)
    r = _FakeResult(ok=True, reason="20/32 match")
    _text_mod.apply_strict_logit_pcc_gate(r, tt_logits_path="/tmp/x.npy", hf_logits=[1.0])
    assert r.ok is False
    assert "LOGIT-PCC FAIL" in r.reason
    assert "PCC=0.8500" in r.reason


def test_strict_gate_fails_when_pcc_compute_returns_none(monkeypatch):
    from scripts.tt_hw_planner.correctness import text as _text_mod

    monkeypatch.setattr(_text_mod, "_compute_step0_logit_pcc", lambda *_a, **_k: None)
    r = _FakeResult(ok=True, reason="20/32 match")
    _text_mod.apply_strict_logit_pcc_gate(r, tt_logits_path="/tmp/x.npy", hf_logits=[1.0])
    assert r.ok is False
    assert "LOGIT-PCC UNVERIFIED" in r.reason
    assert "could not compute PCC" in r.reason


# ─── Wiring: run_evidence_gate MUST call the strict gate ──────────


def test_run_evidence_gate_invokes_strict_gate_source_level():
    """Source-level guard: run_evidence_gate must import and call
    ``apply_strict_logit_pcc_gate``. Without this, the LLM/VLM
    auto-up path stamps SUCCESS on token-overlap alone."""
    from pathlib import Path

    src = Path("scripts/tt_hw_planner/correctness/engine.py").read_text()
    fn_idx = src.find("def run_evidence_gate")
    assert fn_idx >= 0
    next_def_idx = src.find("\ndef ", fn_idx + 10)
    if next_def_idx == -1:
        next_def_idx = len(src)
    body = src[fn_idx:next_def_idx]
    assert "apply_strict_logit_pcc_gate" in body, (
        "run_evidence_gate must invoke apply_strict_logit_pcc_gate "
        "so the strict numerical PCC≥0.99 check fires for LLM/VLM. "
        "Without this, SUCCESS is stamped on token-overlap alone."
    )
    assert "_TT_LOGITS_PATH_RE" in body, "run_evidence_gate must scan captured_output for ==LOGITS PATH:"
    assert (
        "return_logits=True" in body
    ), "run_evidence_gate must call generate_hf_reference with return_logits=True so HF step-0 logits are available"


def test_text_comparator_compare_still_invokes_strict_gate_source_level():
    """Source-level guard: TextComparator.compare must continue to
    invoke the strict gate helper (we refactored its inline impl
    into a shared helper; the call site must remain)."""
    from pathlib import Path

    src = Path("scripts/tt_hw_planner/correctness/text.py").read_text()
    cls_idx = src.find("class TextComparator")
    assert cls_idx >= 0
    compare_idx = src.find("def compare", cls_idx)
    assert compare_idx >= 0
    next_def_idx = src.find("\n    def ", compare_idx + 10)
    if next_def_idx == -1:
        next_def_idx = len(src)
    body = src[compare_idx:next_def_idx]
    assert "apply_strict_logit_pcc_gate" in body, (
        "TextComparator.compare must call apply_strict_logit_pcc_gate "
        "so the strict numerical PCC≥0.99 check fires when the "
        "comparator-routing path is taken (non-LLM categories or "
        "future LLM routing changes)."
    )


def test_logit_pcc_threshold_is_099():
    from scripts.tt_hw_planner.correctness.text import _LOGIT_PCC_MIN

    assert _LOGIT_PCC_MIN == pytest.approx(
        0.99
    ), "spec requires PCC ≥ 0.99 — do not lower this without a separate decision"
