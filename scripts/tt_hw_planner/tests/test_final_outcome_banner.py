"""Unit tests for ``_final_outcome_banner`` outcome labels.

The banner is the single machine-grep-able signal CI and downstream
scripts use to decide "did the bring-up succeed." The 2026-06-02 audit
showed the legacy SUCCESS/FAIL binary couldn't represent the case
where bring-up "succeeded" by the rc but the strict end-to-end PCC
gate never fired — Phi-3.5 got SUCCESS despite logits never being
captured. These tests pin:

  * Legacy rc-derived labels still work (back-compat for existing
    call sites that don't pass ``outcome=``).
  * Explicit outcome overrides rc-derivation (new call sites can
    stamp UNVERIFIED / SUCCESS-PARTIAL without changing their rc).
  * The label appears in the exact grep position downstream scripts
    look for — same line, same prefix.

The banner is a thin print helper, so the tests just capture stdout
and assert on substrings. Keep simple.
"""

from __future__ import annotations

import pytest

from scripts.tt_hw_planner.cli import (
    OUTCOME_FAIL,
    OUTCOME_SUCCESS,
    OUTCOME_SUCCESS_PARTIAL,
    OUTCOME_UNVERIFIED,
    _final_outcome_banner,
)


# ─── Legacy rc-derived labels (preserve back-compat) ─────────────────


def test_rc_zero_defaults_to_success(capsys) -> None:
    """No explicit outcome + rc=0 → SUCCESS. Existing Path A / B / 2
    success paths rely on this; pin it."""
    _final_outcome_banner(rc=0, model_id="test/m", path_label="A. test")
    out = capsys.readouterr().out
    assert "TT_HW_PLANNER OUTCOME: SUCCESS" in out
    assert "rc=0" in out
    assert "model=test/m" in out


def test_rc_nonzero_defaults_to_fail(capsys) -> None:
    """No explicit outcome + rc≠0 → FAIL. Pin so a refactor doesn't
    accidentally swallow non-zero rcs as success."""
    _final_outcome_banner(rc=1, model_id="test/m", path_label="A. test")
    out = capsys.readouterr().out
    assert "TT_HW_PLANNER OUTCOME: FAIL" in out
    assert "rc=1" in out


# ─── New explicit outcome labels ─────────────────────────────────────


def test_explicit_unverified_overrides_rc(capsys) -> None:
    """A successful rc with strict-gate-skipped → UNVERIFIED.
    This is the headline behavior change: rc=0 no longer auto-implies
    SUCCESS when the caller knows the verification wasn't complete."""
    _final_outcome_banner(
        rc=0,
        model_id="test/m",
        path_label="2. ALREADY-SUPPORTED",
        outcome=OUTCOME_UNVERIFIED,
    )
    out = capsys.readouterr().out
    assert "TT_HW_PLANNER OUTCOME: UNVERIFIED" in out
    assert "rc=0" in out  # rc unchanged — the label is the only signal
    assert "SUCCESS" not in out.split("UNVERIFIED")[0]  # SUCCESS never appears before UNVERIFIED


def test_explicit_success_partial_overrides_rc(capsys) -> None:
    """KERNEL_MISSING components on CPU fallback but PCC passes →
    SUCCESS-PARTIAL. rc stays 0 because the run did succeed in the
    permitted sense; the label tells downstream "kernel work pending"."""
    _final_outcome_banner(
        rc=0,
        model_id="test/m",
        path_label="A. Template + iterate",
        outcome=OUTCOME_SUCCESS_PARTIAL,
    )
    out = capsys.readouterr().out
    assert "TT_HW_PLANNER OUTCOME: SUCCESS-PARTIAL" in out


def test_explicit_outcome_none_uses_rc_derivation(capsys) -> None:
    """outcome=None is the back-compat path: behaves exactly like the
    legacy two-argument signature. Pin so future changes don't break
    call sites that don't pass outcome."""
    _final_outcome_banner(rc=0, model_id="test/m", path_label="A", outcome=None)
    out = capsys.readouterr().out
    assert "OUTCOME: SUCCESS" in out


# ─── Constants stay stable (downstream scripts depend on them) ──────


@pytest.mark.parametrize(
    "constant,expected",
    [
        (OUTCOME_SUCCESS, "SUCCESS"),
        (OUTCOME_SUCCESS_PARTIAL, "SUCCESS-PARTIAL"),
        (OUTCOME_UNVERIFIED, "UNVERIFIED"),
        (OUTCOME_FAIL, "FAIL"),
    ],
)
def test_outcome_constants_have_stable_string_values(constant: str, expected: str) -> None:
    """Constants are public API for downstream log-scrapers. Lock the
    string values so a rename here doesn't silently break CI dashboards
    or tooling that greps for "OUTCOME: SUCCESS-PARTIAL"."""
    assert constant == expected


# ─── Format invariants downstream tooling depends on ────────────────


def test_banner_contains_all_required_fields(capsys) -> None:
    """Every banner line carries rc, model, path — pin the 4-field
    format so log-parsers can rely on it."""
    _final_outcome_banner(
        rc=42,
        model_id="microsoft/Phi-3.5-mini-instruct",
        path_label="2. ALREADY SUPPORTED",
        outcome=OUTCOME_UNVERIFIED,
    )
    out = capsys.readouterr().out
    assert "TT_HW_PLANNER OUTCOME:" in out
    assert "rc=42" in out
    assert "model=microsoft/Phi-3.5-mini-instruct" in out
    assert "path=2. ALREADY SUPPORTED" in out


def test_extra_suggestions_are_emitted(capsys) -> None:
    """Existing behavior: 'Suggested next steps' block renders bullet
    list under the banner. Pin so refactors don't drop it — operators
    rely on these hints to decide what to do after a fail/unverified."""
    _final_outcome_banner(
        rc=1,
        model_id="test/m",
        path_label="x",
        extra=["check logs", "re-run with --auto-max-iters 24"],
    )
    out = capsys.readouterr().out
    assert "Suggested next steps:" in out
    assert "- check logs" in out
    assert "- re-run with --auto-max-iters 24" in out
