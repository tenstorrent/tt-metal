"""Tests for ``_should_snapshot_best_native`` — the pure decision
function backing the ``.best_native`` snapshot logic in
``_run_auto_iterate_loop``.

Background — pre-2026-06-03 bug:

  ``_snapshot_best_native_stub``'s docstring lists four rules:

    1. SKIP if the stub is a torch wrapper.
    2. WRITE if no prior `.best_native` exists.
    3. WRITE if PCC strictly improves over the recorded best.
    4. SKIP if PCC is None — we have no quality signal, and we
       don't want to overwrite a known-good 0.88 with an
       unmeasured stub from a different iter.

  Rules 1, 2, 3 were enforced. **Rule 4 was missing from the code.**
  A TT_FATAL iter (where pcc is None because the test crashed before
  the PCC assert) would silently overwrite a prior measurable
  snapshot — exactly what happened in the Phi-3.5 attention run,
  where the iter_005 use_hf_rope=True stub TT_FATAL'd at runtime and
  there was a real risk of clobbering an earlier higher-PCC body.

This module pins the four rules so future refactors can't silently
drop one again.
"""

from __future__ import annotations

import pytest


# ─── rule 2: no prior snapshot → write any native body ──────────────


def test_writes_when_no_prior_snapshot_exists():
    from scripts.tt_hw_planner._cli_helpers.auto_iterate import _should_snapshot_best_native

    assert (
        _should_snapshot_best_native(snap_exists=False, prior_pcc=None, new_pcc=0.5) is True
    ), "no prior → write (any native body > nothing)"


def test_writes_when_no_prior_snapshot_even_if_pcc_none():
    """If no prior snapshot exists, we write even with no quality
    signal — having SOMETHING native captured beats having nothing.
    The "skip if PCC is None" rule only protects an EXISTING snapshot.
    """
    from scripts.tt_hw_planner._cli_helpers.auto_iterate import _should_snapshot_best_native

    assert _should_snapshot_best_native(snap_exists=False, prior_pcc=None, new_pcc=None) is True


# ─── rule 4 (the missing one): preserve prior when new PCC is None ──


def test_skips_when_pcc_is_none_and_prior_exists():
    """REGRESSION: this is the rule that was missing from the code.
    A TT_FATAL'd iter (no measurable PCC) must NOT overwrite a known
    snapshot."""
    from scripts.tt_hw_planner._cli_helpers.auto_iterate import _should_snapshot_best_native

    assert (
        _should_snapshot_best_native(snap_exists=True, prior_pcc=0.88, new_pcc=None) is False
    ), "TT_FATAL iter (pcc=None) must not clobber a prior measurable snapshot"


def test_skips_when_pcc_is_none_even_if_prior_pcc_unknown():
    """When the prior snapshot has no tracked PCC (best_pcc_per_component
    miss) but a snapshot file exists, the new None-PCC stub still must
    not clobber it — we still have no way to know the new is better."""
    from scripts.tt_hw_planner._cli_helpers.auto_iterate import _should_snapshot_best_native

    assert _should_snapshot_best_native(snap_exists=True, prior_pcc=None, new_pcc=None) is False


# ─── rule 3: write when PCC strictly improves ───────────────────────


def test_writes_when_pcc_strictly_improves():
    from scripts.tt_hw_planner._cli_helpers.auto_iterate import _should_snapshot_best_native

    assert _should_snapshot_best_native(snap_exists=True, prior_pcc=0.7, new_pcc=0.9) is True


def test_skips_when_pcc_equals_prior():
    """Strict improvement only — equal PCC doesn't justify a write."""
    from scripts.tt_hw_planner._cli_helpers.auto_iterate import _should_snapshot_best_native

    assert _should_snapshot_best_native(snap_exists=True, prior_pcc=0.88, new_pcc=0.88) is False


def test_skips_when_pcc_regresses():
    from scripts.tt_hw_planner._cli_helpers.auto_iterate import _should_snapshot_best_native

    assert _should_snapshot_best_native(snap_exists=True, prior_pcc=0.88, new_pcc=0.5) is False


# ─── rule 3 corner: prior was unmeasured, new is measurable ─────────


def test_writes_when_prior_pcc_unmeasured_and_new_is_measurable():
    """If the existing snapshot got captured under rule 2 (no prior, so
    we wrote regardless of PCC), best_pcc_per_component may have no
    entry for this component. A subsequent measurable iter should
    REPLACE that unmeasured floor."""
    from scripts.tt_hw_planner._cli_helpers.auto_iterate import _should_snapshot_best_native

    assert _should_snapshot_best_native(snap_exists=True, prior_pcc=None, new_pcc=0.7) is True


# ─── Phi-3.5 scenario reproduced ────────────────────────────────────


def test_phi_3_5_iter_5_does_not_clobber_a_prior_measurable_snapshot():
    """Phi-3.5 attention iter_005 wrote `use_hf_rope=True` and TT_FATAL'd
    (iter_pcc=None). If an earlier iter had captured a measurable PCC
    snapshot, this iter must NOT replace it."""
    from scripts.tt_hw_planner._cli_helpers.auto_iterate import _should_snapshot_best_native

    assert (
        _should_snapshot_best_native(snap_exists=True, prior_pcc=0.65, new_pcc=None) is False
    ), "iter_005's TT_FATAL stub must not clobber a prior 0.65-PCC snapshot"


# ─── Static guard: the closure routes through the pure helper ───────
