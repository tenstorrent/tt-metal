"""Pin the 2026-06-04 Phase-3 wiring fix: pre-flight UNVERIFIED NATIVE
detections must populate ``harness_skipped_this_run`` AND
``skip_reasons_this_run``, so the LLM Tier-2 skip_diagnoser fires on
them at iter-loop end.

Without this wiring, components classified as UNVERIFIED NATIVE during
pre-flight (the 4 seamless-m4t components: ``hifi_gan_residual_block``,
``decoder``, ``encoder``, ``hifi_gan``) never reached the diagnoser
even though the diagnoser was implemented and wired into the iter-loop
end. The diagnoser's gate checks ``harness_skipped_this_run`` — and
pre-flight wasn't populating it.
"""

from __future__ import annotations

from pathlib import Path


_AUTO_ITER = Path(__file__).resolve().parent.parent / "_cli_helpers" / "auto_iterate.py"


def _source() -> str:
    return _AUTO_ITER.read_text(encoding="utf-8")


def _preflight_unverified_native_block() -> str:
    """Extract the pre-flight UNVERIFIED NATIVE branch in run_auto_iterate_loop."""
    src = _source()
    # The block is identified by the print "pre-flight: `{comp}` UNVERIFIED NATIVE"
    marker = "pre-flight: `{comp}` UNVERIFIED NATIVE"
    idx = src.find(marker)
    if idx == -1:
        # Fallback: f-string formatting variations
        idx = src.find('"  pre-flight: `{comp}` UNVERIFIED NATIVE ')
    assert idx != -1, "Pre-flight UNVERIFIED NATIVE branch not found in auto_iterate.py"
    # Walk backwards to the enclosing `if any(any(m in r for m in _pf_harness_markers)` block
    block_start = src.rfind("if any(any(m in r for m in _pf_harness_markers)", 0, idx)
    assert block_start != -1, "Could not locate harness-marker conditional"
    # Walk forward to find the persist_skip call (or end of block)
    block_end = src.find("elif comp in _pf_failed", idx)
    if block_end == -1:
        block_end = idx + 2000
    return src[block_start:block_end]


def test_preflight_populates_harness_skipped_this_run() -> None:
    """The pre-flight UNVERIFIED NATIVE detection branch MUST add the
    component to harness_skipped_this_run. This is the trigger set for
    the LLM Tier-2 diagnoser."""
    block = _preflight_unverified_native_block()
    assert "harness_skipped_this_run.add(comp)" in block, (
        "Pre-flight UNVERIFIED NATIVE detection must add component to "
        "harness_skipped_this_run so Tier-2 diagnoser fires on it. "
        "Without this line, pre-flight harness skips never reach the "
        "diagnoser even though it's wired into iter-loop end."
    )


def test_preflight_populates_skip_reasons_this_run() -> None:
    """The pre-flight UNVERIFIED NATIVE branch must also store the
    skip reason in skip_reasons_this_run. Without the reason, the
    diagnoser has no context to patch the test scaffold."""
    block = _preflight_unverified_native_block()
    assert "skip_reasons_this_run[comp]" in block, (
        "Pre-flight UNVERIFIED NATIVE detection must record the skip "
        "reason in skip_reasons_this_run[comp] for the Tier-2 LLM "
        "diagnoser to read."
    )


def test_preflight_still_marks_unverified_native_this_run() -> None:
    """Sanity check: the existing behavior is preserved — components
    still get added to unverified_native_this_run (the OUTCOME-banner
    bucket). The Phase-3 fix ADDS to harness_skipped_this_run, doesn't
    REMOVE from unverified_native_this_run."""
    block = _preflight_unverified_native_block()
    assert "unverified_native_this_run.add(comp)" in block


def test_preflight_still_persists_skip_as_tool_bug() -> None:
    """Sanity check: the persistent skip-list entry stays — components
    that hit this branch are recorded as TOOL_BUG category for
    next-run visibility."""
    block = _preflight_unverified_native_block()
    assert 'persist_skip(MODEL, comp, _pf_reason_blob, category="TOOL_BUG")' in block


def test_skip_diagnoser_trigger_set_consistent_with_iter_loop_path() -> None:
    """Both pre-flight (Phase-3 fix) and iter-loop pytest analysis
    must use the SAME pair of sets (harness_skipped_this_run +
    skip_reasons_this_run). If anyone diverges in the future, the
    diagnoser's gate would fire inconsistently."""
    src = _source()
    # Count writes to each set across the module.
    harness_writes = src.count("harness_skipped_this_run.add(comp)")
    reason_writes = src.count("skip_reasons_this_run[comp]")
    assert harness_writes >= 2, (
        f"Expected at least 2 writes to harness_skipped_this_run "
        f"(seed pytest + iter-loop pytest + pre-flight Phase-3 fix), "
        f"got {harness_writes}. Verify pre-flight write is present."
    )
    assert reason_writes >= 2, f"Expected at least 2 writes to skip_reasons_this_run, " f"got {reason_writes}."
