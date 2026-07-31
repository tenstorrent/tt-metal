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
