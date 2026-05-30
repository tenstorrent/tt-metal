"""Pin the wiring of agentic engine primitives (G4, G7, G8) into the
auto-iterate loop. These tests don't run the loop end-to-end — they
verify the integration points exist in source so a future refactor
can't silently re-decouple them."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_AUTO_ITER_PY = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "_cli_helpers" / "auto_iterate.py").read_text()


def test_g8_convergence_is_imported_in_auto_iterate() -> None:
    """G8: the per-component plateau detector uses agentic.convergence.is_stagnant.
    Without this, the loop's plateau handling falls back to ad-hoc heuristics."""
    assert "from ..agentic.convergence import is_stagnant" in _AUTO_ITER_PY


def test_g4_mechanical_actions_imported_in_auto_iterate() -> None:
    """G4: when G8 says plateau, the loop applies one untried mechanical
    action (cache invalidate, env var toggle, etc.) BEFORE the next LLM call.
    The action list comes from agentic.actions.default_mechanical_actions."""
    assert "from ..agentic.actions import default_mechanical_actions" in _AUTO_ITER_PY


def test_g7_learnings_lookup_at_preflight() -> None:
    """G7: at pre-flight, the loop tries lookup_fix() for each non-graduated
    component. If a prior run registered a fix for the same HF architecture,
    apply_fix() patches the stub before the loop iterates."""
    assert "from ..agentic.learnings import apply_fix" in _AUTO_ITER_PY
    assert "lookup_fix" in _AUTO_ITER_PY


def test_g7_learnings_register_on_graduation() -> None:
    """G7: every time a component graduates, the loop calls register_fix()
    with (arch_signature, component_name, stub_diff). Future runs read this
    via lookup_fix()."""
    assert "from ..agentic.learnings import compute_arch_signature, register_fix" in _AUTO_ITER_PY
    assert "registered learned fix" in _AUTO_ITER_PY


def test_pcc_history_per_component_state_exists() -> None:
    """G8 needs a list of per-component PCC values. The dict is appended
    on every PCC observation in _track_progress_and_record."""
    assert "pcc_history_per_component" in _AUTO_ITER_PY
    # And it's the (1 - pcc) mismatch-ratio convention used by the engine.
    assert "1.0 - float(pcc_value)" in _AUTO_ITER_PY


def test_tried_actions_per_component_state_exists() -> None:
    """Each component tracks its OWN tried-actions set so the same toggle
    isn't applied repeatedly across iters."""
    assert "tried_actions_per_component" in _AUTO_ITER_PY


# ---------------------------------------------------------------------------
# Auto-decomposition: the loop must AUTO-INVOKE decompose --write-plan
# in a subprocess (NOT just print a hint) when failure_class warrants it.
# ---------------------------------------------------------------------------


def test_auto_iterate_auto_invokes_decompose_in_subprocess() -> None:
    """Pin the auto-invocation: the loop must spawn the existing
    `tt-hw-planner decompose --write-plan` CLI in a subprocess on
    decomposition-worthy failures, not just print a hint. The
    `decomposition_consumer` at loop start picks up the plan next iter.
    """
    src = _AUTO_ITER_PY
    # The auto-invocation block must use a subprocess.run on the
    # decompose CLI command so the HF model load doesn't bloat the
    # auto-loop process.
    assert "decompose-auto" in src, "auto_iterate must AUTO-INVOKE decompose, not just print a manual hint"
    assert (
        "scripts.tt_hw_planner" in src and '"decompose"' in src
    ), "auto_iterate must spawn `python -m scripts.tt_hw_planner decompose ...`"
    assert "--write-plan" in src, "auto-decompose must write the plan file"


def test_auto_decompose_is_attempted_once_per_component() -> None:
    """Pin the dedup: decomposition_auto_attempted set prevents
    re-running the subprocess on every iter for the same parent."""
    src = _AUTO_ITER_PY
    assert "decomposition_auto_attempted" in src, (
        "auto_iterate must track which components have been auto-decomposed "
        "this run, to avoid re-running the subprocess on every iter"
    )


# ---------------------------------------------------------------------------
# Loop-exit escalation: when max_iters exhausts with components still pending,
# each pending component MUST go through the escalation chain (same path that
# per-component cap exhaustion uses) — not just get its stub rewritten to
# a stable fallback. Gap surfaced in the SAM2 brain test 2026-05-30.
# ---------------------------------------------------------------------------


def test_loop_exit_routes_still_pending_through_skip_to_fallback() -> None:
    """Pin: at loop-level max_iters exhaustion, still_pending components
    must be routed through _skip_component_to_fallback so the existing
    escalation chain (failure_classifier, decompose-auto, persist_skip,
    kernel_missing classification) fires for each one. Without this,
    _rewrite_components_to_stable_fallback runs but no classification
    or escalation happens.
    """
    src = _AUTO_ITER_PY
    # The still-pending branch must call _skip_component_to_fallback in
    # a loop over still_pending — BEFORE the stub-rewrite, so escalation
    # state is captured first.
    assert "for _pending_comp in still_pending:" in src, (
        "loop-exit must iterate still_pending and route each through " "_skip_component_to_fallback"
    )
    assert (
        "_skip_component_to_fallback(\n                    _pending_comp," in src
        or "_skip_component_to_fallback(_pending_comp," in src
    ), "loop-exit must call _skip_component_to_fallback per pending component"
    # And the reason string must indicate the trigger is max_iters
    assert "loop-level max_iters" in src, (
        "loop-exit escalation reason must distinguish max_iters exhaustion "
        "from per-component cap exhaustion (so future audits can trace it)"
    )


def test_parallel_extras_bump_attempts_per_component() -> None:
    """Pin: when a parallel-agent extra is scheduled, its attempt count
    must be bumped (same as the primary target). Without this, extras
    accumulate work but their attempts_per_component value stays at
    whatever they hit as PRIMARY — _is_at_cap() never fires for them
    and the per-component escalation chain is unreachable.
    """
    src = _AUTO_ITER_PY
    # The extras loop must bump attempts_per_component[_extra].
    assert "attempts_per_component[_extra] = attempts_per_component.get(_extra, 0) + 1" in src, (
        "parallel-extras spawn must increment attempts_per_component[_extra] "
        "so per-component cap-trigger logic applies"
    )
