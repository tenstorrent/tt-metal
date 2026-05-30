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
