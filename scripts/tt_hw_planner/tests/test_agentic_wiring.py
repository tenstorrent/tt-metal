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
