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


# ---------------------------------------------------------------------------
# Auto-decomposition: the loop must AUTO-INVOKE decompose --write-plan
# in a subprocess (NOT just print a hint) when failure_class warrants it.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Loop-exit escalation: when max_iters exhausts with components still pending,
# each pending component MUST go through the escalation chain (same path that
# per-component cap exhaustion uses) — not just get its stub rewritten to
# a stable fallback. Gap surfaced in the SAM2 brain test 2026-05-30.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Investigative-mode preamble for PCC plateau
# ---------------------------------------------------------------------------


def test_investigative_mode_preamble_exists_in_cli() -> None:
    """Pin the cli._build_investigative_mode_preamble helper exists.
    Mirrors _build_forced_edit_preamble pattern; required for the
    auto_iterate plateau-prepend logic."""
    from scripts.tt_hw_planner.cli import _build_investigative_mode_preamble

    out = _build_investigative_mode_preamble(
        iter_idx=3,
        component="video_layer_norm",
        pcc_history=[0.987, 0.987, 0.987],
    )
    assert "INVESTIGATIVE MODE" in out
    assert "video_layer_norm" in out
    assert "Override any" in out and "DO NOT iterate" in out
    # Tells LLM about precision knobs
    assert "fp32_dest_acc_en" in out
    assert "HiFi4" in out
    # Tool freedom
    assert "Read / Edit / Write / Grep / Bash" in out


# ---------------------------------------------------------------------------
# Graduation chicken-and-egg fix
# ---------------------------------------------------------------------------
