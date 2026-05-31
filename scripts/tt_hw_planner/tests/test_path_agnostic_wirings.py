"""Regression tests for the path-agnostic wirings.

These pin that brain orchestration is UNIFORM across Path 1
(scaffold + iterate, used by SAM2) and Path 2 (ALREADY-SUPPORTED
fast-path, used by Qwen / medgemma).

Each wiring connects an existing brain primitive (in agentic/, or
failure_classifier / overlay_manager / activation_diff) to a code
path where it wasn't fired before.

History
-------
2026-05-31: pcc_repair.py deleted (whole-model retry loop was a
duplication of Path 1's per-component iterate flow). Wirings #4,
#5, #10 (pcc half), #11 used to assert pcc_repair internals; those
assertions are gone. Path 2 now escalates directly to Path 1 via
``_maybe_escalate_pcc_fail`` so the same brain primitives fire.

Helper consolidation: ``classify_failure`` + ``persist_skip`` are
called via the shared ``failure_classifier.classify_and_persist_skip``
function. ``_classify_and_persist_skip`` in runtime_repair.py is a
thin shim that delegates to it (WIRING #1 still pins the helper's
presence at every exit point)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _read(rel: str) -> str:
    return (_REPO_ROOT / rel).read_text()


# ---------------------------------------------------------------------------
# WIRING #1: runtime_repair classifies + persists on failure
# ---------------------------------------------------------------------------


def test_wiring_1_runtime_repair_classifies_and_persists_on_failure() -> None:
    """Pin: _runtime_repair_loop must call the shared
    classify_and_persist_skip helper when irrecoverable, so
    KERNEL_MISSING gaps are recorded for next-run head-start.

    Post-consolidation (2026-05-31): runtime_repair.py uses a thin
    shim ``_classify_and_persist_skip`` that delegates to
    failure_classifier.classify_and_persist_skip — which centralizes
    classify_failure + persist_skip + brain G8 trace."""
    src = _read("scripts/tt_hw_planner/_cli_helpers/runtime_repair.py")
    assert "WIRING #1" in src, "wiring #1 marker must be present"
    assert "_classify_and_persist_skip" in src, "module-level shim must exist"
    # Shim must delegate to the shared classify_and_persist_skip helper.
    assert "from ..failure_classifier import classify_and_persist_skip" in src
    # Must invoke from BOTH not-repairable AND exhaustion paths plus the
    # shim definition — 3 occurrences (def + 2 call sites).
    assert (
        src.count("_classify_and_persist_skip(") >= 3
    ), "shim must be called from at least 2 sites (not-repairable + exhaustion)"


def test_failure_classifier_exposes_shared_consolidated_helper() -> None:
    """Pin: the shared classify_and_persist_skip helper exists in
    failure_classifier and centralizes classify_failure + persist_skip
    + the brain G8 trace. This is the single canonical implementation
    that runtime_repair and any future caller share — no duplication."""
    src = _read("scripts/tt_hw_planner/failure_classifier.py")
    assert "def classify_and_persist_skip(" in src
    assert "classify_failure(" in src
    assert "persist_skip(" in src


# ---------------------------------------------------------------------------
# WIRING #2: runtime_repair calls register_fix on success
# ---------------------------------------------------------------------------


def test_wiring_2_runtime_repair_registers_learned_fix_on_success() -> None:
    """Pin: when _runtime_repair_loop graduates, it persists the fix
    to agentic.learnings so next run gets a head-start."""
    src = _read("scripts/tt_hw_planner/_cli_helpers/runtime_repair.py")
    assert "WIRING #2" in src
    assert "_brain_register_learned_fix" in src
    assert "from ..agentic.learnings import register_fix" in src
    # Called from the success-exit branch.
    fix_idx = src.find("REPAIR LOOP GRADUATED at iter")
    assert fix_idx >= 0
    region = src[fix_idx : fix_idx + 600]
    assert "_brain_register_learned_fix(" in region


# ---------------------------------------------------------------------------
# WIRING #3: runtime_repair consults lookup_fix on entry
# ---------------------------------------------------------------------------


def test_wiring_3_runtime_repair_consults_learned_fix_on_entry() -> None:
    """Pin: at runtime-repair entry, the loop must consult lookup_fix
    to see if a prior run registered a working fix for this
    arch+failure signature. The hit becomes a prompt head-start."""
    src = _read("scripts/tt_hw_planner/_cli_helpers/runtime_repair.py")
    assert "WIRING #3" in src
    assert "_brain_lookup_learned_fix_head_start" in src
    assert "from ..agentic.learnings import lookup_fix" in src


# ---------------------------------------------------------------------------
# WIRING #4 / #5 / #11: pcc_repair removed 2026-05-31
# ---------------------------------------------------------------------------
# These wirings used to assert pcc_repair internals. With pcc_repair.py
# deleted (whole-model retry loop was duplication of Path 1's
# per-component iterate flow), Path 2 escalates directly to Path 1
# via _maybe_escalate_pcc_fail in cli.py. The brain primitives that
# used to be wired here (localize_decode_divergence, classify+persist
# at exhaustion, should_extend_budget) now fire in Path 1 via
# auto_iterate._run_auto_iterate_loop.


def test_pcc_repair_file_deleted_no_orphan_imports() -> None:
    """Pin: pcc_repair.py is fully gone and nothing imports it.
    The escalation hook _maybe_escalate_pcc_fail in cli.py replaces
    every former call site."""
    repo = _REPO_ROOT
    pcc_path = repo / "scripts" / "tt_hw_planner" / "_cli_helpers" / "pcc_repair.py"
    assert not pcc_path.exists(), f"pcc_repair.py must be deleted (still at {pcc_path})"
    cli_src = _read("scripts/tt_hw_planner/cli.py")
    # No live imports — only historical-comment mentions are OK.
    assert "from ._cli_helpers.pcc_repair" not in cli_src
    assert "from ._cli_helpers import pcc_repair" not in cli_src


def test_maybe_escalate_pcc_fail_replaces_pcc_repair_at_call_sites() -> None:
    """Pin: every prior _pcc_repair_loop call site in cli.py is now
    _maybe_escalate_pcc_fail(...) — re-entry into cmd_up via Path 1."""
    src = _read("scripts/tt_hw_planner/cli.py")
    assert "_maybe_escalate_pcc_fail" in src
    # Hook must short-circuit when the backend already exists exactly
    # (avoid re-running cmd_auto_onboard for a model already wired).
    assert "pick_backend_with_quality" in src
    # The flag that prevents infinite re-entry loops.
    assert "_escalated_already" in src


# ---------------------------------------------------------------------------
# WIRING #6: ALREADY-SUPPORTED PCC_FAIL_RC calls decide_demo_recovery
# ---------------------------------------------------------------------------


def test_wiring_6_already_supported_pcc_fail_consults_demo_recovery() -> None:
    """Pin: on Path 2 PCC failure, the brain's decide_demo_recovery
    primitive runs before the banner so its verdict reason is surfaced
    in OUTCOME extras."""
    src = _read("scripts/tt_hw_planner/cli.py")
    assert "WIRING #6" in src
    fix_idx = src.find("WIRING #6")
    region = src[fix_idx : fix_idx + 1500]
    assert "decide_demo_recovery" in region
    assert "demo-recovery verdict" in region
    # The recovery output must reach _final_outcome_banner via extra=.
    assert "_recovery_extra" in region


# ---------------------------------------------------------------------------
# WIRING #7: ALREADY-SUPPORTED success calls sync_graduated_to_main_tree
# ---------------------------------------------------------------------------


def test_wiring_7_already_supported_success_syncs_to_main_tree() -> None:
    """Pin: on Path 2 success, the worktree's edits must be synced to
    main tree. Without this, demo edits made during PCC-repair stay
    in the worktree and the user can't run the fixed demo in main."""
    src = _read("scripts/tt_hw_planner/cli.py")
    assert "WIRING #7" in src
    fix_idx = src.find("WIRING #7")
    region = src[fix_idx : fix_idx + 1500]
    assert "sync_graduated_to_main_tree" in region
    assert "from .agentic.persistence import sync_graduated_to_main_tree" in region


# ---------------------------------------------------------------------------
# WIRING #8: ALREADY-SUPPORTED success consults should_emit_e2e_demo
# ---------------------------------------------------------------------------


def test_wiring_8_already_supported_success_consults_emit_decision() -> None:
    """Pin: on Path 2 success, the brain's should_emit_e2e_demo
    decision runs so the user sees the same confidence label
    (HIGH/MIXED/LOW) Path 1 produces."""
    src = _read("scripts/tt_hw_planner/cli.py")
    assert "WIRING #8" in src
    fix_idx = src.find("WIRING #8")
    region = src[fix_idx : fix_idx + 1500]
    assert "should_emit_e2e_demo" in region
    assert "from .agentic.e2e import should_emit_e2e_demo" in region


# ---------------------------------------------------------------------------
# WIRING #9: runtime_repair uses is_stagnant
# ---------------------------------------------------------------------------


def test_wiring_9_runtime_repair_uses_brain_is_stagnant() -> None:
    """Pin: the runtime-repair stagnation detector consults brain's
    is_stagnant in addition to the local consecutive_no_edit_iters
    counter. Without this, the loop misses plateaus the brain would
    catch (e.g., LLM keeps editing but pytest output never changes)."""
    src = _read("scripts/tt_hw_planner/_cli_helpers/runtime_repair.py")
    assert "WIRING #9" in src
    assert "from ..agentic.convergence import is_stagnant" in src
    assert "_brain_is_stagnant(" in src
    # The termination condition must use BOTH local counter AND brain.
    # File-level check (the brain consult and the OR condition both
    # appear in the file, just not always within a contiguous window).
    assert "_brain_says_stuck" in src
    assert "or _brain_says_stuck" in src


# ---------------------------------------------------------------------------
# Imports are clean across all touched files
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WIRING #12: scaffold ALREADY-SUPPORTED gate bypassed on escalation
# ---------------------------------------------------------------------------


def test_wiring_12_scaffold_respects_escalation_bypass() -> None:
    """Pin: when ``_maybe_escalate_pcc_fail`` re-enters cmd_up with
    ``_escalated_already=True`` after Path 2 PCC failure, the
    Step 2/6 scaffold call MUST bypass the
    "ALREADY SUPPORTED — no scaffolding needed" gate.

    Without this bypass, the scaffold step aborts with rc=2 and the
    whole escalation is wasted — the Path 1 (scaffold + per-component
    iterate) flow never gets to run. Caught 2026-05-31 in the first
    Qwen2.5-14B rewire test."""
    sc_src = _read("scripts/tt_hw_planner/scaffold.py")
    cmd_sc_src = _read("scripts/tt_hw_planner/commands/scaffold.py")
    cli_src = _read("scripts/tt_hw_planner/cli.py")

    # plan_scaffold must accept the kwarg
    assert "force_already_supported: bool = False" in sc_src
    # Gate must respect the kwarg
    assert 'if compat.overall.startswith("ALREADY SUPPORTED") and not force_already_supported:' in sc_src
    # cmd_scaffold must thread the kwarg from args
    assert 'force_already_supported=getattr(args, "force_already_supported", False)' in cmd_sc_src
    # cmd_up's scaffold_argv must set force_already_supported from _escalated_already
    assert 'force_already_supported=bool(vars(args).get("_escalated_already"))' in cli_src


# ---------------------------------------------------------------------------
# WIRING #13: escalation scaffold writes bringup_status.json with REUSE->ADAPT
# ---------------------------------------------------------------------------


def test_wiring_13_escalation_writes_manifest_with_reuse_demoted_to_adapt() -> None:
    """Pin: when ``force_already_supported=True`` (escalation path),
    ``plan_scaffold``'s LLM/VLM branch must call ``build_bringup_plan``
    with ``force_adapt_all=True`` and emit a ``bringup_status.json``
    + ``BRING_UP_PLAN.md`` into the existing demo dir.

    Without this, an already-supported model whose global PCC gate
    fails has NO manifest for Path 1's per-component iterate loop to
    read — auto_iterate sees zero components, autofill aborts, and
    the brain has nothing to work with.

    The shift is: registry says "REUSE" based on static info, but the
    failed global PCC is runtime evidence the registry is wrong. The
    force-adapt-all override demotes every REUSE to ADAPT so the
    per-component PCC iterate loop actually verifies each one.
    Caught 2026-05-31 in the Qwen2.5-14B rewire test."""
    plan_src = _read("scripts/tt_hw_planner/bringup_plan.py")
    scaffold_src = _read("scripts/tt_hw_planner/scaffold.py")

    # build_bringup_plan must accept the kwarg
    assert "force_adapt_all: bool = False" in plan_src
    # And the demotion must actually happen
    assert "if force_adapt_all:" in plan_src
    assert "_c.status = ADAPT" in plan_src

    # scaffold's LLM/VLM branch must take the fast-path when
    # force_already_supported=True + compat is ALREADY SUPPORTED
    assert 'if force_already_supported and compat.overall.startswith("ALREADY SUPPORTED"):' in scaffold_src
    # And must call build_bringup_plan with force_adapt_all=True
    assert "force_adapt_all=True" in scaffold_src
    # And must emit collect_bringup_plan_files into the changes list
    assert "collect_bringup_plan_files(" in scaffold_src


def test_all_touched_modules_import_cleanly() -> None:
    """Last-line-of-defense: every module touched by the wirings
    must still import. A syntax error or wrong import in any wire
    would catch here. pcc_repair.py is deliberately excluded —
    that file was deleted 2026-05-31."""
    from scripts.tt_hw_planner import cli  # noqa: F401
    from scripts.tt_hw_planner._cli_helpers import auto_iterate  # noqa: F401
    from scripts.tt_hw_planner._cli_helpers import runtime_repair  # noqa: F401
    from scripts.tt_hw_planner.agentic import learnings  # noqa: F401
    from scripts.tt_hw_planner.agentic import persistence  # noqa: F401
    from scripts.tt_hw_planner import failure_classifier  # noqa: F401
    from scripts.tt_hw_planner import overlay_manager  # noqa: F401
