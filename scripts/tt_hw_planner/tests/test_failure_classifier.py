"""Unit tests for `failure_classifier.classify_failure`.

Pins the 8-class decision tree: every input combination must map to a
deterministic class with a confidence label and an explanation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from scripts.tt_hw_planner.failure_classifier import (  # noqa: E402
    AGENT_STUCK,
    ALL_CLASSES,
    COLD_INTENDED,
    CONSTRAINT_MISMATCH,
    HF_ERROR,
    ITERATION_BUDGET,
    KERNEL_VERIFIED_MISSING,
    SKIP_CATEGORY_FOR_CLASS,
    TOOL_BUG,
    FailureVerdict,
    classify_failure,
    skip_category_for_verdict,
)


# --- top-level shape checks --------------------------------------------------


def test_all_classes_are_distinct_strings() -> None:
    assert len(set(ALL_CLASSES)) == len(ALL_CLASSES)


def test_skip_category_map_covers_every_class() -> None:
    """SKIP_CATEGORY_FOR_CLASS must have an entry for every class in
    ALL_CLASSES so the bridge from classifier verdict to persistent skip
    record never silently drops a class on the floor."""
    for cls in ALL_CLASSES:
        assert cls in SKIP_CATEGORY_FOR_CLASS, f"missing skip-category mapping for {cls}"


# --- 1. workload-probe COLD short-circuit ------------------------------------


def test_workload_cold_short_circuits_to_cold_intended() -> None:
    """A COLD verdict from hot_cold probe wins regardless of failure text."""
    v = classify_failure(
        reason="exhausted per-component attempt cap",
        failure_text="ttnn does not support permute on sparse_coo",
        hot_cold_kind="COLD",
    )
    assert v.class_name == COLD_INTENDED
    assert v.confidence == "high"


def test_workload_hot_does_not_short_circuit() -> None:
    v = classify_failure(reason="harness: missing positional arg", hot_cold_kind="HOT")
    assert v.class_name == TOOL_BUG  # falls through to step 2


# --- 2. TOOL_BUG patterns ----------------------------------------------------


def test_classifies_missing_positional_arg_as_tool_bug() -> None:
    v = classify_failure(reason="harness: missing 2 required positional args (v13)")
    assert v.class_name == TOOL_BUG
    assert v.confidence == "high"


def test_classifies_modulelist_no_forward_as_tool_bug() -> None:
    v = classify_failure(reason="harness: ModuleList no forward")
    assert v.class_name == TOOL_BUG


def test_classifies_groups_weight_shape_mismatch_as_tool_bug() -> None:
    v = classify_failure(reason="harness: groups=256 weight shape mismatch")
    assert v.class_name == TOOL_BUG


# --- 3. HF_ERROR patterns ----------------------------------------------------


def test_classifies_hf_from_pretrained_failure_as_hf_error() -> None:
    v = classify_failure(
        reason="model load failed",
        failure_text="transformers.AutoModel.from_pretrained failed: 404",
    )
    assert v.class_name == HF_ERROR


def test_classifies_trust_remote_code_required_as_hf_error() -> None:
    v = classify_failure(reason="HF reference forward error: trust_remote_code Required")
    assert v.class_name == HF_ERROR


# --- 4a. KERNEL_VERIFIED_MISSING ---------------------------------------------


def test_classifies_verified_missing_op_as_kernel_missing() -> None:
    """When detect_kernel_missing returns an op AND verify_ttnn_op_exists
    returns False, the verdict is KERNEL_VERIFIED_MISSING with the op
    recorded on the verdict."""
    with patch("scripts.tt_hw_planner.failure_classifier.detect_kernel_missing", return_value="ttnn.some_op"):
        with patch("scripts.tt_hw_planner.failure_classifier.verify_ttnn_op_exists", return_value=False):
            v = classify_failure(reason="agent stuck", failure_text="not implemented: ttnn.some_op")
    assert v.class_name == KERNEL_VERIFIED_MISSING
    assert v.confidence == "high"
    assert v.missing_op == "ttnn.some_op"


# --- 4b. CONSTRAINT_MISMATCH -------------------------------------------------


def test_classifies_present_op_with_constraint_hint_as_constraint_mismatch_high() -> None:
    with patch("scripts.tt_hw_planner.failure_classifier.detect_kernel_missing", return_value="ttnn.conv2d"):
        with patch("scripts.tt_hw_planner.failure_classifier.verify_ttnn_op_exists", return_value=True):
            v = classify_failure(
                reason="agent stuck",
                failure_text="dtype float16 not supported; ttnn.conv2d not implemented for float16",
            )
    assert v.class_name == CONSTRAINT_MISMATCH
    assert v.confidence == "high"
    assert v.missing_op == "ttnn.conv2d"


def test_classifies_present_op_without_constraint_hint_as_constraint_mismatch_medium() -> None:
    with patch("scripts.tt_hw_planner.failure_classifier.detect_kernel_missing", return_value="ttnn.linear"):
        with patch("scripts.tt_hw_planner.failure_classifier.verify_ttnn_op_exists", return_value=True):
            v = classify_failure(reason="not implemented: ttnn.linear")
    assert v.class_name == CONSTRAINT_MISMATCH
    assert v.confidence == "medium"


def test_unverifiable_op_does_not_claim_kernel_missing() -> None:
    """When verify_ttnn_op_exists returns None (couldn't check), the
    classifier must NOT claim KERNEL_MISSING — fall through to remaining
    signals."""
    with patch("scripts.tt_hw_planner.failure_classifier.detect_kernel_missing", return_value="ttnn.mystery"):
        with patch("scripts.tt_hw_planner.failure_classifier.verify_ttnn_op_exists", return_value=None):
            v = classify_failure(reason="exhausted per-component attempt cap")
    assert v.class_name != KERNEL_VERIFIED_MISSING
    assert v.class_name == ITERATION_BUDGET


# --- 5. AGENT_STUCK ----------------------------------------------------------


def test_classifies_no_op_escalation_as_agent_stuck() -> None:
    v = classify_failure(reason="NO_OP escalation: 3 consecutive byte-identical responses")
    assert v.class_name == AGENT_STUCK


def test_classifies_empty_agent_records_as_agent_stuck() -> None:
    v = classify_failure(reason="loop noticed empty agent records over 3 iters")
    assert v.class_name == AGENT_STUCK


# --- 6. ITERATION_BUDGET -----------------------------------------------------


def test_classifies_per_component_cap_as_iteration_budget() -> None:
    v = classify_failure(reason="exhausted per-component attempt cap during target pick")
    assert v.class_name == ITERATION_BUDGET


# --- Bug-I regression: every known auto_iterate reason string routes
# --- to a specific (non-default) class -----------------------------------


def test_classifies_failed_n_attempts_moving_on_as_iteration_budget() -> None:
    """Reason string from auto_iterate line ~3428."""
    v = classify_failure(reason="failed 4 attempt(s); moving on to next ungraduated component")
    assert v.class_name == ITERATION_BUDGET


def test_classifies_agent_no_code_as_agent_stuck() -> None:
    """Reason string from auto_iterate when the agent produces nothing."""
    v = classify_failure(
        reason="agent produced no code (likely component too complex/long for the LLM within the wall-clock budget)"
    )
    assert v.class_name == AGENT_STUCK


def test_classifies_final_sweep_pcc_revelation_as_iteration_budget() -> None:
    """Reason string when final-sweep PCC re-check reveals a missed failure."""
    v = classify_failure(
        reason="final-sweep revealed PCC failure that prior iterations missed (heuristic false positive)"
    )
    assert v.class_name == ITERATION_BUDGET


def test_classifies_snapshot_restore_regression_as_tool_bug() -> None:
    """Reason string when snapshot rollback doesn't restore a working state."""
    v = classify_failure(reason="regression survived snapshot restore (shared infra or test harness changed)")
    assert v.class_name == TOOL_BUG


def test_classifies_hit_attempt_cap_consec_same_class_as_iteration_budget() -> None:
    """Reason string for at-cap classification."""
    v = classify_failure(reason="hit per-component attempt cap (consec-same-class 5/5)")
    assert v.class_name == ITERATION_BUDGET


# --- 7. default ColD_INTENDED low-confidence ---------------------------------


def test_classifies_unknown_reason_as_cold_intended_low_confidence() -> None:
    v = classify_failure(reason="something exotic the loop didn't tag")
    assert v.class_name == COLD_INTENDED
    assert v.confidence == "low"


# --- skip_category_for_verdict mapping ---------------------------------------


def test_skip_category_kernel_verified_missing_maps_to_kernel_missing() -> None:
    v = FailureVerdict(class_name=KERNEL_VERIFIED_MISSING, confidence="high", reason="x")
    assert skip_category_for_verdict(v) == "KERNEL_MISSING"


def test_skip_category_cold_intended_maps_to_cold() -> None:
    v = FailureVerdict(class_name=COLD_INTENDED, confidence="high", reason="x")
    assert skip_category_for_verdict(v) == "COLD"


def test_skip_category_iteration_budget_preserved_distinctly() -> None:
    """ITERATION_BUDGET must round-trip distinctly so the next run
    knows to retry rather than treat as permanent COLD."""
    v = FailureVerdict(class_name=ITERATION_BUDGET, confidence="medium", reason="x")
    assert skip_category_for_verdict(v) == "ITERATION_BUDGET"


def test_skip_category_tool_bug_preserved_distinctly() -> None:
    v = FailureVerdict(class_name=TOOL_BUG, confidence="high", reason="x")
    assert skip_category_for_verdict(v) == "TOOL_BUG"


def test_skip_category_hf_error_preserved_distinctly() -> None:
    v = FailureVerdict(class_name=HF_ERROR, confidence="high", reason="x")
    assert skip_category_for_verdict(v) == "HF_ERROR"


# --- precedence ordering -----------------------------------------------------


def test_workload_cold_beats_tool_bug() -> None:
    """Even if the failure text screams TOOL_BUG, a COLD workload verdict
    is the right placement (component is not on the hot path)."""
    v = classify_failure(reason="harness: missing positional arg", hot_cold_kind="COLD")
    assert v.class_name == COLD_INTENDED


def test_tool_bug_beats_kernel_pattern() -> None:
    """TOOL_BUG fires before kernel-missing detection, because if the
    scaffolder produced bad inputs the kernel-missing trace is suspect."""
    v = classify_failure(
        reason="harness: missing positional arg",
        failure_text="not implemented: ttnn.foo",
    )
    assert v.class_name == TOOL_BUG


# --- empty-input / robustness ------------------------------------------------


def test_empty_inputs_default_to_low_confidence_cold_intended() -> None:
    """No reason, no failure text, no workload signal — safe-default
    placement is COLD with LOW confidence."""
    v = classify_failure(reason="", failure_text="", hot_cold_kind=None)
    assert v.class_name == COLD_INTENDED
    assert v.confidence == "low"


def test_classifier_does_not_crash_on_none_inputs() -> None:
    """The classifier must defend against None reason/failure_text."""
    v = classify_failure(reason="", failure_text="", hot_cold_kind=None)
    assert v.class_name in ALL_CLASSES


# --- auto_iterate integration: retryable category load behavior --------------


def test_auto_iterate_does_not_block_retryable_categories_permanently() -> None:
    """Audit bug-fix: ITERATION_BUDGET / AGENT_STUCK persisted from
    prior runs MUST NOT be auto-added to `permanently_skipped` on the
    next run — otherwise the new fine-grained category system buys
    nothing (every entry was treated as permanent before this fix).

    Source-grep pin: the skip-list loader in auto_iterate.py must
    filter by category against a `_PERMANENT_SKIP_CATEGORIES` set that
    EXCLUDES ITERATION_BUDGET and AGENT_STUCK.
    """
    src = (Path(__file__).resolve().parents[1] / "_cli_helpers" / "auto_iterate.py").read_text()
    assert "_PERMANENT_SKIP_CATEGORIES" in src, (
        "auto_iterate must declare the permanent-skip category set so " "retryable categories aren't blocked forever"
    )
    # The retryable categories must NOT be in the permanent set.
    perm_idx = src.find("_PERMANENT_SKIP_CATEGORIES = {")
    assert perm_idx >= 0
    end = src.find("}", perm_idx)
    block = src[perm_idx : end + 1]
    assert "ITERATION_BUDGET" not in block, "ITERATION_BUDGET must be retryable — not in _PERMANENT_SKIP_CATEGORIES"
    assert "AGENT_STUCK" not in block, "AGENT_STUCK must be retryable — not in _PERMANENT_SKIP_CATEGORIES"


def test_auto_iterate_emits_decompose_cta_via_class_only_gate() -> None:
    """Audit bug-fix: the CTA `[decompose-hint]` previously used
    `should_attempt_decomposition(parent_module=object(), ...)` which
    always returned False because `object()` lacks `named_children`.
    The fix is to use the class-only gate
    `failure_class_warrants_decomposition`.
    """
    src = (Path(__file__).resolve().parents[1] / "_cli_helpers" / "auto_iterate.py").read_text()
    assert "failure_class_warrants_decomposition" in src, (
        "auto_iterate must use the class-only decomposition gate "
        "(not should_attempt_decomposition + a placeholder object())"
    )
    # The buggy object() pattern must be gone.
    assert "parent_module=object()" not in src


def test_auto_iterate_unknown_category_defaults_to_permanent() -> None:
    """Audit bug-G regression: an unknown category string (typo, future
    tool version) MUST default to PERMANENT, not silently slip into the
    retryable pool. Safety: pre-audit behavior was "everything
    permanent"; we preserve that as the unknown-default."""
    src = (Path(__file__).resolve().parents[1] / "_cli_helpers" / "auto_iterate.py").read_text()
    # The loader must select retryable via an explicit RETRYABLE set,
    # not by excluding from the permanent set (which would let unknowns
    # leak into the retryable pool).
    assert "_RETRYABLE_SKIP_CATEGORIES" in src, (
        "auto_iterate must use an explicit retryable-category set so "
        "unknown / typo categories default to permanent (safer)"
    )
    retry_idx = src.find("_RETRYABLE_SKIP_CATEGORIES = {")
    assert retry_idx >= 0
    end = src.find("}", retry_idx)
    block = src[retry_idx : end + 1]
    # Only the two retryable classes should be in the set.
    for cls in ("ITERATION_BUDGET", "AGENT_STUCK"):
        assert cls in block, f"{cls} must be in the retryable set"
    for cls in ("COLD", "KERNEL_MISSING", "TOOL_BUG", "HF_ERROR", "CONSTRAINT_MISMATCH"):
        assert cls not in block, f"{cls} must NOT be in the retryable set"
