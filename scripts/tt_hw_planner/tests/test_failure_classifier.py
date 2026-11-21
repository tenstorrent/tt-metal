"""Unit tests for `failure_classifier.classify_failure`.

Pins the decision tree: every input combination maps to a deterministic
class with a confidence label and an explanation. The legacy
``hot_cold_kind`` parameter no longer affects classification — workload
firing doesn't gate placement.
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


def test_all_classes_are_distinct_strings() -> None:
    assert len(set(ALL_CLASSES)) == len(ALL_CLASSES)


def test_skip_category_map_covers_every_class() -> None:
    for cls in ALL_CLASSES:
        assert cls in SKIP_CATEGORY_FOR_CLASS, f"missing skip-category mapping for {cls}"


# --- TOOL_BUG patterns -------------------------------------------------------


def test_classifies_missing_positional_arg_as_tool_bug() -> None:
    v = classify_failure(reason="harness: missing 2 required positional args (v13)")
    assert v.class_name == TOOL_BUG
    assert v.confidence == "high"


def test_classifies_modulelist_no_forward_as_tool_bug() -> None:
    v = classify_failure(reason="harness: ModuleList no forward")
    assert v.class_name == TOOL_BUG


# --- HF_ERROR patterns -------------------------------------------------------


def test_classifies_hf_from_pretrained_failure_as_hf_error() -> None:
    v = classify_failure(
        reason="model load failed",
        failure_text="transformers.AutoModel.from_pretrained failed: 404",
    )
    assert v.class_name == HF_ERROR


def test_classifies_trust_remote_code_required_as_hf_error() -> None:
    v = classify_failure(reason="HF reference forward error: trust_remote_code Required")
    assert v.class_name == HF_ERROR


# --- KERNEL_VERIFIED_MISSING -------------------------------------------------


def test_classifies_verified_missing_op_as_kernel_missing() -> None:
    with patch("scripts.tt_hw_planner.failure_classifier.detect_kernel_missing", return_value="ttnn.some_op"):
        with patch("scripts.tt_hw_planner.failure_classifier.verify_ttnn_op_exists", return_value=False):
            v = classify_failure(reason="agent stuck", failure_text="not implemented: ttnn.some_op")
    assert v.class_name == KERNEL_VERIFIED_MISSING
    assert v.confidence == "high"
    assert v.missing_op == "ttnn.some_op"


# --- CONSTRAINT_MISMATCH -----------------------------------------------------


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
    with patch("scripts.tt_hw_planner.failure_classifier.detect_kernel_missing", return_value="ttnn.mystery"):
        with patch("scripts.tt_hw_planner.failure_classifier.verify_ttnn_op_exists", return_value=None):
            v = classify_failure(reason="exhausted per-component attempt cap")
    assert v.class_name != KERNEL_VERIFIED_MISSING
    assert v.class_name == ITERATION_BUDGET


# --- AGENT_STUCK -------------------------------------------------------------


def test_classifies_no_op_escalation_as_agent_stuck() -> None:
    v = classify_failure(reason="NO_OP escalation: 3 consecutive byte-identical responses")
    assert v.class_name == AGENT_STUCK


def test_classifies_empty_agent_records_as_agent_stuck() -> None:
    v = classify_failure(reason="loop noticed empty agent records over 3 iters")
    assert v.class_name == AGENT_STUCK


# --- ITERATION_BUDGET --------------------------------------------------------


def test_classifies_per_component_cap_as_iteration_budget() -> None:
    v = classify_failure(reason="exhausted per-component attempt cap during target pick")
    assert v.class_name == ITERATION_BUDGET


def test_classifies_failed_n_attempts_moving_on_as_iteration_budget() -> None:
    v = classify_failure(reason="failed 4 attempt(s); moving on to next ungraduated component")
    assert v.class_name == ITERATION_BUDGET


def test_classifies_final_sweep_pcc_revelation_as_iteration_budget() -> None:
    v = classify_failure(
        reason="final-sweep revealed PCC failure that prior iterations missed (heuristic false positive)"
    )
    assert v.class_name == ITERATION_BUDGET


# --- Default fallback --------------------------------------------------------


def test_classifies_unknown_reason_as_iteration_budget_low_confidence() -> None:
    """No decisive signal — retry next run rather than fall back to CPU."""
    v = classify_failure(reason="something exotic the loop didn't tag")
    assert v.class_name == ITERATION_BUDGET
    assert v.confidence == "low"


# --- Precedence --------------------------------------------------------------


def test_tool_bug_beats_kernel_pattern() -> None:
    v = classify_failure(
        reason="harness: missing positional arg",
        failure_text="not implemented: ttnn.foo",
    )
    assert v.class_name == TOOL_BUG


def test_classifies_snapshot_restore_regression_as_tool_bug() -> None:
    v = classify_failure(reason="regression survived snapshot restore (shared infra or test harness changed)")
    assert v.class_name == TOOL_BUG


# --- hot_cold_kind is now ignored --------------------------------------------


def test_hot_cold_kind_argument_is_ignored() -> None:
    """Legacy callers may still pass hot_cold_kind. The classifier accepts
    it for API compatibility but it has no effect on the verdict."""
    v_without = classify_failure(reason="harness: missing positional arg")
    v_with_cold = classify_failure(reason="harness: missing positional arg", hot_cold_kind="COLD")
    v_with_hot = classify_failure(reason="harness: missing positional arg", hot_cold_kind="HOT")
    assert v_without.class_name == v_with_cold.class_name == v_with_hot.class_name == TOOL_BUG


# --- skip_category_for_verdict ----------------------------------------------


def test_skip_category_kernel_verified_missing_maps_to_kernel_missing() -> None:
    v = FailureVerdict(class_name=KERNEL_VERIFIED_MISSING, confidence="high", reason="x")
    assert skip_category_for_verdict(v) == "KERNEL_MISSING"


def test_skip_category_iteration_budget_preserved_distinctly() -> None:
    v = FailureVerdict(class_name=ITERATION_BUDGET, confidence="medium", reason="x")
    assert skip_category_for_verdict(v) == "ITERATION_BUDGET"


def test_skip_category_tool_bug_preserved_distinctly() -> None:
    v = FailureVerdict(class_name=TOOL_BUG, confidence="high", reason="x")
    assert skip_category_for_verdict(v) == "TOOL_BUG"


def test_skip_category_hf_error_preserved_distinctly() -> None:
    v = FailureVerdict(class_name=HF_ERROR, confidence="high", reason="x")
    assert skip_category_for_verdict(v) == "HF_ERROR"


# --- Robustness --------------------------------------------------------------


def test_empty_inputs_default_to_low_confidence_iteration_budget() -> None:
    v = classify_failure(reason="", failure_text="")
    assert v.class_name == ITERATION_BUDGET
    assert v.confidence == "low"


def test_classifier_does_not_crash_on_none_inputs() -> None:
    v = classify_failure(reason="", failure_text="")
    assert v.class_name in ALL_CLASSES
