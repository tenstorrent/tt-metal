"""Failure classification for stuck bring-up components.

When the auto-iterate loop can't graduate a component, the previous
behavior was to drop straight to COLD or KERNEL_MISSING based on a
regex pattern in the failure trace. That over-skips:

  * components whose failure is a transient agent miss (would converge
    with more iters);
  * components stuck on a TOOL bug (scaffolder produced bad inputs);
  * components where an HF reference error blocked everything;
  * components where a TTNN op exists but the call hit a dtype/layout/
    shape constraint (CONSTRAINT_MISMATCH, not KERNEL_MISSING);
  * components that simply ran out of iter budget but would graduate
    next run.

This module is the single source of truth for "why did this component
not graduate." Every skip path consults `classify_failure` BEFORE
deciding to mark the component COLD / KERNEL_MISSING / etc.

Categories
----------

  ITERATE_MORE          — PCC is close to threshold; budget remains;
                          retry rather than skip. Loop should not call
                          us in this state, but we surface it for sanity
                          checks.
  TOOL_BUG              — scaffolder-produced signature/shape mismatch.
                          The test inputs themselves are wrong; the
                          stub never had a chance. Re-scaffold + retry.
  HF_ERROR              — HF reference forward itself raised. Not a
                          TTNN issue. Block + surface to user.
  KERNEL_VERIFIED_MISSING
                        — pattern matched AND verify_ttnn_op_exists is
                          False AND no decomposition path exists.
                          Genuine TTNN dev work needed.
  CONSTRAINT_MISMATCH   — pattern matched BUT op is verified present.
                          Failure is dtype/layout/shape config, not a
                          missing kernel. Iterate with a constraint
                          hint.
  ITERATION_BUDGET      — hit per-component attempt cap with no other
                          decisive signal. Retry next run with bigger
                          budget; do not mark KERNEL_MISSING.
  AGENT_STUCK           — repeated NO_OP / byte-identical empty
                          responses. Agent is not engaging. Trigger
                          decomposition before skip.
  COLD_INTENDED         — workload probe (hot_cold.json) marked COLD;
                          component is not on the hot path. CPU is
                          correct placement.

Confidence
----------

`classify_failure` returns a `(class_name, confidence, reason)` tuple.
Confidence is "high", "medium", or "low" and is recorded in the
skip-list so future runs and human readers can see how committed the
tool is to the verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .kernel_missing import detect_kernel_missing, verify_ttnn_op_exists


# --- Class constants (string-typed so they round-trip through JSON) ------

ITERATE_MORE = "ITERATE_MORE"
TOOL_BUG = "TOOL_BUG"
HF_ERROR = "HF_ERROR"
KERNEL_VERIFIED_MISSING = "KERNEL_VERIFIED_MISSING"
CONSTRAINT_MISMATCH = "CONSTRAINT_MISMATCH"
ITERATION_BUDGET = "ITERATION_BUDGET"
AGENT_STUCK = "AGENT_STUCK"
COLD_INTENDED = "COLD_INTENDED"

ALL_CLASSES = (
    ITERATE_MORE,
    TOOL_BUG,
    HF_ERROR,
    KERNEL_VERIFIED_MISSING,
    CONSTRAINT_MISMATCH,
    ITERATION_BUDGET,
    AGENT_STUCK,
    COLD_INTENDED,
)


# Map classifier classes back to the skip-list `category` field schema
# used by `final_categorization` + `overlay_manager.persist_skip`. Three
# of the eight classes map to existing skip-list categories; the rest
# carry their full class name through so future categorization steps
# can distinguish them (e.g. retry ITERATION_BUDGET next run vs. surface
# TOOL_BUG to the user).
SKIP_CATEGORY_FOR_CLASS = {
    KERNEL_VERIFIED_MISSING: "KERNEL_MISSING",
    COLD_INTENDED: "COLD",
    CONSTRAINT_MISMATCH: "CONSTRAINT_MISMATCH",
    TOOL_BUG: "TOOL_BUG",
    HF_ERROR: "HF_ERROR",
    ITERATION_BUDGET: "ITERATION_BUDGET",
    AGENT_STUCK: "AGENT_STUCK",
    ITERATE_MORE: "COLD",  # safe fallback if loop reaches us here
}


@dataclass
class FailureVerdict:
    class_name: str
    confidence: str  # "high" | "medium" | "low"
    reason: str
    missing_op: Optional[str] = None  # set when class is KERNEL_VERIFIED_MISSING / CONSTRAINT_MISMATCH


# --- Pattern banks ---------------------------------------------------------

# TOOL_BUG signals: failures the *harness* (test scaffolder, capture
# pipeline, snapshot infrastructure) introduced; the agent never had a
# fair shot.
_TOOL_BUG_PATTERNS = [
    r"\bmissing (?:\d+ )?(?:required )?positional arg",
    r"\bunexpected keyword argument",
    r"\btakes \d+ positional arguments? but \d+ (?:were|was) given",
    r"\bModuleList .*no forward\b",
    r"\bgroups=\d+ weight shape mismatch",
    r"\bpermute\(sparse_coo\) dim mismatch",
    r"\bsynthetic-input generator",
    r"\bharness:",
    r"\bcould not (?:resolve|build) (?:any of )?_CANDIDATE_SUBMODULE_PATHS",
    # Snapshot-restore failure = test harness / shared infra changed under
    # us. Not the agent's fault and not a TTNN gap.
    r"\bregression survived snapshot restore",
    r"\bshared infra or test harness changed",
]

# HF_ERROR signals: the HF reference itself errored. Common forms:
# `transformers.AutoModel.from_pretrained` failures, missing keys in HF
# state dicts, OOM constructing the HF model.
_HF_ERROR_PATTERNS = [
    r"HuggingFace.*(?:not found|404)",
    r"transformers\.\w+\.from_pretrained.*(?:failed|error)",
    r"Unrecognized configuration class",
    r"Trust remote code .* required",
    r"\btrust_remote_code\b.*\bRequired\b",
    r"HF reference forward error",
]

# AGENT_STUCK signals: agent is producing zero progress (literal/empty
# / byte-identical / no_op markers from auto_iterate.py).
_AGENT_STUCK_MARKERS = (
    "NO_OP escalation",
    "consecutive byte-identical responses",
    "empty agent records",
    "agent produced zero usable response",
    "agent produced no code",
    "too complex/long for the LLM",
)

# CONSTRAINT_MISMATCH signals (after kernel-missing pattern already
# matched + op verified present). These are signs the issue is
# dtype/layout/tile config rather than a missing op.
_CONSTRAINT_HINT_PATTERNS = [
    r"\bdtype\b.*\b(?:not supported|mismatch|incompatible)\b",
    r"\blayout\b.*\b(?:not supported|mismatch|incompatible)\b",
    r"\bTILE_LAYOUT\b.*\brequired\b",
    r"\bROW_MAJOR\b.*\brequired\b",
    r"\bshape\b.*\b(?:not supported|incompatible)\b",
    r"\btile.*alignment\b",
]

# ITERATION_BUDGET signals come from the loop itself; the reason string
# is constructed by auto_iterate.py and is well-known.
_ITERATION_BUDGET_MARKERS = (
    "exhausted per-component attempt cap",
    "hit per-component attempt cap",
    "consec-same-class",
    "moving on to next ungraduated component",
    "wall-clock budget",
    # Final-sweep PCC revelation = the heuristic thought we'd graduated
    # but a deeper check disagreed. Retry next run to verify.
    "final-sweep revealed PCC failure",
    "heuristic false positive",
)


def _matches_any(text: str, patterns) -> bool:
    if not text:
        return False
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def _contains_any(text: str, markers) -> bool:
    if not text:
        return False
    return any(m in text for m in markers)


def classify_failure(
    *,
    reason: str,
    failure_text: str = "",
    hot_cold_kind: Optional[str] = None,
) -> FailureVerdict:
    """Classify a stuck component's failure into one of `ALL_CLASSES`.

    Args:
      reason:        the short reason string the loop passed to
                     `_skip_component_to_fallback` (e.g. "exhausted
                     per-component attempt cap", "NO_OP escalation:
                     3 consecutive byte-identical responses").
      failure_text:  the recent failure trace (last_failures +
                     last_failure_details) for pattern scanning.
      hot_cold_kind: "HOT" | "COLD" | None — workload-probe verdict
                     for this component. COLD short-circuits to
                     COLD_INTENDED regardless of failure trace.

    The decision order encodes the priority laid out in the design
    doc — workload signal first (COLD components on CPU is the right
    placement, no matter how they failed), then high-confidence harness
    bug detection, then HF/kernel/constraint analysis, then loop-state
    markers, then a default fallback.
    """
    reason = reason or ""
    failure_text = failure_text or ""
    combined = reason + "\n" + failure_text

    # 1. Workload probe: a COLD component genuinely belongs on CPU.
    if (hot_cold_kind or "").upper() == "COLD":
        return FailureVerdict(
            class_name=COLD_INTENDED,
            confidence="high",
            reason=f"workload probe marked COLD; CPU is correct placement",
        )

    # 2. Tool/harness bug: scaffolder produced bad inputs. Very high
    # confidence because the patterns are concrete signatures the
    # capture/PCC layer emits.
    if _matches_any(combined, _TOOL_BUG_PATTERNS):
        return FailureVerdict(
            class_name=TOOL_BUG,
            confidence="high",
            reason="harness signature/shape mismatch (scaffolder produced bad inputs)",
        )

    # 3. HF reference error: HF model itself failed to load or forward.
    # Not a TTNN issue. Surface to user.
    if _matches_any(combined, _HF_ERROR_PATTERNS):
        return FailureVerdict(
            class_name=HF_ERROR,
            confidence="high",
            reason="HF reference forward error (not a TTNN issue)",
        )

    # 4. Kernel-missing pattern + op verification: distinguishes
    # KERNEL_VERIFIED_MISSING from CONSTRAINT_MISMATCH.
    missing_op = detect_kernel_missing(combined)
    if missing_op:
        op_exists = verify_ttnn_op_exists(missing_op)
        if op_exists is False:
            # Op verified absent — but defer to caller for the final
            # "should we try decomposition first" gate. Caller, after
            # exhausting decomposition routes, accepts this verdict.
            return FailureVerdict(
                class_name=KERNEL_VERIFIED_MISSING,
                confidence="high",
                reason=f"TTNN op `{missing_op}` verified missing",
                missing_op=missing_op,
            )
        if op_exists is True:
            # Op present — likely a constraint mismatch (dtype/layout/
            # shape). Confidence high if explicit hint patterns also
            # matched, medium otherwise.
            hint_match = _matches_any(combined, _CONSTRAINT_HINT_PATTERNS)
            return FailureVerdict(
                class_name=CONSTRAINT_MISMATCH,
                confidence=("high" if hint_match else "medium"),
                reason=(
                    f"TTNN op `{missing_op}` exists; failure is constraint "
                    f"mismatch (dtype/layout/shape), not a missing kernel"
                ),
                missing_op=missing_op,
            )
        # op_exists is None — couldn't verify. Don't claim
        # KERNEL_MISSING; keep evaluating other signals below.

    # 5. Agent-stuck signal: the loop sees no progress. Caller should
    # attempt decomposition before persisting this verdict.
    if _contains_any(combined, _AGENT_STUCK_MARKERS):
        return FailureVerdict(
            class_name=AGENT_STUCK,
            confidence="medium",
            reason="agent persistently produces NO_OP / empty responses",
        )

    # 6. Iteration budget exhausted with no other decisive signal.
    if _contains_any(combined, _ITERATION_BUDGET_MARKERS):
        return FailureVerdict(
            class_name=ITERATION_BUDGET,
            confidence="medium",
            reason="hit per-component attempt cap with no decisive failure-class signal",
        )

    # 7. Default: workload-unverified, no clear signal. Use COLD_INTENDED
    # with LOW confidence so future runs / human readers know this is
    # the safe-default fallback, not a confident verdict.
    return FailureVerdict(
        class_name=COLD_INTENDED,
        confidence="low",
        reason="no decisive failure-class signal; defaulting to CPU placement",
    )


def skip_category_for_verdict(verdict: FailureVerdict) -> str:
    """Map a FailureVerdict to the skip-list `category` field.

    Most classes round-trip 1:1, but KERNEL_VERIFIED_MISSING collapses
    to "KERNEL_MISSING" and ITERATE_MORE collapses to "COLD" as a safe
    fallback (the loop shouldn't be skipping a still-iterating component
    but if it does, CPU is the conservative choice)."""
    return SKIP_CATEGORY_FOR_CLASS.get(verdict.class_name, "COLD")
