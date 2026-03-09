#!/usr/bin/env python3
"""Score Function — Rate the success of an op generation pipeline run.

Produces a score from 0 (worst) to 100 (best) based on weighted criteria:
  1. Test success      — Did all TDD stages pass?
  2. Execution time    — How long did overall and per-stage take?
  3. Retry efficiency  — How many retries were needed?
  4. Helper usage      — Does compute kernel use helpers or raw calls?
  5. Red flags         — Issues from self-reflection not covered above?

Each criterion is scored 0–100, then combined with configurable multipliers
that sum to 1.0.

Usage:
    python3 .claude/scripts/tdd-pipeline/score.py <op_path> [--json] [--verbose]
    python3 .claude/scripts/tdd-pipeline/score.py <op_path> --weights '{"test_success": 0.35}'

Called by: create-op skill (Phase 5/6), or standalone for evaluation.
Reads: .tdd_state.json, self_reflection.md, op_design.md, kernels/*.cpp, git log
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

# ---------------------------------------------------------------------------
# Default Weights — must sum to 1.0
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "test_success": 0.35,  # Highest priority: does it work?
    "execution_time": 0.25,  # Was the pipeline efficient? (2nd priority)
    "retry_efficiency": 0.15,  # How clean was the implementation?
    "helper_usage": 0.13,  # Maintainability: helpers vs raw calls
    "red_flags": 0.12,  # Catch-all from self-reflection
}

# ---------------------------------------------------------------------------
# Time Benchmarks (seconds) — used for execution_time scoring
# ---------------------------------------------------------------------------

# Per-stage time budget: if a stage takes <= this, full marks. Beyond this,
# score degrades linearly to 0 at 4x the budget.
STAGE_TIME_BUDGET_SECONDS = 600  # 10 minutes per TDD stage
OVERALL_TIME_BUDGET_SECONDS = 3600  # 60 minutes total pipeline

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class CriterionResult:
    """Result for a single scoring criterion."""

    name: str
    raw_score: float  # 0-100 before weight
    weight: float  # multiplier (0-1)
    weighted_score: float  # raw_score * weight
    details: str  # human-readable explanation
    sub_scores: dict = field(default_factory=dict)  # optional breakdown


@dataclass
class ScoreReport:
    """Complete scoring report for a pipeline run."""

    op_name: str
    op_path: str
    total_score: float  # 0-100
    grade: str  # A/B/C/D/F
    criteria: list  # list of CriterionResult (as dicts)
    summary: str  # one-line summary


# ---------------------------------------------------------------------------
# Criterion 1: Test Success (0-100)
# ---------------------------------------------------------------------------


def score_test_success(state: dict) -> CriterionResult:
    """Score based on TDD stage pass/fail outcomes.

    Scoring:
    - All stages passed: 100
    - Per failed stage: deduct proportionally
    - A stage that failed permanently: 0 contribution from that stage
    - A stage that passed with retries: still full marks (retries scored separately)
    """
    stages = state.get("stages", [])
    if not stages:
        return CriterionResult(
            name="test_success",
            raw_score=0.0,
            weight=0.0,
            weighted_score=0.0,
            details="No TDD stages found",
        )

    total = len(stages)
    passed = sum(1 for s in stages if s["status"] == "passed")
    failed = sum(1 for s in stages if s["status"] == "failed_permanent")
    in_progress = sum(1 for s in stages if s["status"] in ("in_progress", "pending"))

    # Passed stages get full credit, in_progress/pending get zero, failed get zero
    raw_score = (passed / total) * 100.0 if total > 0 else 0.0

    sub_scores = {
        "total_stages": total,
        "passed": passed,
        "failed": failed,
        "incomplete": in_progress,
    }

    if passed == total:
        details = f"All {total} stages passed"
    elif failed > 0:
        details = f"{passed}/{total} passed, {failed} failed permanently"
    else:
        details = f"{passed}/{total} passed, {in_progress} incomplete"

    return CriterionResult(
        name="test_success",
        raw_score=raw_score,
        weight=0.0,  # filled later
        weighted_score=0.0,
        details=details,
        sub_scores=sub_scores,
    )


# ---------------------------------------------------------------------------
# Criterion 2: Execution Time (0-100)
# ---------------------------------------------------------------------------


def score_execution_time(state: dict, op_path: Path) -> CriterionResult:
    """Score based on how long the pipeline took.

    Uses git commit timestamps to estimate durations. Scores both per-stage
    and overall time.

    Scoring per stage:
    - <= budget: 100
    - Linear decay from budget to 4x budget: 100 → 0
    - > 4x budget: 0

    Overall score = 0.5 * overall_time_score + 0.5 * avg(stage_scores)
    """
    op_name = state.get("op_name", "")
    stages = state.get("stages", [])

    # Get git timestamps for commits touching this operation
    durations = _get_stage_durations_from_git(op_path, op_name, stages)
    overall_duration = durations.get("overall", None)
    stage_durations = durations.get("per_stage", {})

    # Score overall time
    if overall_duration is not None:
        overall_score = _time_score(overall_duration, OVERALL_TIME_BUDGET_SECONDS)
    else:
        overall_score = 50.0  # unknown, give neutral score

    # Score per-stage times
    stage_scores = []
    stage_details = {}
    for stage in stages:
        name = stage["name"]
        dur = stage_durations.get(name)
        if dur is not None:
            s = _time_score(dur, STAGE_TIME_BUDGET_SECONDS)
            stage_scores.append(s)
            stage_details[name] = {"duration_s": round(dur), "score": round(s, 1)}
        else:
            stage_details[name] = {"duration_s": None, "score": "unknown"}

    avg_stage_score = sum(stage_scores) / len(stage_scores) if stage_scores else 50.0

    raw_score = 0.5 * overall_score + 0.5 * avg_stage_score

    details_parts = []
    if overall_duration is not None:
        details_parts.append(f"Overall: {_format_duration(overall_duration)} (score: {overall_score:.0f})")
    else:
        details_parts.append("Overall duration: unknown (git timestamps unavailable)")
    details_parts.append(f"Avg stage score: {avg_stage_score:.0f}")

    return CriterionResult(
        name="execution_time",
        raw_score=round(raw_score, 1),
        weight=0.0,
        weighted_score=0.0,
        details="; ".join(details_parts),
        sub_scores={
            "overall_duration_s": round(overall_duration) if overall_duration else None,
            "overall_score": round(overall_score, 1),
            "avg_stage_score": round(avg_stage_score, 1),
            "stages": stage_details,
        },
    )


def _time_score(actual_seconds: float, budget_seconds: float) -> float:
    """Linear decay: 100 at budget, 0 at 4x budget."""
    if actual_seconds <= budget_seconds:
        return 100.0
    elif actual_seconds >= budget_seconds * 4:
        return 0.0
    else:
        # Linear interpolation between budget and 4*budget
        fraction_over = (actual_seconds - budget_seconds) / (budget_seconds * 3)
        return max(0.0, 100.0 * (1.0 - fraction_over))


def _format_duration(seconds: float) -> str:
    """Format seconds as 'Xm Ys'."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _get_stage_durations_from_git(op_path: Path, op_name: str, stages: list) -> dict:
    """Extract timing from git commits.

    Looks for commits with stage-pass markers like '[kw-tdd] stage X passed'
    and uses their timestamps to compute durations.
    """
    result = {"overall": None, "per_stage": {}}

    try:
        # Get commits touching the op files, with timestamps
        git_log = subprocess.check_output(
            [
                "git",
                "log",
                "--format=%H %aI %s",
                "--all",
                "--",
                str(op_path),
                f"tests/ttnn/unit_tests/operations/{op_name}/",
            ],
            text=True,
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return result

    if not git_log:
        return result

    commits = []
    for line in git_log.splitlines():
        parts = line.split(" ", 2)
        if len(parts) >= 3:
            commits.append(
                {
                    "hash": parts[0],
                    "timestamp": parts[1],
                    "message": parts[2],
                }
            )

    if not commits:
        return result

    # Commits are newest-first; reverse for chronological order
    commits.reverse()

    # Parse ISO timestamps
    from datetime import datetime, timezone

    def parse_ts(iso_str: str) -> Optional[float]:
        try:
            # Handle timezone offset format like 2024-01-01T12:00:00+01:00
            dt = datetime.fromisoformat(iso_str)
            return dt.timestamp()
        except (ValueError, TypeError):
            return None

    # Overall duration: first commit to last commit
    first_ts = parse_ts(commits[0]["timestamp"])
    last_ts = parse_ts(commits[-1]["timestamp"])
    if first_ts and last_ts:
        result["overall"] = last_ts - first_ts

    # Per-stage duration: time between consecutive stage-pass commits
    stage_pass_times = {}
    for commit in commits:
        msg = commit["message"]
        # Match patterns like: [kw-tdd] stage passthrough passed
        match = re.search(r"stage\s+(\w+)\s+passed", msg)
        if match:
            stage_name = match.group(1)
            ts = parse_ts(commit["timestamp"])
            if ts:
                stage_pass_times[stage_name] = ts

    # Compute per-stage durations as intervals between consecutive passes
    sorted_stages = [s["name"] for s in stages]
    prev_ts = first_ts
    for stage_name in sorted_stages:
        pass_ts = stage_pass_times.get(stage_name)
        if pass_ts and prev_ts:
            result["per_stage"][stage_name] = pass_ts - prev_ts
            prev_ts = pass_ts

    return result


# ---------------------------------------------------------------------------
# Criterion 3: Retry Efficiency (0-100)
# ---------------------------------------------------------------------------


def score_retry_efficiency(state: dict) -> CriterionResult:
    """Score based on how few retries were needed.

    Scoring per stage:
    - 0 hard attempts, 0 free retries: 100 (first-try pass)
    - Each hard attempt: -15 points
    - Each free retry: -5 points (less severe, compilation/shape fixes)
    - Floor at 0

    Overall = average across all passed/attempted stages.
    """
    stages = state.get("stages", [])
    if not stages:
        return CriterionResult(
            name="retry_efficiency",
            raw_score=0.0,
            weight=0.0,
            weighted_score=0.0,
            details="No stages to evaluate",
        )

    stage_scores = {}
    total_hard = 0
    total_free = 0

    for stage in stages:
        name = stage["name"]
        hard = stage.get("attempts", 0)
        free = stage.get("free_retries", 0)
        total_hard += hard
        total_free += free

        # Score this stage
        score = max(0.0, 100.0 - (hard * 15.0) - (free * 5.0))

        # Failed stages get 0 regardless
        if stage["status"] == "failed_permanent":
            score = 0.0

        stage_scores[name] = {
            "hard_attempts": hard,
            "free_retries": free,
            "score": round(score, 1),
        }

    scores = [v["score"] for v in stage_scores.values()]
    raw_score = sum(scores) / len(scores) if scores else 0.0

    details = f"Total retries: {total_hard} hard, {total_free} free across {len(stages)} stages"

    return CriterionResult(
        name="retry_efficiency",
        raw_score=round(raw_score, 1),
        weight=0.0,
        weighted_score=0.0,
        details=details,
        sub_scores={
            "total_hard_attempts": total_hard,
            "total_free_retries": total_free,
            "stages": stage_scores,
        },
    )


# ---------------------------------------------------------------------------
# Criterion 4: Helper Usage (0-100)
# ---------------------------------------------------------------------------


def score_helper_usage(op_path: Path, state: dict) -> CriterionResult:
    """Score based on how much the compute kernel uses helpers vs raw calls.

    The pipeline strongly encourages using helpers from ttnn/cpp/ttnn/kernel_lib/.
    Raw tile manipulation calls (cb_wait_front, cb_pop_front, tile_regs_acquire, etc.)
    in the compute kernel are a code smell when helpers exist.

    Scoring:
    - 100: All compute phases use helpers (no raw tile/CB ops in compute kernel)
    - 70: Partial helper usage (some phases use helpers, some use raw calls)
    - 40: No helpers but functional raw implementation
    - 0: No compute kernel found or empty stubs

    Also checks the design doc for helper mapping to see if the agent followed
    the architect's recommendations.
    """
    op_name = state.get("op_name", "")
    compute_kernel = op_path / "kernels" / f"{op_name}_compute.cpp"

    if not compute_kernel.exists():
        return CriterionResult(
            name="helper_usage",
            raw_score=0.0,
            weight=0.0,
            weighted_score=0.0,
            details="Compute kernel not found",
        )

    with open(compute_kernel) as f:
        code = f.read()

    # Check if it's just a stub
    if "// STUB" in code or len(code.strip()) < 100:
        return CriterionResult(
            name="helper_usage",
            raw_score=0.0,
            weight=0.0,
            weighted_score=0.0,
            details="Compute kernel is a stub",
        )

    # Detect helper includes
    helper_includes = re.findall(r'#include\s+"ttnn/cpp/ttnn/kernel_lib/(\w+)\.hpp"', code)

    # Detect helper function calls (namespace compute_kernel_lib::)
    helper_calls = re.findall(r"compute_kernel_lib::(\w+)", code)

    # Detect raw tile manipulation calls (signs of NOT using helpers)
    raw_tile_ops = []
    raw_patterns = [
        (r"\bcb_wait_front\b", "cb_wait_front"),
        (r"\bcb_pop_front\b", "cb_pop_front"),
        (r"\bcb_reserve_back\b", "cb_reserve_back"),
        (r"\bcb_push_back\b", "cb_push_back"),
        (r"\btile_regs_acquire\b", "tile_regs_acquire"),
        (r"\btile_regs_commit\b", "tile_regs_commit"),
        (r"\btile_regs_wait\b", "tile_regs_wait"),
        (r"\btile_regs_release\b", "tile_regs_release"),
        (r"\bcopy_tile\b", "copy_tile"),
        (r"\bmatmul_tiles\b", "matmul_tiles"),
        (r"\breduce_tile\b", "reduce_tile"),
    ]
    for pattern, name in raw_patterns:
        if re.search(pattern, code):
            raw_tile_ops.append(name)

    # Check design doc for helper mapping recommendations
    design_helpers = _extract_design_helpers(op_path)

    # Compute score
    has_helpers = len(helper_includes) > 0 or len(helper_calls) > 0
    has_raw = len(raw_tile_ops) > 0

    if has_helpers and not has_raw:
        # Pure helper usage — best case
        raw_score = 100.0
        usage_level = "full"
    elif has_helpers and has_raw:
        # Mixed: helpers for some phases, raw for others
        # Score based on ratio of helper calls to total
        total_ops = len(helper_calls) + len(raw_tile_ops)
        helper_ratio = len(helper_calls) / total_ops if total_ops > 0 else 0
        raw_score = 40.0 + (helper_ratio * 60.0)  # 40-100 range
        usage_level = "partial"
    elif not has_helpers and has_raw:
        # All raw — functional but not using library
        raw_score = 40.0
        usage_level = "none"
    else:
        # No ops at all — likely empty or stub-like
        raw_score = 0.0
        usage_level = "empty"

    # Bonus/penalty: did they follow the design doc's helper recommendations?
    design_match = _check_design_compliance(design_helpers, helper_includes, helper_calls)
    if design_match is not None:
        if design_match:
            raw_score = min(100.0, raw_score + 5.0)  # small bonus
        else:
            raw_score = max(0.0, raw_score - 10.0)  # penalty for ignoring design

    raw_score = round(min(100.0, max(0.0, raw_score)), 1)

    details = f"Helper usage: {usage_level}"
    if helper_includes:
        details += f"; includes: {', '.join(helper_includes)}"
    if raw_tile_ops:
        details += f"; raw ops: {', '.join(raw_tile_ops[:5])}"

    return CriterionResult(
        name="helper_usage",
        raw_score=raw_score,
        weight=0.0,
        weighted_score=0.0,
        details=details,
        sub_scores={
            "usage_level": usage_level,
            "helper_includes": helper_includes,
            "helper_calls": list(set(helper_calls)),
            "raw_tile_ops": raw_tile_ops,
            "design_helpers_recommended": design_helpers,
            "followed_design": design_match,
        },
    )


def _extract_design_helpers(op_path: Path) -> list:
    """Extract helper recommendations from op_design.md Part 2."""
    design_path = op_path / "op_design.md"
    if not design_path.exists():
        return []

    with open(design_path) as f:
        content = f.read()

    # Look for helper mapping table or mentions
    helpers = []
    # Match patterns like: "use tilize()" or "helper: reduce()" or "USE HELPER: tilize_helpers"
    for match in re.finditer(r"(?:use|helper|USE HELPER)[:\s]+(\w+(?:_helpers)?)\b", content, re.IGNORECASE):
        helpers.append(match.group(1))

    # Also look for kernel_lib references
    for match in re.finditer(r"kernel_lib/(\w+)\.hpp", content):
        helpers.append(match.group(1))

    return list(set(helpers))


def _check_design_compliance(design_helpers: list, includes: list, calls: list) -> Optional[bool]:
    """Check if kernel follows design doc helper recommendations.

    Returns True if compliant, False if not, None if no recommendations exist.
    """
    if not design_helpers:
        return None

    # Normalize names for comparison
    used = set()
    for inc in includes:
        used.add(inc.lower().replace("_helpers", "").replace("_compute", ""))
    for call in calls:
        used.add(call.lower())

    recommended = set()
    for h in design_helpers:
        recommended.add(h.lower().replace("_helpers", "").replace("_compute", ""))

    if not recommended:
        return None

    # Check if at least half the recommended helpers are used
    overlap = recommended & used
    return len(overlap) >= len(recommended) * 0.5


# ---------------------------------------------------------------------------
# Criterion 5: Red Flags (0-100)
# ---------------------------------------------------------------------------


def score_red_flags(op_path: Path, state: dict) -> CriterionResult:
    """Score based on issues found in self-reflection that aren't already
    captured by other criteria.

    Reads self_reflection.md and counts HIGH/MEDIUM/LOW severity issues,
    excluding those that are already accounted for by test_success,
    execution_time, retry_efficiency, or helper_usage.

    Scoring:
    - Start at 100
    - HIGH severity issue: -20 each
    - MEDIUM severity issue: -10 each
    - LOW severity issue: -5 each
    - Floor at 0

    If no self_reflection.md exists, reports an error with score 0.
    """
    reflection_path = op_path / "self_reflection.md"
    if not reflection_path.exists():
        return CriterionResult(
            name="red_flags",
            raw_score=0.0,
            weight=0.0,
            weighted_score=0.0,
            details="ERROR: self_reflection.md not found — Phase 6 did not run",
        )

    with open(reflection_path) as f:
        content = f.read()

    # Count severity levels in the Issues Found section
    # Look for "| Severity | HIGH |" or "| Severity | MEDIUM |" patterns
    high_count = len(re.findall(r"\|\s*Severity\s*\|\s*HIGH\s*\|", content, re.IGNORECASE))
    medium_count = len(re.findall(r"\|\s*Severity\s*\|\s*MEDIUM\s*\|", content, re.IGNORECASE))
    low_count = len(re.findall(r"\|\s*Severity\s*\|\s*LOW\s*\|", content, re.IGNORECASE))

    # Filter out issues that are already captured by other criteria
    # We look for categories that overlap with our other scoring criteria
    already_scored_patterns = [
        r"numerical_mismatch",
        r"compilation_error",
        r"helper.*usage",
        r"execution.*time|duration|slow",
        r"retry|attempt",
    ]

    # Try to identify issues section and filter
    issues_section = _extract_section(content, "Issues Found", "Efficiency Analysis")
    filtered_high = high_count
    filtered_medium = medium_count
    filtered_low = low_count

    if issues_section:
        # Split into individual issues
        issue_blocks = re.split(r"### Issue \d+", issues_section)
        for block in issue_blocks:
            is_already_scored = any(re.search(p, block, re.IGNORECASE) for p in already_scored_patterns)
            if is_already_scored:
                if re.search(r"HIGH", block, re.IGNORECASE):
                    filtered_high = max(0, filtered_high - 1)
                elif re.search(r"MEDIUM", block, re.IGNORECASE):
                    filtered_medium = max(0, filtered_medium - 1)
                elif re.search(r"LOW", block, re.IGNORECASE):
                    filtered_low = max(0, filtered_low - 1)

    deduction = (filtered_high * 20.0) + (filtered_medium * 10.0) + (filtered_low * 5.0)
    raw_score = max(0.0, 100.0 - deduction)

    # Also check inter-agent communication quality
    comm_section = _extract_section(content, "Inter-Agent Communication", "Upstream Feedback")
    poor_handoffs = len(re.findall(r"\|\s*Quality\s*\|\s*POOR\s*\|", content, re.IGNORECASE))
    if poor_handoffs > 0:
        raw_score = max(0.0, raw_score - (poor_handoffs * 10.0))

    # Check overall assessment if available
    overall_result = re.search(r"\|\s*Overall Result\s*\|\s*(\w+)", content)
    if overall_result:
        result_str = overall_result.group(1).upper()
        if result_str == "FAILED":
            raw_score = min(raw_score, 20.0)
        elif result_str == "PARTIAL":
            raw_score = min(raw_score, 60.0)

    raw_score = round(raw_score, 1)

    details_parts = []
    if filtered_high > 0:
        details_parts.append(f"{filtered_high} HIGH severity")
    if filtered_medium > 0:
        details_parts.append(f"{filtered_medium} MEDIUM severity")
    if filtered_low > 0:
        details_parts.append(f"{filtered_low} LOW severity")
    if poor_handoffs > 0:
        details_parts.append(f"{poor_handoffs} POOR handoffs")
    if not details_parts:
        details_parts.append("No red flags found")

    return CriterionResult(
        name="red_flags",
        raw_score=raw_score,
        weight=0.0,
        weighted_score=0.0,
        details="Red flags: " + ", ".join(details_parts),
        sub_scores={
            "high_issues": filtered_high,
            "medium_issues": filtered_medium,
            "low_issues": filtered_low,
            "poor_handoffs": poor_handoffs,
            "total_deduction": round(100.0 - raw_score, 1),
        },
    )


def _extract_section(content: str, start_heading: str, end_heading: str) -> str:
    """Extract text between two markdown headings."""
    pattern = rf"## \d*\.?\s*{re.escape(start_heading)}(.*?)(?=## \d*\.?\s*{re.escape(end_heading)}|\Z)"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else ""


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def compute_grade(score: float) -> str:
    """Convert 0-100 score to letter grade."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"


# ---------------------------------------------------------------------------
# Main Scoring Function
# ---------------------------------------------------------------------------


def score_pipeline_run(op_path: Path, weights: Optional[dict] = None) -> ScoreReport:
    """Compute the composite score for a pipeline run.

    Args:
        op_path: Path to the operation directory.
        weights: Optional override for criterion weights. Missing keys
                 use defaults. Must sum to 1.0.

    Returns:
        ScoreReport with total score, grade, and per-criterion breakdown.
    """
    # Resolve weights
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    # Validate weights sum to ~1.0
    total_weight = sum(w.values())
    if abs(total_weight - 1.0) > 0.01:
        print(f"WARNING: Weights sum to {total_weight}, not 1.0. Normalizing.", file=sys.stderr)
        for k in w:
            w[k] /= total_weight

    # Load state
    state_path = op_path / ".tdd_state.json"
    if not state_path.exists():
        return ScoreReport(
            op_name=op_path.name,
            op_path=str(op_path),
            total_score=0.0,
            grade="F",
            criteria=[],
            summary="No .tdd_state.json found — pipeline did not run",
        )

    with open(state_path) as f:
        state = json.load(f)

    # Score each criterion
    criteria_results = [
        score_test_success(state),
        score_execution_time(state, op_path),
        score_retry_efficiency(state),
        score_helper_usage(op_path, state),
        score_red_flags(op_path, state),
    ]

    # Apply weights
    total_score = 0.0
    for cr in criteria_results:
        cr.weight = w.get(cr.name, 0.0)
        cr.weighted_score = round(cr.raw_score * cr.weight, 2)
        total_score += cr.weighted_score

    total_score = round(min(100.0, max(0.0, total_score)), 1)
    grade = compute_grade(total_score)

    # Build summary
    op_name = state.get("op_name", op_path.name)
    stages = state.get("stages", [])
    passed = sum(1 for s in stages if s["status"] == "passed")
    summary = f"{op_name}: {total_score}/100 ({grade}) — {passed}/{len(stages)} stages passed"

    return ScoreReport(
        op_name=op_name,
        op_path=str(op_path),
        total_score=total_score,
        grade=grade,
        criteria=[asdict(cr) for cr in criteria_results],
        summary=summary,
    )


# ---------------------------------------------------------------------------
# CLI Output Formatting
# ---------------------------------------------------------------------------


def print_report(report: ScoreReport, verbose: bool = False) -> None:
    """Print a human-readable score report."""
    print(f"\n{'='*60}")
    print(f"  PIPELINE SCORE: {report.op_name}")
    print(f"{'='*60}")
    print(f"\n  Total Score: {report.total_score}/100  Grade: {report.grade}\n")
    print(f"  {report.summary}\n")

    print(f"  {'Criterion':<20} {'Raw':>6} {'Weight':>7} {'Weighted':>9}")
    print(f"  {'-'*20} {'-'*6} {'-'*7} {'-'*9}")

    for cr in report.criteria:
        name = cr["name"].replace("_", " ").title()
        raw = f"{cr['raw_score']:.1f}"
        weight = f"x{cr['weight']:.2f}"
        weighted = f"{cr['weighted_score']:.1f}"
        print(f"  {name:<20} {raw:>6} {weight:>7} {weighted:>9}")

    print(f"  {'-'*20} {'-'*6} {'-'*7} {'-'*9}")
    print(f"  {'TOTAL':<20} {'':>6} {'':>7} {report.total_score:>8.1f}\n")

    if verbose:
        print(f"  Details per criterion:")
        print(f"  {'-'*55}")
        for cr in report.criteria:
            name = cr["name"].replace("_", " ").title()
            print(f"\n  [{name}]")
            print(f"    {cr['details']}")
            if cr.get("sub_scores"):
                for k, v in cr["sub_scores"].items():
                    print(f"    {k}: {v}")

    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Score a pipeline run for op generation quality",
        prog="score.py",
    )
    parser.add_argument("op_path", help="Path to the operation directory")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed breakdown")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="JSON string of weight overrides, e.g. '{\"test_success\": 0.40}'",
    )

    args = parser.parse_args()
    op_path = Path(args.op_path)

    if not op_path.exists():
        print(f"ERROR: Operation path not found: {op_path}", file=sys.stderr)
        sys.exit(1)

    weights = None
    if args.weights:
        try:
            weights = json.loads(args.weights)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid weights JSON: {e}", file=sys.stderr)
            sys.exit(1)

    report = score_pipeline_run(op_path, weights)

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print_report(report, verbose=args.verbose)

    # Exit with non-zero if score is failing (< 50)
    sys.exit(0 if report.total_score >= 50 else 1)


if __name__ == "__main__":
    main()
