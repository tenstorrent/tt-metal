#!/usr/bin/env python3
"""Score Function — Rate the success of an op generation pipeline run.

Produces a score from 0 (worst) to 100 (best) in two stages:

Stage 1 — Base score (additive, weights sum to 1.0):
  1. Golden score      — Did the op pass real golden tests? (highest weight)
  2. Test success      — Did TDD stages pass? (lower weight, tends to always pass)
  3. Execution time    — How long did overall and per-stage take?
  4. Retry efficiency  — How many retries were needed?
  5. Red flags         — Issues from self-reflection not covered above?

Stage 2 — Helper usage multiplier:
  6. Helper usage      — Multiplies the base score.
                         100% compliance = 1.0x, 0% compliance = 0.2x.
                         Not using helpers when they exist tanks the entire grade.

Usage:
    python3 .claude/scripts/tdd-pipeline/score.py <op_path> [--json] [--verbose]
    python3 .claude/scripts/tdd-pipeline/score.py <op_path> --golden-results path/to/golden_results.txt
    python3 .claude/scripts/tdd-pipeline/score.py <op_path> --weights '{"golden_score": 0.40}'

Called by: run_eval.sh (with --golden-results), create-op skill (Phase 5/6), or standalone.
Reads: .tdd_state.json, self_reflection.md, op_design.md, kernels/*.cpp, git log, golden_results.txt
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
KERNEL_LIB_DIR = REPO_ROOT / "ttnn" / "cpp" / "ttnn" / "kernel_lib"

# ---------------------------------------------------------------------------
# Default Weights — must sum to 1.0
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "golden_score": 0.35,  # Highest priority: does it pass real golden tests?
    "test_success": 0.15,  # TDD stages (tend to always pass, lower weight)
    "execution_time": 0.20,  # Was the pipeline efficient?
    "retry_efficiency": 0.15,  # How clean was the implementation?
    "red_flags": 0.15,  # Catch-all from self-reflection
}
# helper_usage is NOT in the additive weights — it acts as a multiplier on
# the final score. 100% compliance = 1.0x, 0% compliance = 0.2x.

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
# Criterion 1: Golden Score (0-100)
# ---------------------------------------------------------------------------


def score_golden_tests(golden_results_path: Optional[Path]) -> CriterionResult:
    """Score based on golden test pass rate.

    Golden tests are the ground truth — they test the generated operation
    against a PyTorch reference across ~70 shapes, multiple modes, and
    edge cases. This is the most important quality signal.

    Scoring: (passed / total) * 100. If no golden results exist, score is 0.

    Sources (in priority order):
    1. golden_results.txt passed via --golden-results CLI arg
    2. Falls back to 0 if not available
    """
    if golden_results_path is None or not golden_results_path.exists():
        return CriterionResult(
            name="golden_score",
            raw_score=0.0,
            weight=0.0,
            weighted_score=0.0,
            details="No golden test results found",
        )

    try:
        content = golden_results_path.read_text()
    except OSError:
        return CriterionResult(
            name="golden_score",
            raw_score=0.0,
            weight=0.0,
            weighted_score=0.0,
            details="Could not read golden results file",
        )

    # Parse: "PASSED=N FAILED=N ERRORS=N SKIPPED=N HANGS=N TOTAL=N"
    def _extract(key: str) -> int:
        m = re.search(rf"{key}=(\d+)", content)
        return int(m.group(1)) if m else 0

    passed = _extract("PASSED")
    failed = _extract("FAILED")
    errors = _extract("ERRORS")
    skipped = _extract("SKIPPED")
    hangs = _extract("HANGS")
    total = _extract("TOTAL")

    if total == 0:
        return CriterionResult(
            name="golden_score",
            raw_score=0.0,
            weight=0.0,
            weighted_score=0.0,
            details="Golden tests ran but produced 0 total tests",
        )

    raw_score = (passed / total) * 100.0

    details = f"Golden: {passed}/{total} passed"
    if failed > 0:
        details += f", {failed} failed"
    if errors > 0:
        details += f", {errors} errors"
    if hangs > 0:
        details += f", {hangs} hangs"
    if skipped > 0:
        details += f", {skipped} skipped"

    return CriterionResult(
        name="golden_score",
        raw_score=round(raw_score, 1),
        weight=0.0,
        weighted_score=0.0,
        details=details,
        sub_scores={
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "hangs": hangs,
            "total": total,
        },
    )


# ---------------------------------------------------------------------------
# Criterion 2: Test Success (0-100)
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

    Primary source: self_reflection.md tables (breadcrumb-based, most accurate).
    Fallback: git commit timestamps.

    Scoring per stage:
    - <= budget: 100
    - Linear decay from budget to 4x budget: 100 → 0
    - > 4x budget: 0

    Overall score = 0.5 * overall_time_score + 0.5 * avg(stage_scores)
    """
    op_name = state.get("op_name", "")
    stages = state.get("stages", [])

    # Try self_reflection.md first (breadcrumb-based, most accurate)
    durations = _get_durations_from_self_reflection(op_path)

    # Fall back to git timestamps if self_reflection.md unavailable
    if durations["overall"] is None and not durations["per_stage"]:
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

    # Include per-phase breakdown if available from self_reflection.md
    phase_details = {}
    per_phase = durations.get("per_phase", {})
    for phase_name, dur in per_phase.items():
        phase_details[phase_name] = {"duration_s": round(dur), "duration": _format_duration(dur)}

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
            "phases": phase_details,
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


def _parse_duration_str(duration_str: str) -> Optional[float]:
    """Parse duration strings like '~3m 35s', '1m 41s', '~13m', '2m 50s' to seconds."""
    duration_str = duration_str.strip().lstrip("~")
    total = 0.0
    m_match = re.search(r"(\d+)\s*m", duration_str)
    s_match = re.search(r"(\d+)\s*s", duration_str)
    if m_match:
        total += int(m_match.group(1)) * 60
    if s_match:
        total += int(s_match.group(1))
    return total if total > 0 else None


def _get_durations_from_self_reflection(op_path: Path) -> dict:
    """Parse timing data from self_reflection.md tables.

    Primary source: the '### Phase Timeline' table's **Total** row, which
    is computed by the self-reflection agent as earliest Phase 0 start to
    latest Phase 5 end. Falls back to summing phases 0-4 if the Total row
    is missing.

    Secondary source: the TDD Stage table for per-stage durations.

    These are more accurate than git timestamps because they come from
    breadcrumb event timestamps recorded during execution.
    """
    result = {"overall": None, "per_stage": {}}
    reflection_path = op_path / "self_reflection.md"
    if not reflection_path.exists():
        return result

    try:
        content = reflection_path.read_text()
    except OSError:
        return result

    # Extract just the Phase Timeline section to avoid false positives
    # from other tables (e.g., Agent Duration Breakdown has timestamps
    # like "10:02:43" that match the phase pattern).
    phase_timeline_section = ""
    match = re.search(r"###\s*Phase Timeline\s*\n(.*?)(?=\n###|\n---|\n##)", content, re.DOTALL)
    if match:
        phase_timeline_section = match.group(1)

    # Parse Phase Timeline table rows.
    # Format: "| 0: Discovery | orchestrator | ~10:00 | ~10:02 | ~2m | Done | ... |"
    # Capture: phase number and duration column.
    phase_durations = {}
    for match in re.finditer(
        r"\|\s*(\d+):\s*(\w[\w\s]*?)\s*\|.*?\|.*?\|.*?\|\s*(~?[\d]+m[\s\d]*s?)\s*\|",
        phase_timeline_section,
    ):
        phase_num = int(match.group(1))
        phase_name = match.group(2).strip()
        dur = _parse_duration_str(match.group(3))
        if dur:
            phase_durations[phase_num] = {"name": phase_name, "duration": dur}

    # Overall: prefer the Total row from the Phase Timeline table (self-reflection
    # agent computes this as earliest Phase 0 start → latest Phase 5 end).
    # Fall back to summing phases 0-4 if the Total row is missing.
    total_row_match = re.search(
        r"\|\s*\*?\*?Total\*?\*?\s*\|.*?\|\s*\*?\*?(~?[\d]+m[\s\d]*s?)\*?\*?\s*\|",
        phase_timeline_section,
    )
    if total_row_match:
        total_dur = _parse_duration_str(total_row_match.group(1))
        if total_dur:
            result["overall"] = total_dur
    if result["overall"] is None:
        production_durations = [v["duration"] for k, v in phase_durations.items() if k <= 4]
        if production_durations:
            result["overall"] = sum(production_durations)

    # Store per-phase breakdown for reporting
    result["per_phase"] = {v["name"]: v["duration"] for k, v in phase_durations.items()}

    # Parse TDD Stage table for per-stage durations:
    #   "| data_pipeline | 1m 41s | 0 free, 0 hard | PASS | ... |"
    for match in re.finditer(r"\|\s*(\w+)\s*\|\s*([\d]+m[\s\d]*s?)\s*\|\s*\d+ free.*?\|", content):
        stage_name = match.group(1)
        dur = _parse_duration_str(match.group(2))
        if dur:
            result["per_stage"][stage_name] = dur

    return result


def _get_stage_durations_from_git(op_path: Path, op_name: str, stages: list) -> dict:
    """Extract timing from git commits for the CURRENT pipeline run.

    Uses .tdd_state.json commit hashes as anchors to identify which pipeline
    run to measure. Only considers commits from the same run — not prior runs
    on the same branch.

    Overall duration = first commit of this run → last stage-pass commit.
    Per-stage duration = interval between consecutive stage-pass commits.
    """
    result = {"overall": None, "per_stage": {}}

    try:
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

    from datetime import datetime

    def parse_ts(iso_str: str) -> Optional[float]:
        try:
            dt = datetime.fromisoformat(iso_str)
            return dt.timestamp()
        except (ValueError, TypeError):
            return None

    # Parse all commits into chronological order
    all_commits = []
    for line in git_log.splitlines():
        parts = line.split(" ", 2)
        if len(parts) >= 3:
            all_commits.append(
                {
                    "hash": parts[0],
                    "timestamp": parts[1],
                    "message": parts[2],
                }
            )
    all_commits.reverse()  # oldest first

    if not all_commits:
        return result

    # Use .tdd_state.json stage commit hashes as ground truth anchors.
    # Only these hashes identify commits from the CURRENT pipeline run.
    anchor_hashes = set()
    for stage in stages:
        commit = stage.get("commit")
        if commit:
            anchor_hashes.add(commit[:11])

    if not anchor_hashes:
        return result

    # Find first anchor commit — this grounds us in the current run.
    first_anchor_idx = None
    first_anchor_ts = None
    for i, c in enumerate(all_commits):
        if c["hash"][:11] in anchor_hashes:
            first_anchor_idx = i
            first_anchor_ts = parse_ts(c["timestamp"])
            break

    if first_anchor_idx is None:
        return result

    # Find the last stage-pass commit that belongs to this run.
    # Walk forward from the first anchor — any stage-pass commit within
    # a tight time window is part of this run.
    last_stage_pass_idx = first_anchor_idx
    prev_ts = first_anchor_ts
    for i in range(first_anchor_idx, len(all_commits)):
        c = all_commits[i]
        ts = parse_ts(c["timestamp"])
        # Stop if we hit a commit > 2 hours after the first anchor
        if ts and first_anchor_ts and (ts - first_anchor_ts) > 7200:
            break
        if re.search(r"stage\s+\w+.*passed", c["message"]):
            last_stage_pass_idx = i

    # Walk backwards from the first anchor to find the pipeline start.
    # Include production-phase commits (analyzer, architect, builder) that
    # precede the TDD stages. Skip self-reflection and report commits
    # (post-processing from prior runs).
    AGENT_RE = re.compile(r"\[ttnn-|\[create-op\]")
    PRODUCTION_RE = re.compile(
        r"\[ttnn-operation-analyzer\]|\[ttnn-operation-architect\]" r"|\[ttnn-generic-op-builder\]|\[ttnn-kernel-writer"
    )
    run_start_idx = first_anchor_idx
    next_agent_ts = first_anchor_ts
    for i in range(first_anchor_idx - 1, -1, -1):
        c = all_commits[i]
        if not PRODUCTION_RE.search(c["message"]):
            continue
        ts = parse_ts(c["timestamp"])
        if ts and next_agent_ts and (next_agent_ts - ts) > 1800:
            break  # > 30 min gap = different run
        run_start_idx = i
        next_agent_ts = ts

    # Slice: all agent commits from pipeline start to last stage pass.
    # Excludes post-processing (REPORT, self-reflection).
    run_commits = [c for c in all_commits[run_start_idx : last_stage_pass_idx + 1] if AGENT_RE.match(c["message"])]

    if not run_commits:
        return result

    # Overall duration: first commit of this run → last commit of this run
    first_ts = parse_ts(run_commits[0]["timestamp"])
    last_ts = parse_ts(run_commits[-1]["timestamp"])
    if first_ts and last_ts:
        result["overall"] = last_ts - first_ts

    # Per-stage: find stage-pass commits within this run
    stage_pass_times = {}
    for commit in run_commits:
        msg = commit["message"]
        match = re.search(r"stage\s+(\w+).*passed", msg)
        if match:
            stage_name = match.group(1)
            ts = parse_ts(commit["timestamp"])
            if ts:
                stage_pass_times[stage_name] = ts

    # First stage's duration starts from the run start (includes analysis/design/build).
    # Subsequent stages measure from the previous stage's pass commit.
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


_helper_abstracted_ops_cache: Optional[set] = None


def _get_helper_abstracted_ops() -> set:
    """Scan kernel_lib .inl files to find which raw ops helpers abstract away.

    Any function call that appears inside a helper .inl implementation is
    something the helpers are designed to replace. Using these directly in a
    compute kernel (when helpers are available) indicates the kernel writer
    bypassed the helper library.

    Results are cached for the lifetime of the process.
    """
    global _helper_abstracted_ops_cache
    if _helper_abstracted_ops_cache is not None:
        return _helper_abstracted_ops_cache

    abstracted = set()
    if not KERNEL_LIB_DIR.exists():
        # Fallback: if kernel_lib dir is missing, return empty set.
        # This means no raw ops will be penalized — safe default.
        _helper_abstracted_ops_cache = abstracted
        return abstracted

    for inl_file in KERNEL_LIB_DIR.glob("*.inl"):
        try:
            content = inl_file.read_text()
        except OSError:
            continue
        # Extract function calls: word followed by ( or <...>(
        for match in re.finditer(r"\b([a-z_]+)\s*(?:<[^>]*>)?\s*\(", content):
            name = match.group(1)
            # Skip C++ keywords, control flow, and trivially common names
            if name in (
                "if",
                "for",
                "while",
                "return",
                "sizeof",
                "static_cast",
                "constexpr",
                "const",
                "auto",
                "void",
                "uint32_t",
                "get",
                "set",
                "min",
                "max",
                "get_compile_time_arg_val",
                "get_arg_val",
                "static_assert",
            ):
                continue
            abstracted.add(name)

    # Always exclude cb_wait_front / cb_pop_front from the abstracted set.
    # These appear inside helpers BUT are also required outside helpers for:
    #   - Persistent CBs (scaler, eps, gamma, beta) that need a one-time wait
    #   - NoWaitNoPop policy where the caller MUST manually pop
    # Penalizing these would punish correct, design-mandated usage.
    abstracted -= {"cb_wait_front", "cb_pop_front", "cb_reserve_back", "cb_push_back"}

    _helper_abstracted_ops_cache = abstracted
    return abstracted


def _get_helper_score_from_self_reflection(op_path: Path) -> Optional[CriterionResult]:
    """Parse helper compliance data from self_reflection.md Section 10.

    Looks for the Helper Compliance Summary table which contains:
      - Total kernel phases
      - Phases using helpers correctly (✅)
      - Phases with justified raw code (✅)
      - Phases with missed helpers (❌)
      - Phases with misused helpers (❌)
      - Helper compliance rate (%)

    Returns a CriterionResult if the table is found, None otherwise.
    """
    reflection_path = op_path / "self_reflection.md"
    if not reflection_path.exists():
        return None

    try:
        content = reflection_path.read_text()
    except OSError:
        return None

    # Extract the Helper Compliance Summary section
    section_match = re.search(
        r"###\s*Helper Compliance Summary\s*\n(.*?)(?=\n###|\n---|\n##)",
        content,
        re.DOTALL,
    )
    if not section_match:
        return None

    section = section_match.group(1)

    # Parse table rows: "| Metric | Value |"
    def _parse_int_from_row(label: str) -> Optional[int]:
        m = re.search(rf"\|\s*{re.escape(label)}\s*\|\s*(\d+)", section)
        return int(m.group(1)) if m else None

    total_phases = _parse_int_from_row("Total kernel phases")
    helpers_correct = _parse_int_from_row("Phases using helpers correctly")
    raw_justified = _parse_int_from_row("Phases with justified raw code")
    helpers_missed = _parse_int_from_row("Phases with missed helpers")
    helpers_misused = _parse_int_from_row("Phases with misused helpers")

    # Parse compliance rate: "| **Helper compliance rate** | **85%** |"
    rate_match = re.search(
        r"\|\s*\*?\*?Helper compliance rate\*?\*?\s*\|\s*\*?\*?(\d+)%?\*?\*?\s*\|",
        section,
    )
    compliance_rate = int(rate_match.group(1)) if rate_match else None

    # Need at least the compliance rate or enough data to compute a score
    if compliance_rate is None and total_phases is None:
        return None

    # Use compliance rate directly as the raw score if available
    if compliance_rate is not None:
        raw_score = float(compliance_rate)
    elif total_phases and total_phases > 0:
        good = (helpers_correct or 0) + (raw_justified or 0)
        raw_score = (good / total_phases) * 100.0
    else:
        return None

    raw_score = round(min(100.0, max(0.0, raw_score)), 1)

    # Build details string
    details_parts = [f"Helper compliance: {raw_score:.0f}% (from self_reflection.md)"]
    if helpers_missed and helpers_missed > 0:
        details_parts.append(f"{helpers_missed} missed helper opportunities")
    if helpers_misused and helpers_misused > 0:
        details_parts.append(f"{helpers_misused} misused helpers")

    return CriterionResult(
        name="helper_usage",
        raw_score=raw_score,
        weight=0.0,
        weighted_score=0.0,
        details="; ".join(details_parts),
        sub_scores={
            "source": "self_reflection.md",
            "total_phases": total_phases,
            "helpers_correct": helpers_correct,
            "raw_justified": raw_justified,
            "helpers_missed": helpers_missed,
            "helpers_misused": helpers_misused,
            "compliance_rate": compliance_rate,
        },
    )


def score_helper_usage(op_path: Path, state: dict) -> CriterionResult:
    """Score based on how much the compute kernel uses helpers vs raw calls.

    Primary source: the Helper Compliance Summary table in self_reflection.md
    (Section 10), which has a compliance rate computed by the self-reflection
    agent after auditing all kernel phases against the helper library.

    Fallback: direct kernel scanning (original approach).

    The pipeline strongly encourages using helpers from ttnn/cpp/ttnn/kernel_lib/.
    Raw compute calls (tile_regs_acquire, reduce_tile, pack_tile, etc.) in the
    compute kernel are a code smell when helpers exist. CB sync calls
    (cb_wait_front, cb_pop_front) are expected primitives — helpers don't fully
    abstract these because policies like NoWaitNoPop require manual management.

    Also checks the design doc for helper mapping to see if the agent followed
    the architect's recommendations.
    """
    # Try self_reflection.md first (most accurate — based on per-phase audit)
    reflection_result = _get_helper_score_from_self_reflection(op_path)
    if reflection_result is not None:
        return reflection_result

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

    # Dynamically determine which raw ops are "abstracted by helpers" by
    # scanning the kernel_lib .inl files. Any function call that appears
    # inside a helper implementation is something helpers are designed to
    # replace — using it directly in a compute kernel is a code smell.
    # Calls NOT found in helpers (like standalone cb_wait_front for
    # persistent CBs) are expected primitives.
    abstracted_ops = _get_helper_abstracted_ops()

    # Find raw function calls in the compute kernel.
    # First, collect all namespaced calls (compute_kernel_lib::func) — these
    # are helper API calls that should NOT be penalized. The :: and function
    # name can span lines (e.g., "compute_kernel_lib::\n    reduce<...>(").
    namespaced_calls = set(re.findall(r"::\s*([a-z_]+)\s*(?:<|[(])", code))

    # Now find all bare function calls (not namespaced, >= 4 chars)
    all_calls_in_kernel = set()
    for match in re.finditer(r"\b([a-z_]{4,})\s*(?:<[^>]*>)?\s*\(", code):
        name = match.group(1)
        if name in namespaced_calls:
            continue
        if name in (
            "constexpr",
            "kernel_main",
            "compute_kernel_hw_startup",
            "get_compile_time_arg_val",
        ):
            continue
        all_calls_in_kernel.add(name)

    # Partition into abstracted (bad) vs expected (benign)
    raw_compute_ops = sorted(all_calls_in_kernel & abstracted_ops)
    cb_sync_ops = sorted(
        all_calls_in_kernel & {"cb_wait_front", "cb_pop_front", "cb_reserve_back", "cb_push_back"} - abstracted_ops
    )

    # Check design doc for helper mapping recommendations
    design_helpers = _extract_design_helpers(op_path)

    # Compute score — only raw COMPUTE ops penalize, not CB sync ops
    has_helpers = len(helper_includes) > 0 or len(helper_calls) > 0
    has_raw_compute = len(raw_compute_ops) > 0

    if has_helpers and not has_raw_compute:
        # Helpers used for all compute — best case (CB sync ops are fine)
        raw_score = 100.0
        usage_level = "full"
    elif has_helpers and has_raw_compute:
        # Mixed: helpers for some compute phases, raw for others
        total_ops = len(helper_calls) + len(raw_compute_ops)
        helper_ratio = len(helper_calls) / total_ops if total_ops > 0 else 0
        raw_score = 40.0 + (helper_ratio * 60.0)  # 40-100 range
        usage_level = "partial"
    elif not has_helpers and has_raw_compute:
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
    if raw_compute_ops:
        details += f"; raw compute ops: {', '.join(raw_compute_ops[:5])}"
    if cb_sync_ops:
        details += f"; cb sync ops (expected): {', '.join(cb_sync_ops)}"

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
            "raw_compute_ops": raw_compute_ops,
            "cb_sync_ops": cb_sync_ops,
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

    helpers = []

    # Match kernel_lib include paths: kernel_lib/reduce_helpers_compute.hpp
    for match in re.finditer(r"kernel_lib/(\w+)\.hpp", content):
        helpers.append(match.group(1))

    # Match compute_kernel_lib:: namespaced calls (the actual helper API)
    for match in re.finditer(r"compute_kernel_lib::(\w+)", content):
        name = match.group(1)
        # Skip policy/enum names, only keep function names
        if name not in (
            "ReduceInputPolicy",
            "ReduceInputBlockShape",
            "ReduceInputMemoryLayout",
            "BinaryInputPolicy",
            "BinaryInputBlockShape",
            "BroadcastDim",
            "NoAccumulation",
            "Accumulate",
            "WaitUpfrontNoPop",
            "WaitAndPopPerTile",
            "NoWaitNoPop",
            "WaitUpfrontPopAtEnd",
            "BulkWaitBulkPop",
        ):
            helpers.append(name)

    # Match dataflow helper calls: prepare_reduce_scaler, generate_bcast_scaler, etc.
    for match in re.finditer(
        r"\b(prepare_reduce_scaler|generate_bcast_scaler|generate_mask_w|generate_reduce_scaler)\b", content
    ):
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


def score_pipeline_run(
    op_path: Path,
    weights: Optional[dict] = None,
    golden_results_path: Optional[Path] = None,
) -> ScoreReport:
    """Compute the composite score for a pipeline run.

    Args:
        op_path: Path to the operation directory.
        weights: Optional override for criterion weights. Missing keys
                 use defaults. Must sum to 1.0.
        golden_results_path: Optional path to golden_results.txt from
                             eval_test_runner.sh.

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
    additive_results = [
        score_golden_tests(golden_results_path),
        score_test_success(state),
        score_execution_time(state, op_path),
        score_retry_efficiency(state),
        score_red_flags(op_path, state),
    ]
    helper_result = score_helper_usage(op_path, state)

    # Apply additive weights
    base_score = 0.0
    for cr in additive_results:
        cr.weight = w.get(cr.name, 0.0)
        cr.weighted_score = round(cr.raw_score * cr.weight, 2)
        base_score += cr.weighted_score

    # Apply helper_usage as a multiplier on the base score.
    # 100% compliance → 1.0x (no penalty)
    # 0% compliance   → 0.2x (floors the score at 20% of base)
    helper_multiplier = 0.2 + 0.8 * (helper_result.raw_score / 100.0)
    helper_result.weight = helper_multiplier
    helper_result.weighted_score = round(helper_multiplier, 2)

    total_score = round(min(100.0, max(0.0, base_score * helper_multiplier)), 1)
    grade = compute_grade(total_score)

    criteria_results = additive_results + [helper_result]

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

    base_score = 0.0
    for cr in report.criteria:
        name = cr["name"].replace("_", " ").title()
        raw = f"{cr['raw_score']:.1f}"
        if cr["name"] == "helper_usage":
            # Helper usage is a multiplier, show differently
            weight = f"{cr['weighted_score']:.2f}x"
            print(f"  {'-'*20} {'-'*6} {'-'*7} {'-'*9}")
            print(f"  {'Base Score':<20} {'':>6} {'':>7} {base_score:>8.1f}")
            print(f"  {name:<20} {raw:>6} {'mult':>7} {weight:>9}")
        else:
            weight = f"x{cr['weight']:.2f}"
            weighted = f"{cr['weighted_score']:.1f}"
            base_score += cr["weighted_score"]
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
    parser.add_argument(
        "--golden-results",
        type=str,
        default=None,
        help="Path to golden_results.txt from eval_test_runner.sh",
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

    golden_path = Path(args.golden_results) if args.golden_results else None

    report = score_pipeline_run(op_path, weights, golden_results_path=golden_path)

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print_report(report, verbose=args.verbose)

    # Exit with non-zero if score is failing (< 50)
    sys.exit(0 if report.total_score >= 50 else 1)


if __name__ == "__main__":
    main()
