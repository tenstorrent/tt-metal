# Self-Reflection Report Template

This template is used by the self-reflection agent to produce structured analysis of a pipeline run.
Write the completed report to `{op_path}/self_reflection.md`.

Replace all `{placeholders}` with actual values.

---

# Self-Reflection: {operation_name}

## Metadata
| Field | Value |
|-------|-------|
| Operation | `{operation_name}` |
| Operation Path | `{op_path}` |
| Pipeline Phases Executed | {phases_list} |
| Agents Invoked | {agent_list} |
| Total Git Commits | {commit_count} |
| Total Pipeline Duration | {total_duration} |
| Overall Result | {SUCCESS / PARTIAL / FAILED} |

---

## 1. Pipeline Execution Summary

### Phase Timeline

| Phase | Agent(s) | Duration | Status | Key Observations |
|-------|----------|----------|--------|------------------|
| 0: Discovery | orchestrator | {duration} | {status} | {what happened} |
| 1: Analysis | ttnn-operation-analyzer(s) | {duration} | {status} | {what happened} |
| 2: Design | ttnn-operation-architect | {duration} | {status} | {what happened} |
| 3: Build | ttnn-generic-op-builder | {duration} | {status} | {what happened} |
| 4: TDD Kernels | ttnn-kernel-writer-tdd | {duration} | {status} | {what happened} |
| 5: Report | orchestrator | {duration} | {status} | {what happened} |
| **Total** | | **{total_duration}** | | {Sum of all phase durations. Calculate from earliest Phase 0 start to latest Phase 5 end.} |

### Agent Duration Breakdown

{Per-agent timing derived from breadcrumb `start` → `complete` events, or from first/last git commit timestamps if breadcrumbs are incomplete.}

| Agent | Start Time | End Time | Wall Duration | Retries | Active vs Debugging |
|-------|------------|----------|---------------|---------|---------------------|
| {agent_name} | {ISO timestamp} | {ISO timestamp} | {Xm Ys} | {N} | {e.g., "~5m active, ~20m debugging numerical mismatch"} |

**Duration calculation method**: Use breadcrumb `"event":"start"` and `"event":"complete"` timestamps. If `complete` event is missing, fall back to the last breadcrumb timestamp or last git commit touching this operation's files. Note which method was used.

### Duration Visualization

```
Phase 0  |██|                                           (~{Xm})
Phase 1  |████████|                                     (~{Xm}) {N analyzers in parallel}
Phase 2       |██████|                                  (~{Xm})
Phase 3            |████|                               (~{Xm})
Phase 4                 |████████████████████████████|   (~{Xm}) ← typically longest
Phase 5                                              |██| (~{Xm})
         0    5    10   15   20   25   30   35   40 min

Longest phase: Phase {N} ({Xm}) — {reason}
```

### Time Distribution

| Category | Duration | % of Total | Notes |
|----------|----------|------------|-------|
| Analysis (Phase 0-1) | {duration} | {%} | {N} analyzers |
| Design (Phase 2) | {duration} | {%} | |
| Build (Phase 3) | {duration} | {%} | |
| Kernel implementation (Phase 4) | {duration} | {%} | {N} TDD stages |
| ↳ Productive coding | {duration} | {%} | Writing kernel code that passed |
| ↳ Debugging/retries | {duration} | {%} | Hypothesis→fix→retest cycles |
| Reporting (Phase 5) | {duration} | {%} | |
| **Total** | **{duration}** | **100%** | |

---

## 2. What Went Well

{Credit what worked. Be specific — cite evidence (CB counts, retry counts, stage pass rates, etc.).}

### {N}. {short_title}

**Phase/Agent**: {which phase or agent}
**Evidence**: {concrete data — e.g., "13 CBs all correctly sized, zero CB-related bugs across 4 TDD stages"}
**Why it worked**: {what about the pipeline/instructions enabled this}

---

## 3. Issues Found

{List all issues ordered by severity. Each issue is self-contained: problem + root cause + fix together. Include time/retry cost where measurable.}

### Issue {N}: {short_title}

| Field | Value |
|-------|-------|
| Severity | {HIGH / MEDIUM / LOW} |
| Phase / TDD Stage | {e.g., Phase 4 — subtract_mean} |
| Agent | {which agent} |
| Retries Consumed | {e.g., "2 free retries wasted" or "1 hard attempt consumed"} |
| Time Cost | {estimated duration spent on this issue} |

**Problem**: {What happened. Include file:line references where relevant. Be specific — quote error messages, breadcrumb entries, or design doc lines.}

**Root Cause**: {Why it happened. Trace it back to the responsible agent/artifact. Was it an ambiguous spec? A helper used in the wrong context? A model hallucination?}

**Fix for agents**:
- {Agent 1}: {specific instruction or validation change}
- {Agent 2}: {specific behavioral change}

---

## 4. Efficiency Analysis

### Per-TDD-Stage Breakdown

{Phase 4 is typically the longest. Break it down by TDD stage.}

| TDD Stage | Duration | Attempts (free/hard) | Result | Bottleneck |
|-----------|----------|---------------------|--------|------------|
| {stage_name} | {Xm Ys} | {N free, M hard} | {PASS/FAIL} | {what slowed it down, or "clean"} |

### Time Sinks

{Identify stages or actions that took disproportionately long. Use breadcrumb timestamps to measure actual time spent on each debugging cycle.}

| Rank | Area | Agent | Duration | % of Total | Description | Retry Count | Likely Cause |
|------|------|-------|----------|------------|-------------|-------------|--------------|
| 1 | {area} | {agent} | {Xm Ys} | {%} | {what took long} | {N} | {why} |

### Wasted Work

{Identify work that was done but ultimately discarded or redone.}

| Agent | What Was Done | Why It Was Wasted | How to Avoid |
|-------|--------------|-------------------|--------------|
| {agent} | {description} | {reason} | {suggestion} |

---

## 5. Inter-Agent Communication Issues

{Analyze handoff quality between agents. Did the output of one agent meet the expectations of the next?}

### Handoff {N}: {source_agent} → {target_agent}

| Field | Value |
|-------|-------|
| Artifact Passed | {file name — e.g., op_design.md} |
| Quality | {GOOD / ADEQUATE / POOR} |
| Issues | {what was missing, ambiguous, or wrong} |
| Downstream Impact | {how it affected the receiving agent} |
| Suggestion | {how to improve this handoff} |

---

## 6. Upstream Feedback Synthesis

{Aggregate all `upstream_feedback` breadcrumb events and execution log Section 7 recommendations across all agents into a unified view.}

### Agent Instruction Improvements

| Target | Source Agent | Recommendation | Confidence | Priority |
|--------|-------------|----------------|------------|----------|
| {agent whose instructions should change} | {agent that observed the issue} | {specific change} | {H/M/L} | {H/M/L} |

### Pipeline-Level Improvements

| Area | Observation | Recommendation | Priority |
|------|-------------|----------------|----------|
| {discovery / analysis / design / build / TDD / logging} | {what was observed} | {what to change} | {H/M/L} |

---

## 7. Comparison with Known Issues

{Cross-reference findings with `.claude/pipeline-improvements.md`. Did this run hit any known issues? Did it reveal new ones?}

### Known Issues Encountered

| Issue # | Title | Encountered? | Notes |
|---------|-------|-------------|-------|
| {N} | {from pipeline-improvements.md} | YES/NO | {how it manifested} |

### New Issues Discovered

| Title | Description | Suggested Priority |
|-------|-------------|-------------------|
| {title} | {description} | {H/M/L} |

---

## 8. Actionable Recommendations

{Prioritized list of concrete changes to make. Each should be specific enough to implement.}

### Recommendation {N}: {short_title}

- **Type**: {instruction_change / new_validation / tool_improvement / pipeline_change}
- **Target**: {which file, agent, or script to modify}
- **Change**: {exactly what to do}
- **Expected Benefit**: {what improves}
- **Priority**: {HIGH / MEDIUM / LOW}
- **Effort**: {SMALL / MEDIUM / LARGE}

---

## 9. Overall Assessment

### Pipeline Maturity Score

Rate each dimension (1-5):

| Dimension | Score | Notes |
|-----------|-------|-------|
| Discovery accuracy | {1-5} | {did it find the right references?} |
| Analysis quality | {1-5} | {was the analysis useful to the architect?} |
| Design completeness | {1-5} | {did the design cover all needed details?} |
| Build correctness | {1-5} | {did the builder produce correct infrastructure?} |
| Kernel implementation | {1-5} | {how smoothly did TDD go?} |
| Inter-agent communication | {1-5} | {handoff quality} |
| Logging/observability | {1-5} | {were logs sufficient for this analysis?} |
| Helper usage compliance | {1-5} | {were helpers used when available? see Section 10} |

### Top 3 Things to Fix

1. {most impactful improvement}
2. {second most impactful}
3. {third most impactful}

### What Worked Best

{Summarize the single strongest aspect of this pipeline run — the item from Section 2 with the most impact.}

---

## 10. Helper Usage Audit

{This section is MANDATORY. Every report must include a helper usage analysis. Helpers in `ttnn/cpp/ttnn/kernel_lib/` are the preferred implementation — if a helper exists for an operation, it MUST be used.}

### Available Helpers

{List all helpers found in `ttnn/cpp/ttnn/kernel_lib/` and what they cover.}

| Helper Header | Functions Provided | Relevant to This Op? |
|---------------|-------------------|----------------------|
| `tilize_helpers.hpp` | `tilize<>()` | {YES/NO — why} |
| `untilize_helpers.hpp` | `untilize<>()` | {YES/NO — why} |
| `reduce_helpers_compute.hpp` | `reduce<Policy, Dim>()` | {YES/NO — why} |
| `reduce_helpers_dataflow.hpp` | reduce scaler generation | {YES/NO — why} |
| `binary_op_helpers.hpp` | `add<>()`, `sub<>()`, `mul<>()` | {YES/NO — why} |
| `dest_helpers.hpp` | `DEST_AUTO_LIMIT` | {YES/NO — why} |
| `copy_tile_helpers.hpp` | tile copy utilities | {YES/NO — why} |
| `cb_helpers.hpp` | CB utility functions | {YES/NO — why} |
| {other helpers found} | {functions} | {YES/NO — why} |

### Per-Phase Helper Compliance

{For each kernel phase, compare what the architect designed vs what was implemented.}

| Kernel | Phase | Design Says | Actually Used | Status | Notes |
|--------|-------|-------------|---------------|--------|-------|
| {reader/compute/writer} | {phase name} | {helper name / "raw"} | {helper name / "raw"} | {✅ Correct / ❌ Helper Missed / ❌ Helper Misused / ✅ Raw Justified} | {explanation} |

### Helper Compliance Summary

| Metric | Value |
|--------|-------|
| Total kernel phases | {N} |
| Phases using helpers correctly | {N} (✅) |
| Phases with justified raw code | {N} (✅) |
| Phases with missed helpers | {N} (❌) |
| Phases with misused helpers | {N} (❌) |
| **Helper compliance rate** | **{%}** |

### Redundant CB Operations Around Helpers

{Helpers handle their own `cb_reserve_back`, `cb_push_back`, `tile_regs_acquire/commit/wait/release`. Wrapping helpers with these operations is a bug that causes deadlocks or double-waits. Note: `cb_wait_front` and `cb_pop_front` are sometimes legitimately needed alongside helpers (e.g., inter-phase transitions), so only flag them if they directly duplicate what the helper already does internally for the same CB.}

| Kernel File | Line(s) | Redundant Operation | Helper It Wraps | Risk |
|-------------|---------|--------------------|-----------------|----- |
| {file:line} | {line numbers} | {e.g., `cb_wait_front` before `tilize<>()`} | {helper name} | {deadlock / double-wait / benign} |

{If no redundant operations found, state: "No redundant CB operations detected around helper calls."}

### Missed Helper Opportunities

{For each ❌ Helper Missed entry above, explain what should have been used and why it wasn't.}

| Phase | Available Helper | What Was Written Instead | Likely Cause | Fix |
|-------|-----------------|------------------------|--------------|-----|
| {phase} | {helper function} | {raw code description} | {e.g., "agent didn't check helper library" / "design doc said raw" / "helper misunderstood"} | {which agent's instructions to update and how} |

{If no missed opportunities, state: "All available helpers were used correctly. No missed opportunities."}
