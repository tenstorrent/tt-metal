# Op Creation Pipeline — Improvement Tracker

## Architectural Issues

### 1. Kernel writer burns massive context on numerical debugging
**Status**: Proposed — HIGH PRIORITY

A single kernel writer session runs for up to **an hour (300k tokens)** on a stage. The progression of issues within a session:

1. **Compile errors** → resolved quickly
2. **Hangs** → mostly get resolved
3. **Numerical mismatches** → extremely hard to debug, this is where the time burns

The design itself is generally fine. The writer doesn't need escalation back to the designer — it needs better tools and strategies for numerical debugging specifically. The current retry budget (6 HARD failures across writer launches) is essentially never reached; it's the wrong abstraction. All the grinding happens within a single writer session.

**Root cause**: Numerical bugs in tile-based kernels are hard because:
- The writer can see "expected 0.5, got 0.0" but can't easily trace which tile, which CB, which compute step produced the wrong value
- Off-by-one in tile indexing, wrong reduction dimension, incorrect scaler packing, etc. all produce plausible-looking but wrong outputs
- The writer resorts to trial-and-error code changes rather than systematic diagnosis

**Proposal**: TBD — needs discussion. Possible directions:
- Better diagnostic output on numerical failures (per-tile comparison, first divergent tile index, magnitude of error)
- A specialized numerical debugging strategy/checklist the writer follows before making code changes
- Intermediate CB dumping to isolate which kernel stage introduces the error
- Reference comparison at each compute phase (not just final output)

---

### 2. Too many planning stages before touching kernel code ("long leash")
**Status**: DONE

Merged Planner + Designer into a single **Architect** agent (`ttnn-operation-architect`). The pipeline now goes Analyzer → Architect → Builder → Kernel Writer.

**What changed**:
- Planner and Designer agent files removed
- New architect produces single `op_design.md` (Part 1: Architecture, Part 2: Kernel Implementation)
- Architect has full visibility into both architecture AND helper library — validates CB decisions against helper requirements immediately
- One fewer agent launch, one fewer handoff, one fewer document
- Builder reads `op_design.md` Part 1, kernel-writer reads Part 2

---

### 3. `.tdd_state.json` coupling between architect and builder is fragile
**Status**: Proposed

File-based IPC between two agents. If the architect's format drifts, the builder breaks silently. If the orchestrator script has a bug, stages get lost.

**Proposal**: Add schema + version validation. Architect produces structured artifact, builder validates on read. Or have the orchestrator be the source of truth (architect reports stages to orchestrator, orchestrator writes the file for builder).

---

## Efficiency Issues

### 4. No fast path for simple operations
**Status**: Proposed

A trivial elementwise op (single input, single output, one compute phase) still goes through all 5 phases with 6+ agent spawns.

**Proposal**: Add complexity assessment early on:
- Simple (1 kernel, 1 CB, known pattern) → skip to builder + single TDD stage
- Medium (known components, clear composition) → full pipeline
- Complex (novel patterns, multiple references) → full pipeline + extra analysis

---

### 6. Builder runs on Sonnet while everything else uses Opus
**Status**: Proposed

The builder's job — producing correct `ProgramDescriptor` API calls with very specific naming (`ReaderConfigDescriptor` not `ReaderConfig`, positional args not keyword) — is detail-sensitive work where the stronger model helps. The number of "common errors" documented in the builder's prompt suggests Sonnet struggles here.

**Proposal**: Run builder on Opus too, or invest in stronger validation (a linter/type-checker for ProgramDescriptor code rather than relying on the model to memorize API quirks).

---

## Robustness Issues

### 7. Discovery phase uses keyword matching
**Status**: Proposed

Phase 0 pattern-matches on strings like "row-major input" and "sharded" to determine references. Brittle — different phrasing ("input is in RM layout", "data comes in stick format") may miss.

**Proposal**: Use LLM reasoning for format requirements instead of regex-style matching.

---

### 9. No validation between architect output and builder output
**Status**: Proposed

After Phase 3 (Build), nobody checks that the builder's CB allocation matches the architect's design. If the builder allocates CB 0 with 2 pages but the architect's kernel design expects 4, you don't find out until Phase 4 fails at runtime.

**Proposal**: Static cross-check between `op_design.md` CB table and `program_descriptor.py` before any kernel runs.

---

### 12. Architect conflates data-level and tile-level reduce dimensions
**Status**: DONE

**Discovered**: softmax run (2026-03-16). The architect designed a "Unified Compute Design" where "both dim=-1 and dim=-2 use REDUCE_ROW." This is mathematically incorrect: REDUCE_ROW reduces along the W axis within tiles, so reducing along dim=-2 (H) requires REDUCE_COL.

**What changed**: Mandatory per-dimension verification table added to architect instructions. Architect now includes per-dimension ReduceDim/BroadcastDim validation.

---

### 13. Phase 1/2 overlap: architect may start before analyzers commit
**Status**: Proposed

**Discovered**: softmax run (2026-03-16). The architect started at 09:07:24 UTC, but analyzers did not complete their commits until 09:09:17 and 09:09:26. The architect's breadcrumbs show `"input_files":["N/A"]`, suggesting it may not have read the analyzer output. This means the architect may have designed the operation without the benefit of detailed reference analyses, potentially contributing to design errors (e.g., the REDUCE_ROW vs REDUCE_COL issue).

**Root cause**: The orchestrator launches phases based on timing or heuristics rather than hard-gating on predecessor completion.

**Proposal**: The orchestrator must verify that all Phase 1 analyzer commits are confirmed (via git log or file existence checks) before launching the Phase 2 architect. Simple implementation: after launching analyzers, poll for their output files and git commits before proceeding.

---

### 14. Builder includes nonexistent tensor_accessor.hpp in kernel stubs
**Status**: DONE

**Discovered**: softmax run (2026-03-16). The builder's kernel stubs included a nonexistent `tensor_accessor.hpp` header.

**What changed**: Updated builder's include mapping table — TensorAccessor row now states no extra include is needed since it's already provided by `api/dataflow/dataflow_api.h`.

---

### 15. Kernel writer does not generate execution logs
**Status**: Proposed

**Discovered**: softmax run (2026-03-16). The kernel writer (ttnn-kernel-writer-tdd) produced breadcrumbs (34 entries) but no execution log (`ttnn-kernel-writer-tdd_execution_log.md`). By contrast, the builder agent produced both breadcrumbs and a comprehensive execution log with structured sections (Input Interpretation, Execution Timeline, Recovery Summary, Deviations, Handoff Notes, Instruction Improvement Recommendations). The kernel writer's breadcrumbs were also truncated -- the last 2 stages had no breadcrumb coverage, only git commit evidence.

**Root cause**: The kernel writer agent's instructions likely do not include a mandatory execution log generation step. The builder's instructions do include this requirement (evidenced by consistent generation across runs).

**Impact**: The kernel writer runs the longest phase (Phase 4, typically 38-50% of total pipeline time). Without an execution log, self-reflection analysis loses structured recovery tables, handoff notes, and instruction improvement recommendations from the agent with the most implementation experience per run.

**Proposal**: Add mandatory execution log generation to the kernel writer's instructions, triggered at session end. Template should match the builder's format: Sections 1-7 covering Input Interpretation, per-stage Execution Timeline with attempt counts, Recovery Summary table, Deviations from Design, Artifacts produced, Handoff Notes, and Instruction Improvement Recommendations.

---

## Missing Capabilities

### 11. No incremental re-run capability
**Status**: Proposed

If Phase 4 Stage 2 fails and you fix something manually, there's no way to resume from that point without manipulating `.tdd_state.json` by hand or re-running the whole pipeline.

**Proposal**: Support `--resume-from-stage N` in the orchestrator.

---

### 16. Architect does not validate intermediate CB sizing under tile_regs synchronization
**Status**: DONE

**Discovered in**: softmax pipeline run (2026-03-16). Intermediate CBs between sequential compute helpers were sized too small, causing deadlocks.

**What changed**: Added "Intermediate CB Sizing Between Compute Helpers" section to `ttnn-cb-memory-fundamentals.md`. Architect now references this section and validates intermediate CB sizing against block dimensions.

---

### 17. (Removed — incorrect root cause attribution)

---

### 18. Agent relaunch loses debugging context
**Status**: Proposed

**Discovered in**: softmax pipeline run (2026-03-16)

When the orchestrator relaunches a kernel writer mid-stage (e.g., after consecutive hard failures), the new instance starts from scratch — reading op_design.md and reimplementing without any knowledge of the previous session's hypotheses, attempted fixes, or debugging insights.

In the softmax run, the kernel writer was relaunched after 3 failures in softmax_dim_w. The second launch wasted ~3 minutes re-reading and re-implementing before reaching the actual unsolved problem (CB sizing deadlock).

**Proposal**: On relaunch, either (a) the orchestrator passes a "Previous Session Summary" extracted from breadcrumbs (hypotheses + outcomes), or (b) the kernel writer reads its own breadcrumb file on startup and extracts prior context for the current stage.

---

## Priority Matrix

| # | Issue | Impact | Effort | Priority |
|---|-------|--------|--------|----------|
| 1 | Numerical debugging burns context | | | |
| 2 | Long leash (planner/designer gap) | | | DONE |
| 3 | `.tdd_state.json` fragility | | | |
| 4 | No fast path | | | |
| 6 | Builder model choice | | | |
| 7 | Discovery keyword matching | | | |
| 9 | No architect/builder cross-validation | | | |
| 11 | No incremental re-run | | | |
| 12 | Architect conflates reduce dimensions | | | DONE |
| 13 | Phase 1/2 overlap | MEDIUM | SMALL | MEDIUM |
| 14 | Builder bad tensor_accessor.hpp include | | | DONE |
| 15 | Kernel writer missing execution logs | MEDIUM | SMALL | MEDIUM |
| 16 | Intermediate CB sizing validation | | | DONE |
| 18 | Agent relaunch context loss | MEDIUM | MEDIUM | MEDIUM |
