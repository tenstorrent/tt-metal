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

## Missing Capabilities

### 11. No incremental re-run capability
**Status**: Proposed

If Phase 4 Stage 2 fails and you fix something manually, there's no way to resume from that point without manipulating `.tdd_state.json` by hand or re-running the whole pipeline.

**Proposal**: Support `--resume-from-stage N` in the orchestrator.

---

### 12. Architect does not calibrate TDD stage tolerances for bf16 precision
**Status**: Proposed — HIGH PRIORITY

**Discovered in**: layer_norm_rm run (2026-03-11). Stage 2 (center_and_square) set atol=0.1 for a stage computing `(x - mean)^2` with bf16 reduce. The hardware reduce introduces ~0.01 mean error amplified to ~0.375 by squaring. The kernel writer spent 56 minutes and 5 hard attempts debugging what was an inherently impossible tolerance.

**Root cause**: The architect has no instruction to analyze bf16 error propagation when setting tolerances. Intermediate stages involving reduce + nonlinear operations (square, power, reciprocal) amplify bf16 precision loss beyond tight tolerances.

**Proposal**: Add a mandatory "Tolerance Calibration" step to the architect's TDD stage design. For each stage:
1. Identify operations that introduce bf16 precision loss (reduce, transcendental functions)
2. Estimate error amplification from downstream operations (squaring amplifies ~40x for unit-magnitude inputs)
3. Set atol to at least 2x the estimated max error
4. Default: atol >= 0.5 for any stage involving reduce + squaring/power

---

### 13. fp32_dest_acc_en breaks untilize when Wt exceeds halved DEST capacity
**Status**: Proposed

**Discovered in**: layer_norm_rm run (2026-03-11). The kernel writer tried fp32_dest_acc_en=True twice, each time producing zeros in the bottom half of the output (max diff 9.0). fp32 mode halves DEST from 16 to 8 tiles (bf16) or 8 to 4 tiles (fp32), breaking untilize's pack_untilize_block which requires Wt tiles in DEST simultaneously.

**Root cause**: Neither the architect's design document nor the kernel writer's instructions mention the DEST capacity constraint when fp32_dest_acc_en is toggled. The Hardware Constraints Checklist mentions DEST sizes but does not connect them to untilize requirements.

**Proposal**: Add to the architect's Hardware Constraints Checklist template: "If the compute kernel uses untilize and Wt > DEST_AUTO_LIMIT/2 (4 tiles for bf16 half-sync fp32, 8 tiles for bf16 half-sync), fp32_dest_acc_en MUST NOT be enabled." Add the same check to the kernel writer's debugging checklist.

---

### 14. Kernel writer lacks a "numerical mismatch triage" protocol
**Status**: Proposed — HIGH PRIORITY

**Discovered in**: layer_norm_rm run (2026-03-11). The kernel writer spent 5 hard attempts and ~56 minutes on a consistent numerical mismatch (max diff 0.375, same value across attempts 1, 3, 4, 5) without recognizing the pattern as an inherent precision limitation. It tried alternative scaler approaches, fp32 mode, and DPRINT diagnostics before context exhaustion forced an orchestrator intervention.

**Root cause**: The kernel writer has no systematic strategy for distinguishing "implementation bug" from "inherent precision limitation." It defaults to trying code-level fixes until budget exhaustion.

**Proposal**: Add a "Numerical Mismatch Triage" protocol to kernel writer instructions:
1. After first hard attempt with numerical mismatch: check if output pattern is correct (values track reference but with systematic offset, no zeros/NaN/inf)
2. If max diff < 1.0 and pattern is correct: run ONE diagnostic isolating error to a specific compute phase
3. If error is in reduce/transcendental and is consistent across shapes: classify as "bf16 precision limitation"
4. After 2 hard attempts with the same error classification and max diff: escalate to orchestrator requesting tolerance adjustment
5. Do NOT spend more than 3 hard attempts on the same consistent numerical pattern

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
| 12 | Architect tolerance calibration | HIGH | SMALL | HIGH |
| 13 | fp32_dest_acc_en / untilize constraint | MEDIUM | SMALL | MEDIUM |
| 14 | Kernel writer numerical triage protocol | HIGH | SMALL | HIGH |
