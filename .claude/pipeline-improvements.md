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

---

## Precision & Tolerance Issues

### 12. Architect tolerance specification lacks error propagation analysis for intermediate stages
**Status**: Proposed — HIGH PRIORITY

Discovered in layer_norm_rm v2 run2: the architect set `atol=0.1` for the `square_centered` intermediate stage. The bf16 reduce mean error (~0.06) was amplified through squaring by factor `|a+b| ~ 7`, giving theoretical max error ~0.42 -- exceeding the tolerance. The kernel writer spent 57 minutes (50% of pipeline time) and 8 hard attempts debugging a correct kernel against an infeasible tolerance.

**Root cause**: The architect sets tolerances per stage without analyzing how bf16 rounding errors propagate through nonlinear operations (squaring, division, exponentiation). For linear operations, error is additive. For squaring, `|a^2 - b^2| = |a+b| * |a-b|`, so error is amplified by the signal magnitude.

**Proposal**: Add a mandatory "Tolerance Analysis" step to the architect for intermediate TDD stages. For each stage: (1) compute expected bf16 rounding error, (2) analyze error propagation through nonlinear ops, (3) set tolerance >= 2x computed maximum error. Specific rule for squaring: if stage N tolerance is `e` and max magnitude is `M`, stage N+1 (squaring) tolerance must be >= `2 * M * e`.

---

### 13. fp32_dest_acc_en incompatibility with pack_untilize not documented
**Status**: Proposed

Discovered in layer_norm_rm v2 run2: the kernel writer attempted `fp32_dest_acc_en=True` four times to improve reduce precision. Each time, pack_untilize produced zeroed rows. This is a hardware limitation on Wormhole B0 that is not documented in any reference analysis, design document, or hardware constraints checklist.

**Root cause**: The batch_norm reference analysis documents that batch_norm uses the SFPU kernel path when fp32 is enabled (which avoids pack_untilize), but does not explicitly flag the pack_untilize incompatibility as a constraint.

**Proposal**: Add to the architect's Hardware Constraints Checklist: "fp32_dest_acc_en=True is incompatible with pack_untilize (produces zeroed rows). If fp32 accumulation is needed, avoid the untilize helper or use manual tile-by-tile pack." Analyzers should flag this when documenting untilize patterns.

---

### 14. Kernel writer lacks "consistent max_diff" pattern recognition heuristic
**Status**: Proposed

Discovered in layer_norm_rm v2 run2: the kernel writer tried 12 hypotheses over 57 minutes on `square_centered`. The same `max_diff=0.375` appeared across 3 structurally different approaches (reduce helper, matmul_tiles, separate SUM+multiply). This pattern strongly indicates a tolerance/spec issue rather than a kernel bug, but the writer did not recognize it until hypothesis H12.

**Root cause**: The kernel writer's numerical debugging strategy does not include a heuristic for "consistent max_diff across different implementations."

**Proposal**: Add debugging heuristic to kernel writer instructions: "If the same max_diff value (within 10%) appears across 3+ structurally different kernel implementations, the issue is almost certainly NOT in the kernel code. Perform mathematical error analysis and adjust tolerance if justified."
