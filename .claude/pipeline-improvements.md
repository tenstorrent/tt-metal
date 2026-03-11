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

## Hardware Compatibility Issues

### 12. fp32_dest_acc_en + reduce helpers incompatibility on Wormhole B0
**Status**: Proposed — HIGH PRIORITY

The architect recommends `fp32_dest_acc_en=True` based on reference analyses (e.g., softmax uses fp32 accumulation). However, the softmax reference achieves fp32 precision via `matmul_tiles`-based reduction, not via `compute_kernel_lib::reduce` helpers. The reduce helpers use a hardware reduce LLK path that has a known MOVD2B/MOVB2D transpose bug on Wormhole B0 when fp32_dest_acc_en is enabled. This causes partial tile packing: 8 valid rows, 8 zeros, repeating.

**Observed in**: layer_norm_rm run1 (2026-03-11), reduce_mean stage. Cost: 1 hard attempt + ~11 minutes debugging.

**Root cause**: The architect's hardware constraints checklist does not include this incompatibility. The architect sees "softmax uses fp32_dest_acc_en=True" and applies it without checking that the reduction method differs.

**Proposal**: Add to the architect's hardware constraints checklist: "When using compute_kernel_lib::reduce helpers (not matmul-based reduction), fp32_dest_acc_en MUST be False on WH B0." Alternatively, create a hardware incompatibility database that the architect consults before finalizing ComputeConfig.

---

### 13. Stage test generation produces broken Python files
**Status**: Proposed — HIGH PRIORITY

The tdd_orchestrator (or architect's stage test template) generates test files with two classes of bugs:

1. **Relative imports**: `from .layer_norm_rm import layer_norm_rm` — tests live in `tests/` directory, not co-located with operation code. Should be `from ttnn.operations.layer_norm_rm import layer_norm_rm`.
2. **Broken reference functions**: The `reference_body` expression from `.tdd_state.json` is injected as a bare expression without `return` and using the shorthand variable name `x` instead of the function parameter `input_tensor`.

The builder has to fix these every run, costing ~3 minutes per pipeline execution.

**Observed in**: layer_norm_rm run1 (2026-03-11), builder phase. Also documented in builder execution log Recommendation 2.

**Proposal**: Fix the test template in tdd_orchestrator to: (a) use absolute imports, (b) wrap reference_body in proper `return` statement with parameter mapping, (c) validate generated Python syntax before writing.

---

### 14. Final TDD stage does not test optional parameter combinations
**Status**: Proposed

When an operation has optional parameters (gamma/beta for normalization, weight/bias for linear ops), the final TDD stage test only exercises the "all defaults" path. For layer_norm_rm, the compute kernel has 4 compile-time code paths (no-gamma-no-beta, gamma-only, gamma+beta, beta-only) but only the first is tested.

**Observed in**: layer_norm_rm run1 (2026-03-11). The full_normalize test only calls `layer_norm_rm(input)` without gamma or beta.

**Proposal**: Require the final TDD stage to include parametrized test cases covering all combinations of optional parameters. Add `extra_test_cases` support to stage definitions in `.tdd_state.json`.

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
| 12 | fp32_dest_acc_en + reduce helpers WH B0 | HIGH | SMALL | HIGH |
| 13 | Stage test generation broken | HIGH | MEDIUM | HIGH |
| 14 | No optional param test coverage | MEDIUM | MEDIUM | MEDIUM |
