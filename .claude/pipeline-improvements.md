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
| 12 | Architect broadcast validation lacks guardrails | HIGH | SMALL | HIGH |
| 13 | Design code snippets lack namespace qualification | MEDIUM | SMALL | MEDIUM |
| 14 | Shared writer for non-row-major output | LOW | SMALL | MEDIUM |
| 15 | TDD test reference wrong variable name | LOW | SMALL | LOW |
| 16 | Builder wrong TensorAccessor include path | LOW | SMALL | MEDIUM |

## Design Quality Issues

### 12. Architect broadcast validation lacks guardrails
**Status**: Proposed — HIGH PRIORITY
**Discovered**: softmax run (2026-03-11), Issue 1 in self_reflection.md

The architect confused "single tile output" with "scalar value" when choosing broadcast types. REDUCE_ROW produces a tile with valid data in Col0 (each of 32 rows holds an independent result), but the architect specified SCALAR broadcast instead of COL. The architect even had the correct answer initially (COL) but "self-corrected" it to SCALAR during a Pass 2 revision, making the error worse.

**Impact**: 1 hard attempt, ~8 minutes debugging in kernel writer phase.

**Proposal**: Add explicit validation rule to architect instructions: "Col0 valid region requires COL broadcast. Row0 valid region requires ROW broadcast. SCALAR is only valid when the valid region is a single element at (0,0). A single output tile does NOT imply SCALAR -- tiles always have 32 independent rows and 32 independent columns."

---

### 13. Design code snippets use unqualified namespace identifiers
**Status**: Proposed
**Discovered**: softmax run (2026-03-11), Issue 2 in self_reflection.md

The architect's code snippets in `op_design.md` use bare identifiers like `NoAccumulation{}` instead of `compute_kernel_lib::NoAccumulation{}`. The kernel writer copies these snippets verbatim, causing predictable compilation errors that consume a free retry.

**Proposal**: Require fully-qualified namespace identifiers in all code snippets within `op_design.md`. Add this as a checklist item in the architect's instructions.

---

### 14. Shared writer incorrectly recommended for non-row-major output ordering
**Status**: Proposed
**Discovered**: softmax run (2026-03-11), Issue 5 in self_reflection.md

The architect recommended a shared generic writer for both dim=-1 and dim=-2, but dim=-2 produces tiles in chunked column order (not sequential row-major). The kernel writer had to create a dedicated `writer_h.cpp` to handle the non-sequential DRAM write addresses. This was not a failure (kernel writer handled it correctly, first attempt), but it represents design inaccuracy.

**Proposal**: Add rule to architect instructions: "If the compute kernel produces tiles in non-row-major order, the writer MUST handle address remapping. Do NOT recommend a shared generic writer for such cases."

---

## Template / Tooling Issues

### 15. TDD test reference bodies use wrong variable name
**Status**: Proposed
**Discovered**: softmax run (2026-03-11), Issue 2 in self_reflection.md

The `reference_body` field in `.tdd_state.json` uses `input` as the variable name, but the generated test function parameter is `input_tensor`. All 5 TDD stage tests had broken `pytorch_reference` functions that the kernel writer had to fix before starting stage 1.

**Proposal**: Fix the tdd_orchestrator template to use `input_tensor` in reference body expressions, or add a post-generation lint step.

---

### 16. Builder instructions have wrong TensorAccessor include path
**Status**: Proposed
**Discovered**: softmax run (2026-03-11), Issue 3 in self_reflection.md

The builder's helper-to-include mapping table includes `ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp`, which does not exist in the kernel compilation environment. TensorAccessor is auto-included via `dataflow_api.h`.

**Proposal**: Remove the entry from the mapping table and add a note that TensorAccessor needs no explicit include.

---
