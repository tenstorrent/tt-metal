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

### 12. Architect does not validate same-thread sequential CB page counts
**Status**: Proposed — HIGH PRIORITY
**Discovered**: rms_norm run (2026-03-11)

When two compute helpers run sequentially on the same compute thread (e.g., `square()` produces into `cb_xsq`, then `reduce()` consumes from `cb_xsq`), the intermediate CB must hold the full batch of tiles (typically Wt pages). The architect incorrectly specified "Streaming" (1 page) for `cb_xsq` in the rms_norm design, because streaming semantics only work when producer and consumer run on different hardware threads (reader-compute or compute-writer). Same-thread sequential operations require the full output to be buffered before the next operation starts consuming.

This caused 3 of 5 total failures in the rms_norm run (60% of all failures), consuming ~25 minutes of debugging time. The 1x1x32x32 (Wt=1) test passed because 1 page is sufficient for Wt=1, masking the issue until larger shapes were tested.

**Proposal**: Add an architect-level validation rule: "When a CB is both produced and consumed entirely within the compute thread (no reader/writer boundary), its page count must equal the full batch size (Wt for row operations), not 1. Mark these CBs as 'Same-Thread Block' in the CB table, not 'Streaming'." Add a checklist item requiring the architect to verify each intermediate CB's producer-consumer thread assignment.

---

### 13. Architect does not detect in-place CB usage in binary op helpers
**Status**: Proposed
**Discovered**: rms_norm run (2026-03-11)

The architect specified `final_tile_cb = cb_x` for the no-gamma RM path, meaning Phase 6 (`mul<COL>`) would read from `cb_x` AND write back to `cb_x`. Binary op helpers do not support in-place operation (same CB as both input and output). The kernel writer caught this during implementation and deviated from the design, using `cb_normed` instead.

While this did not consume any retries (caught before testing), it represents a design error that could cause hangs if the kernel writer followed the design doc literally.

**Proposal**: Add an architect-level validation rule: "A binary op helper's output CB must be distinct from both input CBs. If you need to reuse a CB index to save L1, introduce a new intermediate CB instead." Add this to the binary op broadcast verification table.

---

### 14. Architect hallucinates non-existent kernel include paths
**Status**: Proposed
**Discovered**: rms_norm run (2026-03-11)

The architect specified `sfpu_init.h` as a required include for the compute kernel, but this header does not exist in the codebase. The architect also used `compute_kernel_api/eltwise_unary/rsqrt.h` instead of the correct `api/compute/eltwise_unary/rsqrt.h`. Similarly, the host-side path `ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp` was specified instead of the device-side `api/tensor/tensor_accessor.h`.

These caused 1 free retry in the kernel writer and 2 compilation attempts in the builder (~5 minutes total).

**Proposal**: Add a validated reference table of device-side kernel include paths to the architect's instructions. Instruct the architect to ONLY use paths from this list. Key entries: `api/compute/compute_kernel_hw_startup.h`, `api/compute/eltwise_unary/rsqrt.h`, `api/dataflow/dataflow_api.h`, `api/tensor/tensor_accessor.h`, `ttnn/cpp/ttnn/kernel_lib/*.hpp`.

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
| 12 | Same-thread CB page count validation | HIGH | SMALL | HIGH |
| 13 | In-place CB deadlock detection | MEDIUM | SMALL | MEDIUM |
| 14 | Include path hallucination | MEDIUM | SMALL | MEDIUM |
