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

### 12. Kernel include path mapping table uses host-side paths for device kernels
**Status**: Proposed — encountered in layer_norm_rm run (2026-03-10)

The system prompt's include mapping table gives host-side `.hpp` paths (e.g., `ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp`) that don't exist in the kernel compilation include path. Device-side kernels need `api/` prefixed `.h` paths (e.g., `api/tensor/tensor_accessor.h`, `api/compute/compute_kernel_api.h`). Every builder run hits this as a 1 free retry.

**Evidence**: Builder breadcrumb recovery event H1 (2026-03-10T12:26:10Z), builder execution log Section 7 Recommendation 1.

**Proposal**: Fix the include mapping table in the builder's system prompt / reference documentation. Map: TensorAccessor -> `api/tensor/tensor_accessor.h`, compute startup -> `api/compute/compute_kernel_api.h`.

---

### 13. Architect Part 1 runtime arg tables may omit args used only for CB fills
**Status**: Proposed — encountered in layer_norm_rm run (2026-03-10)

The architect's Part 1 reader runtime args table listed indices 0-4 but omitted index 5 (epsilon). The epsilon was documented only in Part 2 Critical Notes. The builder followed Part 1 and produced a program descriptor without epsilon in the reader args. The kernel writer had to fix this during the normalize TDD stage.

**Evidence**: Kernel writer breadcrumbs upstream_fix events (2026-03-10T12:38:43-12:39:07), op_design.md Part 1 vs Part 2 inconsistency.

**Proposal**: Add a cross-reference validation rule to the architect instructions: for every `prepare_reduce_scaler`, `fill_with_val`, or CB-fill call in Part 2 that takes a runtime parameter, the corresponding runtime arg must appear in the Part 1 arg table.

---

### 14. Kernel writer TDD agent does not generate structured execution logs
**Status**: Proposed — observed in layer_norm_rm run (2026-03-10)

The builder agent generates both breadcrumbs AND a structured execution log (with recovery tables, handoff notes, instruction improvement recommendations). The kernel writer TDD agent generates only breadcrumbs. Since the kernel writer is the most complex and time-consuming agent, the absence of structured logs reduces post-hoc analysis quality.

**Proposal**: Add execution log generation to the kernel writer TDD agent's completion protocol, following the same format as the builder's execution log.
