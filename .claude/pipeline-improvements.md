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

## Design Quality Issues

### 12. Architect does not validate host-side API constraints for sub-tile parameter tensors
**Status**: Proposed
**Discovered**: layer_norm_rm v2 run1 (2026-03-10), self-reflection

When the architect recommends host-side API calls (e.g., `ttnn.tilize()`) for parameter tensors like gamma/beta with shape (1,1,1,W), the recommendation may be silently invalid. `ttnn.tilize()` requires H >= 32, W >= 32, and volume >= 1024 (TILE_HW). A gamma tensor with H=1 fails this constraint, causing a runtime `TT_FATAL`.

In the layer_norm_rm run, the architect's design (Critical Notes item 4) recommended "host converts to TILE before kernel launch (simpler, recommended for initial impl)" without noting the volume constraint. The kernel writer discovered this at runtime, consuming 1 hard attempt and ~5 minutes.

**Proposal**: Add validation to the architect agent: when recommending host-side tilize for parameter tensors, verify the tensor shape meets `ttnn.tilize()` constraints. If parameters have H=1 (common for gamma, beta, bias, scalers), the design must explicitly include the `ttnn.repeat` or `ttnn.pad` step before tilize.

---

### 13. Analyzer output volume disproportionate to architect consumption
**Status**: Proposed
**Discovered**: layer_norm_rm v2 run1 (2026-03-10), self-reflection

Three analyzers produced 87KB of combined analysis (tilize: 20KB, untilize: 24KB, softmax: 43KB). The architect consumed the key findings from each in approximately 2 seconds per file (based on breadcrumb timestamps: 12:08:55 to 12:09:00 for all three reads). The vast majority of analysis content — line-by-line code commentary, full function transcriptions — was never referenced during design.

Despite the volume, the tilize analysis missed the `ttnn.tilize()` minimum volume constraint (TILE_HW=1024), which directly caused a downstream failure. Depth without targeted coverage of API constraints is inefficient.

**Proposal**: Cap analysis output at ~10KB per reference. Use a structured format with mandatory sections: (1) Data Flow Pattern, (2) CB Layout, (3) Helper Call Signatures with exact parameter types, (4) API Constraints and Edge Cases, (5) Key Code Patterns. Eliminate line-by-line code walkthroughs.

---

### 14. Builder confuses buffer_page_size() with tile_size() for hybrid RM/tile operations
**Status**: Proposed
**Discovered**: layer_norm_rm v2 run1 (2026-03-10), self-reflection

When an operation has RM (row-major) input/output but tile-based internal computation (common for operations that tilize on-chip), the builder uses `input_tensor.buffer_page_size()` for CB page sizes. For RM tensors, `buffer_page_size()` returns the stick size (W * 2 bytes), not the tile size (2048 bytes for bf16). This produces incorrect CB page sizes for all tile-based intermediate CBs.

In the layer_norm_rm run, the kernel writer caught and fixed this during stage 1 (data_pipeline), replacing `buffer_page_size()` with `ttnn.tile_size(ttnn.bfloat16)`. The fix is now documented in the program descriptor's comments, but the builder's instructions do not guard against this confusion.

Related to #9 (no architect/builder cross-validation) — a static cross-check would catch this. However, this is worth tracking separately because it represents a specific builder instruction gap, not just a missing validation step.

**Proposal**: Add explicit instruction to the builder: "For operations where input_tensor.layout == ROW_MAJOR_LAYOUT but CBs are tile-sized, ALWAYS use `ttnn.tile_size(dtype)` for tile CB page sizes. NEVER use `input_tensor.buffer_page_size()` for tile CBs — it returns stick_size for RM tensors."

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
| 12 | Architect API constraint validation for sub-tile tensors | Medium | Small | Medium |
| 13 | Analyzer output volume vs consumption | Low | Medium | Low |
| 14 | Builder buffer_page_size vs tile_size confusion | Medium | Small | Medium |
