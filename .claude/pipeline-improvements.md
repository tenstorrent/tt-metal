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

### 12. Analyzers run sequentially, wasting ~15-20 minutes per pipeline run
**Status**: Proposed — HIGH PRIORITY

Observed in layer_norm_rm run 2: three analyzer instances (tilize ~10m, reduce_w ~9m, untilize ~5m) ran sequentially for a total of ~26 minutes. Since they are completely independent (each reads a different reference file and produces a separate analysis), they could run in parallel, bounded by the slowest analyzer (~10m).

**Evidence**: Breadcrumb timestamps show tilize analyzer completing at 12:03:42, reduce_w starting at 12:04:46, and untilize starting at 12:14:13. No overlap.

**Proposal**: Launch all analyzer subagent instances concurrently from the orchestrator. Wait for all to complete before starting the architect. Expected savings: ~16 minutes per run with 3 analyzers.

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

### 13. Architect design documents contain host-side include paths instead of device-side paths
**Status**: Proposed

Observed in layer_norm_rm runs: the architect references host-side include paths (e.g., `ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp`) in the design document. Device-side kernel compilation requires different paths (e.g., `api/tensor/tensor_accessor.h`). This causes compile failures in the builder phase, consuming retries.

**Evidence**: Builder execution log (layer_norm_rm run 2) explicitly logged this as upstream feedback. Builder hypothesis H1 at 12:38:44: "fatal error: ttnn/cpp/ttnn/tensor/accessor/tensor_accessor.hpp: No such file or directory".

**Proposal**: Add a device-side include path reference table to the architect's instructions. Common mappings: `TensorAccessor -> api/tensor/tensor_accessor.h`, `dataflow API -> api/dataflow/dataflow_api.h`, `compute startup -> api/compute/compute_kernel_hw_startup.h`. Architect should validate all mentioned includes against this table.

---

### 14. Kernel-writer breadcrumb timestamps become unreliable after first TDD stage
**Status**: Proposed

Observed in layer_norm_rm run 2: the kernel-writer-tdd agent emitted breadcrumbs for stages 2-5 with synthetic timestamps (12:00:00Z, 12:01:00Z, 12:02:00Z, etc.) that predate the agent's own start time (12:44:26). Only stage 1 (data_pipeline) had real timestamps. This makes per-stage timing analysis from breadcrumbs impossible.

**Evidence**: Breadcrumb entries 24-45 in `ttnn-kernel-writer-tdd_breadcrumbs.jsonl` all have timestamps in the range 12:00:00Z-12:26:00Z, while the agent started at 12:44:26 and git commits place these stages between 12:48 and 13:11.

**Proposal**: Either (a) have `append_breadcrumb.sh` always generate timestamps server-side using `date -Iseconds`, overriding any timestamp in the caller's JSON, or (b) add a validation rule that rejects timestamps older than the most recent breadcrumb entry.

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
| 12 | Analyzers run sequentially | HIGH | SMALL | HIGH |
| 13 | Host-side include paths in design | MEDIUM | SMALL | HIGH |
| 14 | Kernel-writer breadcrumb timestamps unreliable | MEDIUM | SMALL | MEDIUM |
