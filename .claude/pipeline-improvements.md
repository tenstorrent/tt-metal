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

### 12. DST register mode incompatibility undocumented
**Status**: Proposed — HIGH PRIORITY
**Discovered**: softmax pipeline run (2026-03-11)

Mixing `acquire_dst/release_dst` (full-sync, used by matmul accumulation) with `tile_regs_acquire/release` (half-sync, used by reduce helpers and per-tile pack) causes DST register deadlocks. In the softmax run, the architect recommended matmul-based sum accumulation (from tt-train reference) which uses `acquire_dst`. The kernel writer implemented it alongside reduce helper calls, causing a device hang that consumed 2 hard attempts and ~18 minutes.

Neither the architect's instructions, the kernel writer's instructions, nor any reference documentation warns about this incompatibility.

**Proposal**: Add a "DST Register Mode Compatibility" section to both the architect and kernel writer agent prompts:
- `tile_regs_acquire/release` = half-sync mode (reduce helpers, per-tile compute+pack)
- `acquire_dst/release_dst` = full-sync mode (matmul accumulation, persistent DST)
- These modes CANNOT be mixed in the same compute phase
- When a design needs both reduce helpers and sum accumulation, use `reduce<SUM>` with `post_reduce_op` instead of manual matmul accumulation

---

### 13. Architect stage test generation produces invalid Python
**Status**: Proposed
**Discovered**: softmax pipeline run (2026-03-11)

The architect's auto-generated stage test files consistently contain syntax errors: missing `return` in `pytorch_reference()`, missing commas between parameters and `extra_args` in function signatures, wrong variable names (`x` instead of `input_tensor`), and relative import paths that fail from the test directory. The builder must fix all stage test files every time.

In the softmax run, all 4 stage test files needed fixes (commit 51d95a2417 shows 17-23 lines modified per file).

**Proposal**: After generating each stage test file, the architect should validate with `python -c "import ast; ast.parse(open(path).read())"`. The template should be fixed to:
1. Always prefix reference body with `return`
2. Separate `extra_args` from parameters with `, ` not raw concatenation
3. Use `input_tensor` consistently (not `x`) as the reference function parameter

---

### 14. Architect uses "mandatory" for precision preferences, blocking fallback exploration
**Status**: Proposed
**Discovered**: softmax pipeline run (2026-03-11)

The architect's design doc stated matmul-based sum was "mandatory for the sum phase" (op_design.md line 307). This was actually a precision preference, not a hardware constraint. The kernel writer invested ~18 minutes trying to make matmul work before discovering `reduce<SUM>` achieves acceptable precision (tests pass at rtol=0.05, atol=0.2).

**Proposal**: Introduce terminology standards for design docs:
- "REQUIRED" / "MUST" = hardware constraint or correctness requirement (e.g., "scaler CB MUST be Float16_b")
- "RECOMMENDED" / "SHOULD" = best practice with explicit fallback (e.g., "RECOMMENDED: matmul-based sum for precision. FALLBACK: reduce<SUM> + recip_tile post_reduce_op")
- Each "RECOMMENDED" item should include CB sizing implications for the fallback path

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
| 12 | DST register mode incompatibility | HIGH | SMALL | HIGH |
| 13 | Stage test generation invalid Python | MEDIUM | SMALL | MEDIUM |
| 14 | "Mandatory" vs "recommended" ambiguity | MEDIUM | SMALL | MEDIUM |
