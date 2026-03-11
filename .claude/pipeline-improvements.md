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

### 12. TDD tolerance not scaled for cascading operations
**Status**: Proposed — HIGH PRIORITY

**Discovered in**: layer_norm_rm run1 (2026-03-11), affine stage

The TDD framework applies the same tolerance (rtol/atol) to all stages regardless of how many bf16 operations they cascade. A 3-operation chain (normalize + gamma_mul + beta_add) naturally accumulates more bf16 rounding error than a single operation. In the layer_norm_rm run, this caused 5 hard attempts and ~44 minutes of futile precision engineering before tolerance was relaxed from 0.02 to 0.05.

**Proposal**: Two complementary approaches:
1. **Architect-side**: Architect specifies per-stage tolerance in the TDD stage plan, scaling by operation chain depth: `tolerance = base_tol * (1 + 0.5 * additional_ops_beyond_first_stage)`.
2. **Framework-side**: Add "precision floor detection" to the TDD orchestrator: if max_diff is stable (within 10%) across 2+ consecutive hard attempts and is within 5x of tolerance, auto-suggest tolerance relaxation rather than consuming more hard attempts.

---

### 13. fp32_dest_acc_en incompatible with tilize/untilize on Wormhole B0
**Status**: Proposed — HIGH PRIORITY

**Discovered in**: layer_norm_rm run1 (2026-03-11), affine stage

Enabling `fp32_dest_acc_en=True` in the ComputeConfigDescriptor causes the tilize and untilize hardware helpers to produce corrupted output (repeated rows, zeros). This is a known WH B0 LLK limitation (related to MOVD2B/MOVB2D transpose bugs). The kernel writer spent 2 hard attempts (~12 minutes) discovering this at runtime.

**Proposal**: Add as a hard constraint in the architect's instructions and a shared "Known Hardware Limitations" reference: "Any operation using in-kernel tilize or untilize MUST set fp32_dest_acc_en=False. fp32 destination accumulation is incompatible with the tilize/untilize pack path on Wormhole B0."

---

### 14. Architect deliberation text leaks into op_design.md
**Status**: Proposed

**Discovered in**: layer_norm_rm run1 (2026-03-11)

The architect's op_design.md contains inline deliberation text showing the evolution of design decisions (e.g., lines 216-226: "Actually, this creates a problem...", "Revised gamma/beta approach...", "Simplest correct approach..."). While the final approach is correct, the deliberation trail can confuse downstream agents who may try to implement an intermediate approach.

**Proposal**: Add instruction to architect: "Before finalizing op_design.md, remove all deliberation text. Present only the final validated approach. Deliberation history belongs in breadcrumbs, not the design artifact."

---

### 15. TensorAccessorArgs safe-offset pattern for conditional parameters
**Status**: Proposed

**Discovered in**: layer_norm_rm run1 (2026-03-11), normalize stage

`TensorAccessorArgs<N>` reads `get_compile_time_arg_val(N)` at class instantiation time via a static constexpr member. Even inside `if constexpr(false)`, the template is instantiated and triggers "Index out of range" static assertions if the arg doesn't exist. This caused 3 free retries in the kernel writer.

**Proposal**: Document the safe-offset pattern in the architect's kernel args design section: "For conditional TensorAccessor parameters, compute the offset using a ternary OUTSIDE any if constexpr block: `constexpr uint32_t offset = has_param ? TensorAccessorArgs<prev>::next_compile_time_args_offset() : prev_offset;`."

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
| 12 | TDD tolerance not scaled for cascading ops | HIGH | SMALL | HIGH |
| 13 | fp32_dest_acc / tilize-untilize incompatibility | HIGH | SMALL | HIGH |
| 14 | Architect deliberation text in design doc | MEDIUM | SMALL | MEDIUM |
| 15 | TensorAccessorArgs safe-offset pattern | MEDIUM | SMALL | MEDIUM |
