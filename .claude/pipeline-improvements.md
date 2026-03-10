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

### 12. Architect lacks REDUCE_ROW -> broadcast dimension validation
**Status**: Proposed — HIGH PRIORITY

**Observed in**: layer_norm_rm (2026-03-10) -- Phase 7 mul_inv_std specified SCALAR broadcast for a REDUCE_ROW-produced CB. REDUCE_ROW output stores per-row values in column 0, requiring COL broadcast. SCALAR only replicates [0][0], producing wrong normalization for rows 1-31. Cost: 1 hard attempt + ~3m debugging.

**Root cause**: The architect has no rule enforcing that REDUCE_ROW output CBs use COL broadcast in downstream binary ops. The same design doc correctly used COL broadcast for cb_mean (also REDUCE_ROW output) in Phase 3 but incorrectly used SCALAR for cb_inv_std (also REDUCE_ROW output, via cb_var) in Phase 7 -- an internal inconsistency.

**Proposal**: Add a validation rule to the architect's design checklist: "When a CB is populated by a REDUCE_ROW operation, its valid region is Col0. Any downstream binary op must use BroadcastDim::COL (not SCALAR). Similarly, REDUCE_COL output -> valid region is Row0 -> downstream must use ROW broadcast." Cross-check the Binary Op Broadcast Verification table against CB producers before finalizing.

---

### 13. In-place CB reuse with Bulk output policy deadlocks for Wt > 1
**Status**: Proposed — HIGH PRIORITY

**Observed in**: layer_norm_rm (2026-03-10) -- Phases 8-9 (gamma/beta affine transform) used BinaryOutputPolicy::Bulk on cb_out_pre_untilize where the same CB was both input A and output. Bulk reserves all Wt output pages upfront, but the CB already has Wt input pages occupying all capacity. For Wt=1 this works (pop frees space before reserve blocks), but for Wt>1 it deadlocks. Cost: 1 hard attempt + device hang + tt-smi reset.

**Root cause**: The architect did not model CB capacity constraints when selecting output policies for in-place operations. The design doc even noted that "WaitAndPopPerTile for A ensures tiles are consumed before output tiles occupy the same slots" but failed to apply the complementary reasoning to the output policy.

**Proposal**: Add a hard rule to the architect's instructions: "When input CB == output CB for a binary op (in-place reuse), ALWAYS use BinaryOutputPolicy::PerTile. Never use Bulk for in-place operations. Bulk pre-reserves all output pages, which deadlocks when the CB is already full."

---

## Missing Capabilities

### 11. No incremental re-run capability
**Status**: Proposed

If Phase 4 Stage 2 fails and you fix something manually, there's no way to resume from that point without manipulating `.tdd_state.json` by hand or re-running the whole pipeline.

**Proposal**: Support `--resume-from-stage N` in the orchestrator.

---

### 14. Architect does not specify preprocessing for optional tensor parameters
**Status**: Proposed

**Observed in**: layer_norm_rm (2026-03-10) -- gamma/beta tensors (1,1,1,W) are RM but need to be read as tiles by the reader kernel. The architect's design doc did not specify how to convert them, leading to conflicting approaches: the builder advised "leave as RM, reader handles RM data" while the kernel writer correctly pad+tilized them on the host side. This caused 3 upstream_fix events during the affine_transform stage.

**Proposal**: Add a "Parameter Preprocessing" section to the architect's design template. For each optional tensor parameter, specify: (1) input format, (2) host-side preprocessing (pad, tilize, etc.), (3) resulting format/shape when passed to program descriptor.

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
| 12 | REDUCE_ROW -> broadcast dim validation | HIGH | SMALL | HIGH |
| 13 | In-place CB + Bulk output deadlock | HIGH | SMALL | HIGH |
| 14 | Optional param preprocessing not specified | MEDIUM | SMALL | MEDIUM |
