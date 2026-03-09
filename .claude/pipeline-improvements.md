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

### 12. No cross-run learning mechanism
**Status**: Proposed — HIGH PRIORITY
**Discovered**: layer_norm self-reflection (Mar 9, 2026) — Runs 1-3

The pipeline ran 3 times for layer_norm with no knowledge transfer between runs. Each run started from scratch with new analysis, new design, new build, and new kernels. Approximately 3.5 hours of agent compute was wasted across the first two runs, all of which was entirely superseded by the third run.

**Root cause**: The orchestrator has no concept of "prior attempts" for an operation. When re-running, it does not check for existing artifacts, prior failure patterns, or design decisions that were already validated or rejected.

**Proposal**: Before starting a pipeline run, check for prior artifacts (op_design.md, .tdd_state.json, analysis files, REPORT.md). If found, generate a "prior run lessons" summary: (a) what approach was tried, (b) what failed and why, (c) what should be different. Inject this into the architect's context. Consider persisting a `run_history.jsonl` file that accumulates across runs.

---

### 13. Architect does not validate broadcast types against tilized tensor shapes
**Status**: Proposed
**Discovered**: layer_norm self-reflection (Mar 9, 2026) — Runs 1-3

The architect specified `mul<NONE>` (element-wise) for gamma/beta operations, but gamma/beta tensors shaped `[1,1,1,W]` only have valid data in row 0 when tilized. The kernel writer had to independently discover that ROW broadcast was needed.

**Root cause**: The architect does not reason about what happens to tensor data after tilization. A `[1,1,1,W]` tensor tilized into 32x32 tiles has its values in row 0 of each tile; rows 1-31 are padding zeros.

**Proposal**: Add a mandatory "Tilized Data Layout Check" to architect instructions. For every binary/broadcast op, the architect must annotate which region of each CB tile contains valid data (All, Row0, Col0, Scalar) and verify the broadcast type matches.

---

### 14. Reference selection can drive incorrect layout assumptions
**Status**: Proposed
**Discovered**: layer_norm self-reflection (Mar 9, 2026) — Runs 1-3

In Run 2 of layer_norm, the discovery phase selected tilize and untilize as references. This led the architect to design around ROW_MAJOR input/output with tilize/untilize in the compute pipeline, adding unnecessary complexity.

**Root cause**: Reference selection does not consider whether the reference's input/output layout matches the target operation's natural domain.

**Proposal**: Add a layout-match gate to reference selection. If the target operation naturally works on TILE_LAYOUT data (normalization, reduction, matmul), do not select references whose primary function is layout conversion (tilize, untilize).

---

### 15. InterleavedAddrGenFast missing .data_format is a recurring footgun
**Status**: Proposed
**Discovered**: layer_norm self-reflection (Mar 9, 2026) — Run 4

When the kernel writer uses `InterleavedAddrGenFast` to read secondary tensors (gamma, beta), omitting the `.data_format` field produces silently wrong addresses for multi-tile reads. In Run 4 of layer_norm, this caused the only failure: a numerical mismatch with max diff 11.6875 on the scale_shift stage, requiring 1 hard retry to fix.

**Root cause**: `InterleavedAddrGenFast` has `.data_format` as a struct member required for correct bank offset computation in `get_noc_addr()`. The field is not obviously required from the struct name or usage pattern, and the compiler does not warn about its absence (it is default-initialized to zero/garbage).

**Proposal**: Two-pronged fix:
1. Architect instruction: When designing reader kernels that use `InterleavedAddrGenFast` for secondary tensors, always specify the full struct initialization including `.data_format = get_dataformat(cb_xxx)`.
2. Kernel writer checklist: Add mandatory validation item "Every `InterleavedAddrGenFast` must include `.bank_base_address`, `.page_size`, AND `.data_format`."

---

### 16. Builder agent lacks breadcrumb/observability logging
**Status**: Proposed
**Discovered**: layer_norm self-reflection (Mar 9, 2026) — Run 4

The builder is the only pipeline agent without breadcrumb generation. In Run 4, the builder consumed ~13 minutes (38% of total pipeline time) but there is no way to diagnose where that time was spent: CB generation, kernel descriptor creation, test file generation, integration testing, or API lookup.

**Root cause**: Builder agent prompt does not include breadcrumb generation instructions. Other agents (analyzer, architect, kernel writer) all emit breadcrumbs.

**Proposal**: Add breadcrumb JSONL output to the builder at: start, after CB generation, after kernel descriptor generation, after test generation, and at completion. Include sub-phase timing to enable bottleneck identification.

---

### 17. Architect design documents contain visible deliberation/revision text
**Status**: Proposed
**Discovered**: layer_norm self-reflection (Mar 9, 2026) — Run 4

The architect's `op_design.md` contains inline "thinking out loud" text: alternative approaches that were considered and rejected, CB assignment revisions with "Revise: use cb_24...", and "Actually..." corrections. While the final answer is correct, these deliberation sections risk confusing downstream agents (builder and kernel writer) if they read intermediate text as instructions.

**Root cause**: The architect model streams reasoning directly into the design document without a cleanup pass. No instruction tells it to present only final decisions.

**Proposal**: Add architect instruction: "The design document must present only final decisions. Do not include deliberation, alternatives considered, or revision history. Resolve all design choices internally before writing."

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
| 12 | No cross-run learning | HIGH | MEDIUM | HIGH |
| 13 | Broadcast type validation | MEDIUM | SMALL | MEDIUM |
| 14 | Reference layout gate | HIGH | SMALL | HIGH |
| 15 | InterleavedAddrGenFast .data_format footgun | MEDIUM | SMALL | HIGH |
| 16 | Builder lacks observability logging | MEDIUM | SMALL | MEDIUM |
| 17 | Architect deliberation text in design docs | LOW | SMALL | LOW |
