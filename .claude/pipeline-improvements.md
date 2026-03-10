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
| 12 | REDUCE_SCALAR scaler squaring undocumented | HIGH | SMALL | HIGH |
| 13 | Test template extra_args validation | MEDIUM | MEDIUM | MEDIUM |
| 14 | Architect leaves design alternatives unresolved | MEDIUM | SMALL | MEDIUM |

---

### 12. REDUCE_SCALAR scaler squaring is undocumented
**Status**: Proposed — HIGH PRIORITY

`reduce_tile<SUM, REDUCE_SCALAR>` applies the scaler at both the row-reduction and column-reduction stages. The effective scaler is scaler^2. This is not documented anywhere in the pipeline's reference materials, analysis docs, or design templates. The group_norm run (March 10, 2026) burned 1 hard attempt and ~8 minutes debugging this.

**Discovered in**: group_norm pipeline run, Phase 4 normalize stage. Kernel writer had to use DPRINT to confirm mean was off by factor K.

**Proposal**: Add to a "Known Hardware Behaviors" reference document that both the architect and kernel-writer agents consult. Specific entry: "To achieve effective 1/K with REDUCE_SCALAR, pass 1/sqrt(K) as the scaler."

---

### 13. Test template extra_args does not validate against extra_setup/reference_body
**Status**: Proposed

When the architect registers a TDD stage with `extra_setup` that creates variables (e.g., `gamma`, `beta`) and a `reference_body` that uses them, the `extra_args` field must also include those variables in the function call. The template system does not validate this, leading to test bugs where the reference computes the correct answer but the device function receives default arguments.

**Discovered in**: group_norm pipeline run, Phase 4 affine stage. `test_stage_affine.py` created gamma/beta in setup but did not pass them to `group_norm()`. Cost: 1 hard attempt.

**Proposal**: Add a validation step in `tdd_orchestrator.py` that parses `extra_setup` for variable assignments and warns if any variable appearing in both `extra_setup` and `reference_body` is missing from `extra_args`.

---

### 14. Architect leaves design alternatives unresolved for kernel writer
**Status**: Proposed

The architect's Part 2 (Kernel Implementation) sometimes presents multiple approaches without choosing one (e.g., "store in L1 directly or recompute during normalize" or "Simpler: pack mean^2 to cb_tmp"). This forces the kernel writer to evaluate alternatives during implementation, consuming context in an already context-heavy phase.

**Discovered in**: group_norm pipeline run. The architect's Critical Note #4 listed three approaches for stats storage without resolving. The kernel writer chose a fourth approach (expanded CB) not listed.

**Proposal**: Require the architect to make definitive decisions in Part 2. Add instruction: "Do not leave alternatives for the kernel writer. Choose one approach and justify it."
