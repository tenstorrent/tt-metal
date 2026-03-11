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

### 12. DPRINT side effects mask synchronization bugs
**Status**: Proposed — HIGH PRIORITY
**Discovered**: rms_norm pipeline run (2026-03-11)

Adding DPRINT debug statements to kernels introduces extra `cb_wait_front` calls that change CB synchronization timing. In the rms_norm gamma stage, a diagnostic test with DPRINT passed (max diff 0.09375) while the same code without DPRINT failed (max diff 6.4375). The kernel writer spent 25 minutes investigating a "fix" that only worked due to DPRINT side effects.

**Root cause**: DPRINT tile slice (`DPRINT << TSLICE(cb, ...)`) internally performs a `cb_wait_front` which can mask race conditions between compute phases that share CBs. The kernel writer has no way to know that DPRINT is changing behavior.

**Proposal**: Develop a synchronization-safe diagnostic methodology:
- Post-test CB state dump utility that reads L1 memory without affecting runtime behavior
- Documentation warning in kernel-writer prompt: "If adding DPRINT changes test outcome, this is a synchronization bug -- focus on CB wait/pop policies, not data flow"
- Consider a "CB trace mode" that logs push/pop/wait/reserve events to L1 scratchpad without blocking

---

### 13. Design deliberation text propagates as implementation ambiguity
**Status**: Proposed
**Discovered**: rms_norm pipeline run (2026-03-11)

The architect's design for rms_norm Phase 6 (gamma multiply) included unresolved deliberation: "Alternatively, add a dedicated cb_gamma_out. For simplicity, the kernel writer should use c_4 as the gamma multiply output and adjust the writer source CB accordingly." This left the kernel writer to resolve a design decision during implementation. The writer tried the c_4 approach first (87 minutes of debugging), then independently arrived at the CB8 approach (which caused a new hang).

**Root cause**: The architect's prompt does not explicitly prohibit leaving unresolved alternatives in the design document. The word "Alternatively" signals deliberation, not decision.

**Proposal**: Add rule to architect prompt: "The design document must be decisional. Do not include 'Alternatively...', 'For simplicity...', or 'The kernel writer should decide...'. Choose one approach and document it. Rejected alternatives go in a separate section clearly marked as not-to-implement."

---

### 14. CB reuse across compute phases with different page requirements
**Status**: Proposed
**Discovered**: rms_norm pipeline run (2026-03-11)

The architect designed c_4 to be reused for both reduce output (1 tile, Phase 3) and gamma multiply output (Wt tiles, Phase 6). The design specified 2 pages for c_4, which was sufficient for Phase 3 but insufficient for Phase 6. This mismatch was the root cause of the gamma stage debugging spiral (87 minutes, unresolved).

**Root cause**: The architect does not systematically validate maximum page requirements across all phases that use a CB. The "Streaming" label on c_3 (2 pages) was also incorrect for sequential helper calls (needed Wt pages).

**Proposal**:
- Add a CB lifecycle analysis section to the design template: for each CB, list every phase that uses it and the maximum concurrent pages needed
- Add rule: "Never reuse a CB across phases if pages needed differ. Use a dedicated CB instead. CB IDs 0-31 are available."
- Add rule: "Streaming (2-page) sizing is only valid for concurrent producer-consumer pairs on different threads. Sequential compute helper calls need full-buffer sizing."

---

### 15. Device contention during TDD exhausts agent context window
**Status**: Proposed
**Discovered**: rms_norm pipeline run (2026-03-11)

When another agent holds the device lock for 60+ minutes (e.g., running golden tests), the TDD agent's context window fills with lock-wait messages and retry loops rather than productive debugging. The agent cannot checkpoint its state and resume later when the device becomes available.

**Proposal**:
- Implement cooperative device scheduling with per-test-case lock acquisition
- Add TDD agent checkpointing: if device is unavailable for > 5 minutes, save state to `.tdd_state.json` and exit cleanly for later resumption
- Stagger Phase 4 starts across agents on the same machine

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
| 12 | DPRINT side effects mask sync bugs | HIGH | MEDIUM | HIGH |
| 13 | Design deliberation text as ambiguity | MEDIUM | SMALL | MEDIUM |
| 14 | CB reuse across phases | HIGH | SMALL | HIGH |
| 15 | Device contention exhausts context | HIGH | LARGE | MEDIUM |
