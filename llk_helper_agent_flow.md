# LLK Helper Agent Flow

A human-gated iterative pipeline for developing LLK helpers. Each phase produces a self-contained artifact interpretable by a downstream agent without shared context. Human checkpoints gate design and test-coverage decisions. Validation failures re-enter design rather than producing band-aid fixes. All work happens in a single working directory.

---

## Flow Overview

```
  USER REQUEST
       |
       v
 [Phase 0] Prior Work Detection -----> resume at appropriate phase
       |
       v
 [Phase 1] Research
       |
       v
 [Phase 2] Verification
       |
       v
 [Phase 3] Design Options  <----------+
       |                               |
   ** HUMAN CHECKPOINT 1 **            |  validation failures
       |  (choose approach)            |  re-enter here
       v                               |
 [Phase 4] Test Design                 |
       |                               |
   ** HUMAN CHECKPOINT 2 **            |
       |  (approve coverage)           |
       v                               |
 [Phase 5] Implementation + Validation +
       |
       v
 [Phase 6] Report
```

---

## Phase 0: Prior Work Detection

**Agents**: Orchestrator only (no subagent).

Check for existing artifacts and resume at the appropriate phase:

| Artifact Found | Resume At |
|----------------|-----------|
| None | Phase 1 |
| `catalog.md` | Phase 1b |
| `investigation.md` | Phase 2 |
| `verification.md` | Phase 3 |
| `proposal.md` (approved) | Phase 4 |
| `test_design.md` (approved) | Phase 5 |
| `.hpp` / `.inl` files | Phase 5 (validation sub-stage) |

**Artifact**: `status.md` -- current phase, what was found, resume point.

---

## Phase 1: Research

Build a complete understanding of the target operations before any design work. You cannot design an abstraction covering all cases without first enumerating them.

### Phase 1a: Catalog

**Agents**: Claude Code may launch parallel Explore subagents internally.

Enumerate all operations, group by functional similarity, locate source files:
- **Bottom-up**: Grep LLK function prefixes (`llk_`, `ckernel_`), map signatures, locate device implementations.
- **Top-down**: Grep compute API headers and TTNN op factories, map which ops call which LLK functions, identify groupings.

**Artifact**: `catalog.md` -- operation list, functional groups, file paths to headers/device code/host factories.

### Phase 1b: Investigation

**Agents**: Claude Code may launch parallel Explore subagents internally (one per op group is preferred since groups are independent).

Each group is analyzed across six areas:
1. **Device behavior** -- what the LLK does (tile math, register usage, CB patterns)
2. **Host integration** -- how the program factory calls it (runtime args, compile-time defines)
3. **Usage patterns** -- all call sites across the codebase
4. **Encapsulation boundary** -- what can hide behind a helper vs. what stays exposed
5. **CB management** -- setup, count, special patterns
6. **Existing helpers** -- prior abstractions, common wrappers

**Artifact**: `investigation.md` -- per-group analysis with file:line references for all six areas.

---

## Phase 2: Verification

**Agents**: 1 Explore agent.

A fresh agent checks every factual claim in `investigation.md` against actual source code. Classifies each as **CONFIRMED** / **INCORRECT** / **UNVERIFIABLE**. INCORRECT findings are highest value -- they change the design.

A separate agent is essential: self-verification is unreliable (confirmation bias). LLK code is complex enough that incorrect inferences about register usage, CB ownership, and tile formats are common.

**Artifact**: `verification.md` -- annotated investigation with claim statuses and corrections.

---

## Phase 3: Design Options

**Agents**: 1 general-purpose agent (opus).

Reads all prior artifacts. Produces **2-3 design options**, each containing:
- API contract (function signatures, enums, template parameters)
- Abstraction boundary (what's encapsulated vs. exposed)
- Before/after migration example (concrete code)
- Migration tiers (Tier 1: straightforward, Tier 2: needs refactoring, Tier 3: blocked)
- Trade-off summary (complexity, flexibility, performance risk, migration effort)
- Coverage statement: which cases it handles, which it handles poorly and why

**Artifact**: `proposal.md` -- all options with trade-offs and a recommended default.

### HUMAN CHECKPOINT 1

Present options and **stop**. No further work until the human selects an option (or requests revision), approves the API contract, and confirms migration tiers.

---

## Phase 4: Test Design

**Agents**: 1 general-purpose agent (opus).

Interfaces and tests are designed before implementation. This is deliberate: designing tests with knowledge of the implementation biases them toward confirming the implementation rather than validating the contract. Tests designed from the API contract alone are more likely to catch real bugs.

Using the approved design, produce:

### New Helper Tests

Helper tests should isolate the helper as much as possible -- test the helper's behavior and contract, not the underlying LLK. When a helper test fails, it should be clear the bug is in the helper, not in lower layers.

- Raw LLK baseline (validates our understanding independent of the helper)
- Parameter coverage matrix (data formats x template args x runtime args)
- Helper integration (default invocation, explicit dtype/template/runtime args, chaining)
- Edge cases (boundary tile counts, unsupported combos, zero-size inputs)

### Existing Test Audit

Audit existing call sites and their associated tests. Identify tests that exercise code paths the helper will replace. These tests must be migrated to use the helper -- this provides additional coverage from battle-tested test infrastructure and ensures the helper works in real-world contexts, not just isolated tests.

The audit should produce a list of:
- Existing tests that cover affected call sites
- Which can be migrated to use the helper
- Which cannot (and why)

### Performance Tests

- Helper vs. raw LLK benchmarked across a representative range of tile counts
- Measured via Tracy device kernel duration (not wall-clock)
- Acceptable overhead thresholds determined by human during checkpoint review

**Artifact**: `test_design.md` -- test matrix, expected outcomes, benchmark protocol, existing test audit with migration plan.

### HUMAN CHECKPOINT 2

Present test design and **stop**. Human reviews parameter coverage completeness, edge cases, existing test migration plan, and performance thresholds.

---

## Phase 5: Implementation + Validation

**Agents**: 1 general-purpose agent (opus). Sub-stages are sequential -- each depends on the previous result.

**5a: Write tests** from `test_design.md`. Helper tests will fail until the helper exists.

**5b: Raw LLK validation** -- run baseline tests on device.
- PASS: continue.
- FAIL/HANG: **BLOCKER** -- investigation was wrong. Return to Phase 3 with corrections.

**5c: Implement helper** (`{name}_helpers.hpp`, `{name}_helpers.inl`) per approved design.

**5d: Helper integration** -- run new helper tests.
- PASS: continue.
- FAIL (raw passed): implementation bug. Fix helper, re-run 5d.

**5e: Parameter coverage** -- run full matrix.
- PASS: continue.
- Observed failure (raw works, helper doesn't): **BLOCKER** -- abstraction incomplete. Return to Phase 3.
- Unobserved failure (both fail): record as UNSUPPORTED, continue.

**5f: Migrate existing tests** -- update the call sites identified in the Phase 4 audit to use the helper. Run both the migrated existing tests and the new helper tests together.
- PASS: continue.
- FAIL: fix helper or migration, re-run.

**5g: Performance** -- benchmark helper vs. raw.
- Acceptable: continue. Unacceptable: fix helper, re-run from 5d.
- Thresholds are those approved by the human in Checkpoint 2.

**5h: Migrate Tier 1 call sites**. Re-run full test suite (new + migrated existing) to confirm no regression.

When multiple fix approaches exist, rank by: correctness first, then performance, then simplicity.

**Artifact**: `validation_log.md` -- per-sub-stage results, iteration history, blockers encountered.

---

## Phase 6: Report

**Agents**: 1 agent (sonnet).

**Artifact**: `report.md` containing:
- **Summary**: what was created, overall pass/fail/partial
- **Validation results**: per-op pass/fail, parameter matrix, performance table
- **Migration status**: Tier 1 sites updated, existing tests migrated, Tier 2/3 identified with rationale
- **Open items**: unsupported combos, deferred migrations, known limitations

---

## Agent Summary

| Phase | Agent Type | Count | Model |
|-------|-----------|-------|-------|
| 0 | Orchestrator | 1 | -- |
| 1a | Explore | subagents at Claude Code's discretion | sonnet |
| 1b | Explore | subagents at Claude Code's discretion | sonnet |
| 2 | Explore | 1 | sonnet |
| 3 | general-purpose | 1 | opus |
| 4 | general-purpose | 1 | opus |
| 5 | general-purpose | 1 (sequential sub-stages) | opus |
| 6 | general-purpose | 1 | sonnet |

Sonnet for research/verification (breadth-oriented, grep-heavy). Opus for design, test design, and implementation (architectural reasoning, code generation). Claude Code may internally parallelize subagents within any phase.

---

## Artifacts

Each artifact is a self-contained markdown document interpretable by a different agent instance without shared context.

| Artifact | Phase | Contents |
|----------|-------|----------|
| `status.md` | 0 | Resume state |
| `catalog.md` | 1a | Op list, groups, file paths |
| `investigation.md` | 1b | Per-group deep analysis |
| `verification.md` | 2 | Claim statuses and corrections |
| `proposal.md` | 3 | Design options + human's selection |
| `test_design.md` | 4 | Test matrix, existing test audit + human's approval |
| `validation_log.md` | 5 | Test results, iterations, blockers |
| `report.md` | 6 | Final summary |
