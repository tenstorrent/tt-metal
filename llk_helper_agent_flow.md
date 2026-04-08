# LLK Helper Agent Flow

Each phase produces a self-contained markdown artifact interpretable by a downstream agent without shared context. Human checkpoints gate design and test-coverage decisions. Include the current git commit hash in artifacts for staleness detection across sessions.

**Mode discipline**: Phases 0-4 are analysis only (no file edits, no commands). Phase 5 implements. Phase 6 is read-only.

**Priority**: Correctness > performance > simplicity.

```
 [Phase 0] Prior Work Detection -----> resume at appropriate phase
 [Phase 1] Research
 [Phase 2] Verification
 [Phase 3] Design Options  <--------- failure taxonomy routes here
         ** HUMAN CHECKPOINT 1 **
 [Phase 4] Test Design
         ** HUMAN CHECKPOINT 2 **
 [Phase 5] Implementation + Validation
 [Phase 6] Report
```

---

## Phase 0: Prior Work Detection

**Agents**: Orchestrator only. Check for existing phase artifacts and resume at the earliest incomplete phase.

**Artifact**: `status.md`

---

## Phase 1: Research

**Agents**: Explore subagents (Claude Code may parallelize internally).

### 1a: Catalog

Enumerate all target operations, group by functional similarity, locate source files. Search bottom-up (LLK function prefixes) and top-down (compute API headers, TTNN op factories).

**Artifact**: `catalog.md` -- operation list, groups, file paths.

### 1b: Investigation

Analyze each group across: device behavior, host integration, usage patterns / call sites, encapsulation boundary, CB management, existing helpers.

Every factual claim gets a unique ID (C-001, C-002...) with file:line anchor and a 1-2 line excerpt. This **Claim Ledger** is the interface Phase 2 verifies against.

**Artifact**: `investigation.md` -- per-group analysis as a Claim Ledger.

---

## Phase 2: Verification

**Agents**: 1 fresh Explore agent (separate from Phase 1 to avoid confirmation bias).

Verify every Claim Ledger entry against source code. Classify each as **CONFIRMED** / **INCORRECT** (with correction + evidence) / **UNVERIFIABLE**. Prioritize falsification over confirmation. INCORRECT findings are highest value -- they change the design.

End with summary counts and the top corrections by impact.

**Artifact**: `verification.md` -- annotated Claim Ledger with statuses, corrections, and summary.

---

## Phase 3: Design Options

**Agents**: 1 general-purpose (opus).

Produce **2-3 design options**, each with:
- API contract (signatures, enums, template parameters)
- Abstraction boundary
- Before/after migration example
- Migration tiers (Tier 1: straightforward, Tier 2: needs refactoring, Tier 3: blocked)
- Trade-offs and coverage gaps

**Artifact**: `proposal.md`

### HUMAN CHECKPOINT 1

Present options and **stop**. Human selects an option (or requests revision) and approves the API contract.

---

## Phase 4: Test Design

**Agents**: 1 general-purpose (opus).

Tests are designed before implementation. This is deliberate: designing tests with knowledge of the implementation biases them toward confirming it rather than validating the contract.

### Raw LLK baseline

Tests that validate our understanding of the underlying LLK independent of the helper. These run first in Phase 5 to confirm the foundation before building on it.

### Helper tests

Isolate the helper as much as possible -- when a test fails, it should be clear the bug is in the helper, not lower layers.
- Parameter coverage matrix (data formats x template args x runtime args)
- Integration (default invocation, explicit args, chaining)
- Edge cases

### Existing test audit

Audit existing call sites and their tests. Identify tests that exercise code paths the helper will replace. These must be migrated to use the helper for additional real-world coverage. List which can be migrated and which cannot (and why).

### Performance tests

Helper vs. raw LLK across representative tile counts. Tracy device kernel duration (not wall-clock). Acceptable overhead thresholds determined by human during checkpoint.

**Artifact**: `test_design.md` -- test matrix, existing test audit, benchmark protocol.

### HUMAN CHECKPOINT 2

Present test design and **stop**. Human reviews coverage, existing test migration plan, and performance thresholds.

---

## Phase 5: Implementation + Validation

**Agents**: 1 general-purpose (opus). Sub-stages are sequential.

**5a**: Write tests from `test_design.md`.

**5b**: Run raw LLK baseline on device. FAIL/HANG = **BLOCKER**.

**5c**: Implement helper per approved design.

**5d**: Run helper tests. FAIL (raw passed) = implementation bug, fix and re-run 5d.

**5e**: Run full parameter matrix. Observed failure (raw works, helper doesn't) = **BLOCKER**. Unobserved failure (both fail) = record as UNSUPPORTED.

**5f**: Migrate existing tests from Phase 4 audit. Run all tests (new + migrated) together. FAIL = fix and re-run 5f.

**5g**: Performance benchmark. Unacceptable per Checkpoint 2 thresholds = fix helper, re-run from 5f.

**5h**: Migrate Tier 1 call sites. Re-run all tests to confirm no regression.

### Failure taxonomy

Not all failures are design failures. On any BLOCKER, classify the root cause before looping back:

| Root Cause | Route To |
|------------|----------|
| Design flaw | Phase 3 |
| Research gap (investigation was wrong) | Phase 1/2 |
| Test harness bug | Fix in Phase 5 |
| Environment issue | Escalate to human |

**Artifact**: `validation_log.md`

---

## Phase 6: Report

**Agents**: 1 general-purpose (sonnet).

**Artifact**: `report.md` -- quickstart (how to run tests + benchmarks), API contract synopsis, validation results, migration status (Tier 1 + existing tests), open items (Tier 2/3, unsupported combos).

---

## Agent Summary

| Phase | Type | Model | Notes |
|-------|------|-------|-------|
| 0 | Orchestrator | -- | File-existence check |
| 1 | Explore | sonnet | May parallelize internally |
| 2 | Explore | sonnet | Fresh agent, not Phase 1 agent |
| 3 | general-purpose | opus | |
| 4 | general-purpose | opus | |
| 5 | general-purpose | opus | Sequential sub-stages |
| 6 | general-purpose | sonnet | |

Sonnet for research (grep-heavy). Opus for design, test design, and implementation (reasoning-heavy).
