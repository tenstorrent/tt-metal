# LLK Helper Agent Flow

A human-gated iterative pipeline for developing LLK helpers. Each phase produces an artifact that fully conveys its findings to downstream agents without shared context. Human checkpoints gate all design and test-coverage decisions. Feedback loops from validation re-enter design rather than producing band-aid fixes.

---

## Flow Overview

```
  USER REQUEST
       |
       v
 [Phase 0] Prior Work Detection -----> resume at appropriate phase
       |
       v
 [Phase 1] Research                     2+ Explore agents, PARALLEL
       |
       v
 [Phase 2] Verification                 1 Explore agent, SERIAL
       |
       v
 [Phase 3] Design Options  <----------+
       |                               |
   ** HUMAN CHECKPOINT 1 **            |  feedback from
       |  (choose approach)            |  Phase 5
       v                               |
 [Phase 4] Test Design                 |
       |                               |
   ** HUMAN CHECKPOINT 2 **            |
       |  (approve coverage)           |
       v                               |
 [Phase 5] Implementation + Validation + 1 agent, SERIAL sub-stages
       |
       v
 [Phase 6] Report                        1 agent, SERIAL
```

---

## Phase 0: Prior Work Detection

**Agents**: Orchestrator only (file-existence check, no subagent).

Check for existing artifacts and resume at the appropriate phase:

| Artifact Found | Resume At |
|----------------|-----------|
| None | Phase 1 |
| `catalog.md` | Phase 1b (investigation) |
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

**Agents**: 2 Explore agents, PARALLEL.
- **Bottom-up**: Grep LLK function prefixes (`llk_`, `ckernel_`), map signatures, locate device implementations.
- **Top-down**: Grep compute API headers and TTNN op factories, map which ops call which LLK functions, identify groupings.

**Artifact**: `catalog.md` -- operation list, functional groups, file paths to headers/device code/host factories.

### Phase 1b: Investigation

**Agents**: N Explore agents (one per op group), ALL PARALLEL.

Each agent covers:
1. **Device behavior** -- what the LLK does (tile math, register usage, CB patterns)
2. **Host integration** -- how the program factory calls it (runtime args, compile-time defines)
3. **Usage patterns** -- all call sites across the codebase
4. **Encapsulation boundary** -- what can hide behind a helper vs. what stays exposed
5. **CB management** -- setup, count, special patterns
6. **Existing helpers** -- prior abstractions, common wrappers

Groups are independent, so parallel agents maximize throughput without context cross-contamination.

**Artifact**: `investigation.md` -- per-group analysis with file:line references for all six areas.

---

## Phase 2: Verification

**Agents**: 1 Explore agent, SERIAL.

A fresh agent checks every factual claim in `investigation.md` against actual source code. Classifies each as **CONFIRMED** / **INCORRECT** / **UNVERIFIABLE**. INCORRECT findings are highest value -- they change the design.

A separate agent is essential: self-verification by the investigation agent is unreliable (confirmation bias). LLK code is complex enough that incorrect inferences about register usage, CB ownership, and tile formats are common.

**Artifact**: `verification.md` -- annotated investigation with claim statuses and corrections.

---

## Phase 3: Design Options

**Agents**: 1 general-purpose agent (opus), SERIAL.

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

**Agents**: 1 general-purpose agent (opus), SERIAL.

Interfaces and tests are designed before implementation. This is deliberate: designing tests with knowledge of the implementation biases them toward confirming the implementation rather than validating the contract. Tests designed from the API contract alone are more likely to catch real bugs.

Using the approved design, produce:

**Correctness tests**:
- Raw LLK baseline (validates our understanding independent of the helper)
- Parameter coverage matrix (data formats x template args x runtime args)
- Helper integration (default invocation, explicit dtype/template/runtime args, chaining)
- Edge cases (boundary tile counts, unsupported combos, zero-size inputs)

**Performance tests**:
- Helper vs. raw LLK across tile counts (8, 64, 512, 4K, 32K)
- Thresholds: <2% = PASS, 2-5% = REVIEW, >5% = BLOCKER
- Measured via Tracy device kernel duration (not wall-clock)

**Artifact**: `test_design.md` -- test matrix, expected outcomes, benchmark protocol.

### HUMAN CHECKPOINT 2

Present test design and **stop**. Human reviews parameter coverage completeness, edge cases, and performance thresholds.

---

## Phase 5: Implementation + Validation

**Agents**: 1 general-purpose agent (opus), SERIAL sub-stages.

Each sub-stage depends on the previous result. Running raw tests before helper tests is essential -- if raw fails, the investigation was wrong, not the helper.

**5a: Write tests** from `test_design.md`. Helper tests will fail until the helper exists.

**5b: Raw LLK validation** -- run baseline tests on device.
- PASS: continue.
- FAIL/HANG: **BLOCKER** -- investigation was wrong. Return to Phase 3 with corrections.

**5c: Implement helper** (`{name}_helpers.hpp`, `{name}_helpers.inl`) per approved design.

**5d: Helper integration** -- run integration tests.
- PASS: continue.
- FAIL (raw passed): implementation bug. Fix `.hpp/.inl`, re-run 5d.

**5e: Parameter coverage** -- run full matrix.
- PASS: continue.
- Observed failure (raw works, helper doesn't): **BLOCKER** -- abstraction incomplete. Return to Phase 3.
- Unobserved failure (both fail): record as UNSUPPORTED, continue.

**5f: Performance** -- benchmark helper vs. raw.
- <2%: PASS. 2-5%: flag for review. >5%: **BLOCKER** -- fix helper (5c), re-run 5d-5f.

**5g: Migrate Tier 1 sites**. Re-run 5d-5f to confirm no regression.

### Feedback Loops

```
5b fail ---------> Phase 3  (understanding wrong, redesign)
5d fail ---------> 5c       (implementation bug, fix and retry)
5e observed fail > Phase 3  (abstraction incomplete)
5f >5% ----------> 5c       (performance issue, fix and retry)
```

When multiple fix approaches exist, rank by: correctness first, then performance, then simplicity.

**Artifact**: `validation_log.md` -- per-sub-stage results, iteration history, feedback loop entries.

---

## Phase 6: Report

**Agents**: 1 agent (sonnet), SERIAL.

**Artifact**: `report.md` containing:
- **Summary**: what was created, overall pass/fail/partial
- **Validation results**: per-op pass/fail, parameter matrix, performance table
- **Migration status**: Tier 1 sites updated, Tier 2/3 identified with rationale
- **Open items**: unsupported combos, deferred migrations, known limitations

---

## Agent Summary

| Phase | Agent Type | Count | Execution | Model |
|-------|-----------|-------|-----------|-------|
| 0 | Orchestrator | 1 | -- | -- |
| 1a | Explore | 2 | parallel | sonnet |
| 1b | Explore | N (per group) | parallel | sonnet |
| 2 | Explore | 1 | serial | sonnet |
| 3 | general-purpose | 1 | serial | opus |
| 4 | general-purpose | 1 | serial | opus |
| 5 | general-purpose | 1 | serial (sub-stages serial) | opus |
| 6 | general-purpose | 1 | serial | sonnet |

**Model rationale**: Sonnet for research/verification (breadth-oriented, grep-heavy). Opus for design, test design, and implementation (architectural reasoning, code generation).

---

## Artifacts

All artifacts live in a designated output directory (e.g., `llk_helpers/{op_group}/`). Each artifact is a self-contained markdown document that can be interpreted by a different agent instance without shared context.

| Artifact | Phase | Contents |
|----------|-------|----------|
| `status.md` | 0 | Resume state |
| `catalog.md` | 1a | Op list, groups, file paths |
| `investigation.md` | 1b | Per-group deep analysis |
| `verification.md` | 2 | Claim statuses and corrections |
| `proposal.md` | 3 | Design options + human's selection |
| `test_design.md` | 4 | Test matrix + human's approval |
| `validation_log.md` | 5 | Test results, iterations, feedback |
| `report.md` | 6 | Final summary |

---

## Multi-Instance Orchestration

For working across multiple operation groups simultaneously:
- Each instance works in its own `llk_helpers/{op_group}/` directory
- Instances are fully independent -- no shared state beyond the repo
- `catalog.md` (Phase 1a) can be shared across groups
- Investigation onward is per-group and can run on separate machines
- Final reports are merged manually or by a coordinator
