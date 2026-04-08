# LLK Helper Agent Flow

## Design Rationale

This flow synthesizes three sources (in priority order):

1. **Pavle's Agent Flow** -- research, multi-option design with contract, human-gated test design, TDD-first implementation, iterative ranking by correctness and performance
2. **Kernel Helper Pipeline** -- phased catalog/investigation/verification/proposal/validation/implementation with parallel agents and explicit feedback loops
3. **Multi-agent orchestration experience** -- parallel subagents, state tracking, worktree isolation, structured human checkpoints

The result is a **human-gated iterative pipeline**: each phase produces a concrete artifact, human checkpoints gate all design and test-coverage decisions, and feedback loops ensure blockers discovered during validation re-enter the design phase rather than producing band-aid fixes.

---

## Flow Overview

```
  USER REQUEST
       |
       v
 [Phase 0] Prior Work Detection -----> resume at appropriate phase
       |
       v
 [Phase 1] Research (Catalog + Investigation)
       |          parallel Explore agents
       v
 [Phase 2] Verification
       |          claims checked against source
       v
 [Phase 3] Design Options  <----------+
       |                               |
   ** HUMAN CHECKPOINT 1 **            |  feedback loops
       |  (choose approach)            |  from Phase 5
       v                               |
 [Phase 4] Test Design                 |
       |                               |
   ** HUMAN CHECKPOINT 2 **            |
       |  (approve coverage)           |
       v                               |
 [Phase 5] Implementation + Validation +
       |          TDD loop, iterate by correctness & perf
       v
 [Phase 6] Report
```

---

## Phase 0: Prior Work Detection

**Purpose**: Avoid redundant work when resuming or iterating.

**Mechanism**: Check for existing artifacts in the working directory or output location:

| Artifact Found | Resume At |
|----------------|-----------|
| None | Phase 1 |
| `catalog.md` | Phase 1 (investigation) |
| `investigation.md` | Phase 2 |
| `verification.md` | Phase 3 |
| `proposal.md` (approved) | Phase 4 |
| `test_design.md` (approved) | Phase 5 |
| `.hpp` / `.inl` files | Phase 5 (validation sub-stage) |

**Agent**: Orchestrator (main Claude Code instance). No subagent needed -- this is a file-existence check.

---

## Phase 1: Research

**Purpose**: Build a complete, verified understanding of the target operation group before any design work begins. This directly serves Pavle's requirement that the agent "research and come up with an abstraction plan that covers ALL cases with a contract" -- you cannot cover all cases without first enumerating them.

**Structure**: Two sequential sub-phases, each using parallel agents.

### Phase 1a: Catalog

Enumerate all operations, group them by functional similarity, and locate their source files.

**Agents**: 2 Explore agents in parallel:
- **Bottom-up agent**: Grep for LLK function prefixes (`llk_`, `ckernel_`), map function signatures, locate device-side implementations
- **Top-down agent**: Grep compute API headers and TTNN op factories, map which ops call which LLK functions, identify groupings

**Output**: `catalog.md` -- operation list, functional groups, file locators (paths to headers, device code, host factories).

### Phase 1b: Investigation

Deep analysis of each operation group. **One Explore agent per group, all launched in parallel.**

Each agent investigates six focus areas:
1. **Device behavior** -- what the LLK function actually does (tile math, register usage, CB patterns)
2. **Host integration** -- how the program factory calls it (runtime args, compile-time defines)
3. **Usage patterns** -- all call sites across the codebase (grep for actual invocations)
4. **Encapsulation boundary** -- what can be hidden behind a helper vs. what must stay exposed
5. **CB management** -- how circular buffers are set up, how many, any special patterns
6. **Existing helpers** -- any prior abstraction attempts, common wrappers, shared utilities

**Output**: `investigation.md` -- per-group analysis covering all six areas, with specific file:line references.

**Why parallel per-group**: Groups are independent -- an `exp` helper has nothing to do with a `matmul` helper. Parallel agents maximize throughput without cross-contamination of context.

---

## Phase 2: Verification

**Purpose**: Catch hallucinations and incorrect claims before they propagate into design decisions. This phase exists because LLK code is complex and poorly documented -- investigation agents will make mistakes.

**Agent**: 1 Explore agent (or general-purpose if source-code execution is needed).

**Process**:
- For each factual claim in `investigation.md`, check against actual source code
- Classify each claim: **CONFIRMED** / **INCORRECT** / **UNVERIFIABLE**
- INCORRECT findings are highest value -- they change the design

**Output**: `verification.md` -- annotated version of investigation with claim statuses and corrections.

**Why a separate phase**: Verification by the same agent that made the claim is unreliable (confirmation bias). A fresh agent with a skeptical mandate catches errors the investigation agent is blind to.

---

## Phase 3: Design Options

**Purpose**: Present multiple abstraction approaches so the human can choose based on trade-offs they understand better than the agent (team conventions, downstream plans, risk tolerance). This directly implements Pavle's requirement: "agent should provide several options for us to choose from."

**Agent**: 1 general-purpose agent (Opus recommended -- complex architectural reasoning).

**Process**:
1. Read `catalog.md`, `investigation.md`, and `verification.md`
2. Produce **2-3 design options**, each containing:
   - **API contract**: Helper function signatures, enums, template parameters
   - **Abstraction boundary**: What the helper encapsulates vs. what it exposes
   - **CRTP / dispatch pattern**: How operations are dispatched (if applicable)
   - **Before/after migration example**: Concrete code showing a call site before and after
   - **Migration tier assignments**: Which call sites are Tier 1 (straightforward), Tier 2 (needs refactoring), Tier 3 (blocked/deferred)
   - **Trade-off summary**: Complexity, flexibility, performance risk, migration effort

3. For each option, explicitly state:
   - What cases it covers (must be ALL cases from the catalog, per Pavle)
   - What cases it handles poorly and why
   - Performance implications (extra indirection, template bloat, etc.)

**Output**: `proposal.md` -- 2-3 options with full details and a recommended default.

### HUMAN CHECKPOINT 1

The orchestrator presents the options and **stops**. No further work until the human:
- Selects an option (or requests a hybrid / revision)
- Approves the API contract
- Confirms the migration tier assignments

**Why gate here**: Design mistakes caught after implementation cost 10x more to fix. The human has context the agent doesn't (team roadmap, related PRs in flight, hardware errata).

---

## Phase 4: Test Design

**Purpose**: Define the test matrix before writing any implementation code. This directly implements Pavle's requirement: "design test which should work for correctness and perf" and "another human review of test coverage."

**Agent**: 1 general-purpose agent.

**Process**: Using the approved design from Phase 3, produce a test plan covering:

### Correctness Tests
- **Raw LLK baseline tests**: Verify the raw LLK sequences produce correct results against golden references (these tests exist independently of the helper -- they validate our understanding)
- **Parameter coverage matrix**: All combinations of data formats x template args x runtime args that the investigation identified as used in practice
- **Helper integration tests**: Default invocation, explicit dtype, explicit template args, runtime arg forwarding, policy/dispatch selection, operation chaining, feature flags
- **Edge cases**: Boundary tile counts, unsupported format combinations (should error gracefully), zero-size inputs if applicable

### Performance Tests
- **Benchmark protocol**: Helper vs. raw LLK across tile counts (8, 64, 512, 4K, 32K)
- **Acceptance thresholds**: <2% overhead = PASS, 2-5% = REVIEW, >5% = BLOCKER
- **Measurement method**: Tracy device kernel duration (NOT wall-clock)

**Output**: `test_design.md` -- test matrix, expected outcomes, benchmark protocol.

### HUMAN CHECKPOINT 2

The orchestrator presents the test design and **stops**. The human reviews:
- Is the parameter coverage matrix complete?
- Are the right edge cases identified?
- Are the performance thresholds appropriate for this op group?

**Why gate here**: Missing test coverage discovered after implementation means re-doing validation. Human domain knowledge catches gaps the agent can't infer from code alone.

---

## Phase 5: Implementation + Validation

**Purpose**: TDD-first implementation with iterative ranking by correctness and performance. This is the core execution phase, combining the Kernel Pipeline's structured validation sub-stages with Pavle's "iterate, rank solutions by correctness and performance."

**Agent**: 1 general-purpose agent (Opus recommended). Optionally uses worktree isolation for safe code changes.

### Sub-stage 5a: Write Tests First

Implement the tests from `test_design.md`:
- Raw LLK baseline tests
- Parameter coverage tests
- Helper integration tests (these will fail until the helper exists)
- Performance benchmarks

### Sub-stage 5b: Raw LLK Validation

Run the raw LLK baseline tests on device.
- **PASS**: Continue. This confirms our understanding of the underlying operations.
- **FAIL / HANG**: **BLOCKER**. Our investigation was wrong. Feed corrections back to Phase 3, re-design with updated understanding. Reset device with `tt-smi -r` if hung.

### Sub-stage 5c: Implement Helper

Write the helper files (`{name}_helpers.hpp`, `{name}_helpers.inl`) following the approved design from Phase 3.

### Sub-stage 5d: Helper Integration Validation

Run the helper integration tests.
- **PASS**: Continue.
- **FAIL (but raw passed)**: Bug is in the helper code, not the LLK. Fix `.hpp/.inl`, re-run 5d. Do NOT re-enter Phase 3 -- the design is sound, the implementation has a bug.

### Sub-stage 5e: Parameter Coverage

Run the full parameter coverage matrix.
- **PASS**: Continue.
- **Observed failure** (a combination that works with raw LLK but fails with helper): **BLOCKER**. The helper doesn't cover all cases. Re-enter Phase 3 to revise the abstraction boundary.
- **Unobserved failure** (a combination that fails with both raw and helper): Record as UNSUPPORTED. Continue.

### Sub-stage 5f: Performance Validation

Run benchmarks: helper vs. raw LLK.
- **<2% overhead**: PASS.
- **2-5% overhead**: Flag for human review. May be acceptable.
- **>5% overhead**: **BLOCKER**. Fix helper implementation (5c), re-run 5d-5f.

### Sub-stage 5g: Migrate Tier 1 Sites

Update the straightforward call sites to use the new helper. Run 5d-5f again on the migrated code to confirm nothing regressed.

### Feedback Loops

```
5b fail ---------> Phase 3 (understanding was wrong, redesign)
5d fail ---------> 5c (implementation bug, fix and retry)
5e observed fail -> Phase 3 (abstraction doesn't cover all cases)
5f >5% ----------> 5c (performance issue in implementation, fix and retry)
```

**Iteration ranking**: When multiple fix approaches exist, rank by:
1. **Correctness** (does it pass all tests?)
2. **Performance** (overhead vs. raw LLK)
3. **Simplicity** (fewer lines, less template complexity)

This directly implements Pavle's "iterate, rank solutions by correctness and performance."

---

## Phase 6: Report

**Purpose**: Produce a complete record of what was built, validated, and left open.

**Agent**: Orchestrator (main instance) or a lightweight agent.

**Output**: `report.md` containing:

| Section | Contents |
|---------|----------|
| Summary | What was created, overall result (pass/fail/partial) |
| Pipeline trace | Each phase: agent used, output file, status, wall-clock time |
| Validation results | Per-op pass/fail, parameter coverage matrix, performance table |
| Migration status | Tier 1 sites updated, Tier 2/3 sites identified with rationale |
| Open items | Unsupported combinations, deferred migrations, known limitations |
| Recommendations | Next steps for Tier 2/3 migration, performance optimization opportunities |

---

## Agent Architecture Summary

| Phase | Agent Type | Count | Parallelism | Model |
|-------|-----------|-------|-------------|-------|
| 0 | Orchestrator | 1 | -- | opus |
| 1a | Explore | 2 | parallel | sonnet |
| 1b | Explore | N (one per group) | parallel | sonnet |
| 2 | Explore | 1 | sequential | sonnet |
| 3 | general-purpose | 1 | sequential | opus |
| 4 | general-purpose | 1 | sequential | opus |
| 5 | general-purpose | 1 | sequential (sub-stages sequential) | opus |
| 6 | Orchestrator | 1 | -- | sonnet |

**Model selection rationale**:
- **Opus** for design (Phase 3), test design (Phase 4), and implementation (Phase 5): These require complex architectural reasoning, understanding of trade-offs, and correct code generation. Quality matters more than speed.
- **Sonnet** for research (Phase 1) and verification (Phase 2): These are read-heavy, search-heavy tasks where speed and breadth matter. Sonnet is sufficient for grep-and-summarize work.
- **Orchestrator** is the main Claude Code instance coordinating the flow.

---

## Human Checkpoint Protocol

Both checkpoints follow the same protocol:

1. **Present**: Show the artifact (proposal options or test design) with clear formatting
2. **Recommend**: State which option / coverage level the agent recommends and why
3. **Stop**: Do not proceed. Wait for explicit human approval
4. **Record**: Save the human's decision and rationale as context for downstream phases
5. **Resume**: Only after approval, proceed to the next phase

If the human requests changes, the agent revises the artifact and re-presents. This loop continues until approval. There is no automatic timeout or fallback.

---

## State Management

Each phase produces a named artifact file. The orchestrator tracks progress via file existence (Phase 0). For multi-session work:

| File | Phase | Purpose |
|------|-------|---------|
| `catalog.md` | 1a | Operation enumeration |
| `investigation.md` | 1b | Deep analysis per group |
| `verification.md` | 2 | Claim validation |
| `proposal.md` | 3 | Design options (annotated with human's choice) |
| `test_design.md` | 4 | Test matrix (annotated with human's approval) |
| `validation_log.md` | 5 | Running log of test results, iterations, feedback loops |
| `report.md` | 6 | Final summary |

All artifacts live in a designated output directory (e.g., `llk_helpers/{op_group}/`). This allows multiple operation groups to be worked on independently, potentially by different Claude Code instances on different machines.

---

## Multi-Instance Orchestration

For working across multiple operation groups simultaneously (e.g., one machine handles `unary_math`, another handles `binary_ops`):

- Each instance works in its own output directory
- Instances are fully independent -- no shared state beyond the source repo
- The catalog (Phase 1a) can be shared since it covers all groups
- Investigation onward is per-group and can run on separate machines
- Final reports can be merged manually or by a coordinator instance

This leverages the existing pattern of running multiple Claude Code instances in parallel across machines.

---

## Key Design Decisions

### Why TDD-first (tests before implementation)?

Pavle's flow explicitly requires test design before implementation ("once tests are implemented we can implement the actual feature"). This catches design flaws early -- if you can't write a clean test for an API, the API is wrong. It also provides immediate feedback during implementation: every code change can be validated against the approved test suite.

### Why separate Verification (Phase 2)?

LLK code is underdocumented and architecturally complex. Investigation agents operating on grep results and code reading will make incorrect inferences about register usage, CB ownership, and tile format handling. A dedicated verification pass by a fresh agent catches these errors before they become design assumptions. The Kernel Pipeline identified INCORRECT findings as "highest value" because they change the design -- better to find them in Phase 2 than in Phase 5b when raw LLK tests fail.

### Why 2-3 design options instead of 1?

Pavle explicitly requires "several options for us to choose from." Beyond that, LLK helper design involves real trade-offs (template flexibility vs. compile time, abstraction level vs. performance, coverage breadth vs. migration effort) where reasonable engineers would disagree. Presenting options with trade-offs lets the human apply judgment the agent lacks.

### Why Opus for design/implementation, Sonnet for research?

Research phases (1, 2) are breadth-oriented -- many files, many grep queries, summarization. Sonnet handles this well and is faster. Design and implementation phases (3, 4, 5) require reasoning about API contracts, architectural trade-offs, and correct kernel code generation. Opus produces meaningfully better results on these tasks, and the slower speed is acceptable since these phases have human gates anyway.

### Why sequential sub-stages in Phase 5?

Each validation sub-stage depends on the previous one's result. Running raw LLK tests (5b) before helper tests (5d) is essential -- if the raw tests fail, we know the investigation was wrong, not the helper. Running parameter coverage (5e) after integration (5d) ensures we're testing real behavior, not a broken helper. The feedback loops encode specific failure-mode knowledge from the Kernel Pipeline's experience.
