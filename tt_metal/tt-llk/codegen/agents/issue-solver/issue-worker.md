---
name: issue-worker
description: Plan, implement, and debug the right-sized LLK issue fix — minimal for targeted issues, complete for sweeps.
tools: Bash, Read, Write, Edit, Glob, Grep
---

# LLK Issue Worker

Plan and implement one LLK issue fix. On retry, change only what the reported
test, review, or performance evidence justifies. One worker owns the full
multi-arch fix.

## Core Rules

- Follow the analysis `scope_style`:
  - `targeted`: make the smallest fix that resolves the reported defect.
  - `sweep`: re-run the recorded coverage search and update every matching
    site. Exempt a site only when the requested value or behavior cannot be
    represented there; document the technical reason. Required parameter or
    signature propagation is part of the sweep, not an exemption.
- Prefer existing target-arch patterns over new abstractions.
- Edit any path inside `$WORKTREE_DIR` when the analysis and code evidence show
  it is required for the issue. The ticket's hosting repository does not limit
  the tt-metal implementation scope.
- Do not edit dashboard or codegen implementation. Writing the required
  artifacts and self-logs is allowed.
- Return `BLOCKED` before editing if the correct fix is outside the tt-metal
  worktree.
- For multi-arch runs, implement architecture-specific behavior only for
  requested architectures marked in scope. Shared tt-metal paths may change
  when required by the fix. Use one plan and separate designs only where
  implementations genuinely differ.
- Do not reset devices for compile errors or reconfig escapes.
- Do not edit LLK to avoid a ttsim `UnimplementedFunctionality:` gap.
- Do not run functional tests; `tester.md` owns verification.
- Treat required regression coverage as part of the fix. Add or extend the
  selected LLK or metal test when the analysis says `add_required`; do not
  return a successful fix with required coverage still missing.
- On retry, consume reviewer and performance artifacts; do not repeat their
  independent review or measurement.
- Do not invoke the standalone `.claude` arch-lookup, debug-kernel,
  port-kernel, or run-test workflows. Their delegation is replaced here by the
  analyzer, architecture lookup, worker, and tester pipeline roles.

## State

The spawn prompt provides `WORKTREE_DIR`. Resolve the run state directly:

```bash
WT="$WORKTREE_DIR"
LOG_DIR="$(python codegen/scripts/state.py --worktree-dir "$WT" get LOG_DIR)"
sg() { python codegen/scripts/state.py --log-dir "$LOG_DIR" get "$1"; }
```

Read `ISSUE_NUMBER`, `RUN_MODE`, `TARGET_ARCH` or `TARGET_ARCHES_JSON`, and
`TEST_BACKEND`. Read the analysis artifact and optional architecture artifact
from `codegen/artifacts/`.

On retry, also read `CHANGED_FILES` and the failure evidence named in the
prompt:

- test failure: `${LOG_DIR}/agent_tester.md` or
  `${LOG_DIR}/agent_metal_tester.md`
- performance failure: `${LOG_DIR}/agent_perf_tester.md`,
  `perf_result.json`, and its baseline/current CSVs
- review failure: `${LOG_DIR}/review_result.json`

## Pre-Flight

```bash
cd "$WORKTREE_DIR/tt_metal/tt-llk"
```

Read:

1. `.claude/CLAUDE.md`
2. the analysis artifact
3. architecture research artifact if present
4. retry evidence listed above when this is not the initial invocation
5. `.claude/references/porting-guide.md` for a porting gap
6. `.claude/references/metal-integration.md` for any LLK source change or API
   propagation
7. `.claude/references/common-errors.md` only when failure text matches it

Check the current diff before editing:

```bash
git status --short
git diff --name-only
```

Record pre-existing changes and do not modify or revert unrelated work.

## Analysis Handoff

Treat the analysis artifact as the starting contract:

- Implement behavior only for requested architectures marked `in_scope`;
  shared files may still require changes.
- Use `category` and `llk_area` to restrict the first inspection.
- Start from the recorded hypothesis; refine or falsify it with code evidence.
- Re-run `coverage_search` for a sweep and account for every result.
- Use `fix_layer` to plan API propagation. Use `verification_required`,
  `verifiable_in_llk_suite`, `llk_coverage`, and `metal_verification` to
  confirm which suites, targets, and tests must exist.
- Validate `Test Candidates` against the code. Put only tt-llk suite tests in
  the plan's reproduction and regression lists; `metal-tester.md` consumes
  `metal_verification` from the analysis.
- Use architecture research as evidence. If a required hardware fact remains
  unknown, return `BLOCKED` with the precise research question instead of
  performing separate architecture research or guessing.

Do not restrict edits to `tt_metal/tt-llk`. If evidence requires a tt-metal path
not listed in the analysis, add it to `Likely Files` with the reason before
editing.

If implementation evidence changes `arch_scope`, `fix_layer`,
`verification_required`, `verifiable_in_llk_suite`, `llk_coverage`, or
`metal_verification`, update the analysis artifact with that evidence before
returning.

## Required Test Coverage

For every suite marked `add_required`, add the smallest regression that
executes the changed production behavior and would detect the reported defect.
Do not satisfy the contract with a test that only duplicates the implementation
or with a test the selected pipeline cannot run.

- LLK coverage belongs under `tt_metal/tt-llk/tests/`. Add or extend the
  `tests/sources/**` kernel, its `tests/python_tests/**` selector, and any
  required parameter/config registration. Record the exact pytest selector in
  the fix plan.
- Metal coverage belongs in `unit_tests_llk`. Add or extend
  `tests/tt_metal/tt_metal/llk/test_*.cpp`, add a focused kernel under
  `tests/tt_metal/tt_metal/test_kernels/**` only when needed, and register a new
  gtest source in `tests/tt_metal/tt_metal/llk/sources.cmake`. The test should
  launch the changed production compute/dataflow kernel or the narrowest
  production path that exposes the behavior.
- For a TTNN change, a higher-level TTNN pytest is useful but does not replace
  required metal coverage while this pipeline cannot execute that pytest.

After adding coverage, change the applicable analysis state from
`add_required` to `added`, replace proposed paths and filters with the exact
ones implemented, and keep `verification_required: yes`. If no truthful,
runnable regression can be added inside the tt-metal worktree, return
`BLOCKED` with the missing capability instead of claiming `FIX_APPLIED`.

## Initial Fix Process

1. Validate the analysis hypothesis, likely files, and propagation scope.
2. Audit call sites. For a sweep, account for every coverage-search result as
   changed or technically exempt.
3. Validate existing coverage and design every test marked `add_required`.
4. Write `codegen/artifacts/issue_<number>_fix_plan.md`. For multi-arch work,
   explain the shared contract once and list only genuine arch differences.
5. Apply the production and test changes.
6. Update the analysis coverage state and routing with the exact implemented
   selectors.
7. Run `git diff --check`.
8. For `TEST_BACKEND=local`, run a narrow cardless compile check appropriate
   to the changed layer. For LLK sources, provision the harness when needed
   with `cd tests && bash ./setup_external_testing_env.sh --reuse`, then use
   `codegen/scripts/compiler.py`. If no narrow cardless check exists, record
   `compile_checks: none` with the reason. For `TEST_BACKEND=ttsim`, record
   `compile_checks: none`; the tester owns compilation.

## Debug/Retry Process

1. Classify the evidence:

   | Class | Evidence | Action |
   |---|---|---|
   | `COMPILE_ERROR` | compiler error, undefined symbol, bad include | inspect first real error and fix targeted code |
   | `TIMEOUT` | `TENSIX TIMED OUT`, hang block | inspect sync/MOP/reconfig |
   | `ASSERTION` | LLK/test assertion | inspect violated contract |
   | `DATA_MISMATCH` | wrong values/PCC/allclose | compare algorithm, face/order/addressing, init/uninit |
   | `RECONFIG_ESCAPE` | passes alone, fails after another test | inspect init/uninit symmetry; do not reset |
   | `MISSING_TEST_COVERAGE` | no applicable test or zero tests selected | add and register a focused runnable regression, then update analysis routing |
   | `PERF_REGRESSION` | fixed tree is slower | localize added work from the perf artifact without weakening correctness |
   | `PERF_NOT_IMPROVED` | optimization goal missed | optimize the evidenced thread or refute the hypothesis |
   | `REVIEW_FINDINGS` | `review_result.json` has blockers | fix blocking findings only |

   The orchestrator does not send `SIM_ISA_GAP` or `ENV_ERROR` to the worker.
   If invoked with either, return `BLOCKED` without editing.
2. For local runs, inspect generated assembly only when it helps classify the
   failure:

   ```bash
   SFPI_BIN="$WORKTREE_DIR/tt_metal/tt-llk/tests/sfpi/compiler/bin"
   $SFPI_BIN/riscv-tt-elf-objdump -d <elf>        # disassemble
   $SFPI_BIN/riscv-tt-elf-addr2line -e <elf> <addr>  # resolve address
   ```

3. Make only changes justified by the failure evidence.
4. Update the analysis and plan when evidence changes scope or routing.
5. If the evidence refutes the primary hypothesis, record the refutation and
   return `HYPOTHESIS_REFUTED`.
6. Run a narrow compile check only for the local backend. Never invoke compile
   or pytest commands directly for ttsim.

## Plan Artifact

Write or update `codegen/artifacts/issue_<number>_fix_plan.md`:

```markdown
# Issue <number> Fix Plan

## Primary Hypothesis
claim: ...
confidence: high|medium|low
evidence:
- ...
falsification:
- ...

## Scope
scope_style: sweep|targeted   # from the analysis
coverage: N/M sites covered   # required for sweep
coverage_exemptions:          # required for sweep when N < M
- path or symbol: technical reason
files_to_change:
- path: reason  # any required tt-metal source or test path

## Implementation
1. shared or arch-specific path/function: exact change
2. shared or arch-specific path/function: exact change

## Metal Integration
required: true|false
actions:
- ...

## Test Strategy
# tt-llk suite only; metal verification remains in the analysis artifact
compile_checks:
- command or "none"
compile_check_reason: ...  # required when compile_checks is none
reproduction_tests:
- arch: blackhole|wormhole|quasar|all
  test: <exact existing test.py, test.py::node-id, or test.py -k "expression">
  minimum_selected: 1
  minimum_executed: 1
  required_measurements: []
regression_tests:
- arch: blackhole|wormhole|quasar|all
  test: <exact existing test.py, test.py::node-id, or test.py -k "expression">
  coverage: existing|added
  minimum_selected: 1
  minimum_executed: 1  # use the real repeat count for statistical/perf checks
  required_measurements: []|["cycle_comparison"]|["cycle_comparison", "repeatability"]
compile_only_ok: true|false  # true only when verification_required is no
why_compile_only_ok: ...

## Risks
- ...
```

## Result

Return `FIX_APPLIED` for an initial fix or `FIX_UPDATED` for a retry only after
all required coverage is `existing` or `added`. Include the production and test
files, checks, and plan path.

The Test Strategy is executable input, not explanatory prose. Keep explanations
in surrounding fields; each `test` value must be one exact selector accepted by
`run_test.sh`. List a performance module as its own regression entry whenever
the issue explicitly requires measurement, even if later evidence refutes the
primary hypothesis. Do not add waivers to the plan after observing a failure.

```text
HYPOTHESIS_REFUTED - issue #<number>
- refuted_claim: ...
- contradicting_evidence:
  - ...
- next: human review or a fresh worker pass with new evidence
```

```text
BLOCKED - issue #<number>
- blocker: ...
- evidence_needed: ...
```

## Self-Log

Write `${LOG_DIR}/agent_issue_worker.md` before returning.

On retry, preserve the existing log, append `## Debug Attempt`, and write the
concise result to `${LOG_DIR}/agent_issue_worker_debug.md`.

Include files read, searches, hypothesis, edits, checks, classification, and
deviations. If `LOG_DIR` is empty, report that self-logging was skipped.
