---
name: issue-solver-orchestrator
description: "Coordinate one LLK-related tt-metal issue fix for one architecture."
model: sonnet
tools: Read, Bash, Grep, Agent
---

# LLK Issue Solver Orchestrator (single-arch)

Fix one issue for one `TARGET_ARCH`. All state changes and dashboard mechanics
live in
`codegen/scripts/issue_solver/orchestrator_steps.sh` (shared with the multi-arch
orchestrator under `RUN_MODE=single`). Use this file only for control flow.
Source the library once per Bash call; do not reproduce or edit its functions.
Run the applicable Bash blocks in order.

## Input & State

The router provides `WORKTREE_DIR` and writes bootstrap state. `setup_run`
copies run metadata to `$LOG_DIR/state.json`; `TTSIM_SO_PATH` remains in
bootstrap state. Read and write run state through the sourced helpers because
shell variables do not persist between Bash calls.

Bootstrap state schema:

- `RUN_MODE=single`
- `TARGET_ARCH`
- `ISSUE_NUMBER`, `ISSUE_TITLE`, `ISSUE_BODY`, `ISSUE_LABELS`,
  `ISSUE_COMMENTS`, `ISSUE_URL`
- `WORKTREE_BRANCH`, `TEST_BACKEND`
- `TTSIM_SO_PATH` when `TEST_BACKEND=ttsim`
- `CREATE_LOCAL_BRANCH`, `CREATE_PR`

Code writes may touch any path inside `$WORKTREE_DIR` when the analysis and
repository evidence show it is required. Do not edit dashboard or codegen
implementation; required artifacts and self-logs are allowed. Editing outside
the tt-metal worktree is a scope violation.

## Git Policy

Do not run git mutations directly. Only
`execute_step_write_generated_patch` may create the final local commit and
patch. Never push, open a PR, checkout, or reset. Leaf agents follow their own
Git policies; in particular, `perf-tester.md` may add and remove only its
temporary detached baseline worktree.

## Agent and Result Conventions

- Spawn one agent at a time and wait for it.
- Give every agent `WORKTREE_DIR`; give `perf-tester.md` the one architecture
  it must measure. Agents resolve other inputs from state.
- Expand prompt placeholders before spawning. The Agent tool does not expand
  shell variables.
- Follow each leaf playbook instead of repeating its implementation here.
- Use the authoritative result for each stage:

  | Stage | Authoritative result |
  |---|---|
  | analyze / research | issue artifact |
  | fix / retry | worker's final marker and fix plan |
  | functional test | `run.json` metrics plus tester result |
  | review | `review_result.json` |
  | performance | `perf_result.json` |

After every `FIX_APPLIED` or `FIX_UPDATED`, route verification again and record
the full worktree diff. That worker edit invalidates all later evidence:
functional verification, review, and performance must run again in that order.

## Stop Conditions

On `NO SPACE LEFT ON DEVICE`, spawn nothing else. Run
`execute_step_report_no_space "<current step>"` and end the run.

Do not send these outcomes to the worker:

- `ENV_ERROR`: the test environment is unusable.
- `SIM_ISA_GAP`: the selected simulator cannot execute the test.
- `PERF_ENV_ERROR` or `PERF_NOT_APPLICABLE`: performance was not comparable or
  does not apply.

Record their evidence and follow the outcome rules below.

## Pipeline

```text
analyze → [research] → fix → functional verification → review → performance → finalize
                          ↑__________________________________________|
                                     any worker edit
```

## 1. Setup

From `$WORKTREE_DIR/tt_metal/tt-llk`:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_validate_input "$WORKTREE_DIR"
execute_step_validate_env
execute_step_setup_run
execute_step_write_initial_run_json
```

Stop on an input rejection. Environment validation is advisory unless a later
stage proves the missing prerequisite is required.

## 2. Analyze and Research

Spawn `issue-analyzer.md` once. It owns scope, architecture classification,
verification routing, perf intent, and research questions. Then run:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_refine_perf_goal
```

Read `in_scope` from the analysis artifact. If false, run:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_finalize_out_of_scope
```

Then stop. Do not spawn another agent or enter any later pipeline stage.

If `needs_arch_research: true`, run `execute_step_advance_arch_lookup`, then
spawn `arch-lookup.md` once. Otherwise leave `PREVIOUS_AGENT=analyzer`.

## 3. Apply the Fix

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_writer
```

Spawn `issue-worker.md` in initial-fix mode.

- `FIX_APPLIED`: continue.
- `BLOCKED`: store the reported reason in `OBSTACLE`, mark the run failed, and
  finalize without verification.
- `HYPOTHESIS_REFUTED`: first call
  `execute_step_route_verification hypothesis_refuted` so any
  explicitly planned performance requirement is sealed and remains auditable.
  Then store the reported reason in `OBSTACLE`, mark the run failed, and
  finalize without claiming that the requirement passed. A refutation is not a
  waiver and must not delete an explicit performance leaf.
- Any other or missing marker: treat it as an environment/orchestration error,
  not as an applied fix.

After a successful worker result:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_route_verification
execute_step_record_changed_files
```

`execute_step_route_verification` must complete before either tester is
advanced or spawned. It seals the checksummed manifest and writes its current
manifest/attempt IDs to run state. `VERIFY_ROUTE=missing` means normalization
rejected coverage, a path, or a selector; do not execute a test command.

If there is no fix-related diff, stop as blocked rather than reporting a
successful empty fix.

## 4. Functional Verification

Use `VERIFY_ROUTE`:

| Route | Action |
|---|---|
| `llk` | spawn `tester.md` |
| `metal` | spawn `metal-tester.md` |
| `both` | run `tester.md`, then `metal-tester.md`; retain both outcomes |
| `missing` | send `MISSING_TEST_COVERAGE` to the worker; do not test or review |
| `none` | run `execute_step_mark_unverifiable`; valid only when `verification_required: no` |

The analyzer and worker must leave every required suite at coverage
`existing` or `added`. If routing returns `missing`, consume one debug retry:

1. Call `execute_step_coverage_feedback` with the missing LLK/metal coverage
   and selector evidence printed by `execute_step_route_verification`.
2. Spawn `issue-worker.md` with
   `FAILURE_CLASS=MISSING_TEST_COVERAGE`.
3. On `FIX_UPDATED`, call `execute_step_bump_debug`, rerun route verification,
   and record changed files.

If the worker cannot add a truthful runnable regression or the retry budget is
exhausted, finalize failed. Never convert missing coverage to `none`.

Before a tester, call its advance helper:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_tester       # pass fix_tests after a retry
# or
execute_step_advance_metal_test
```

After an `llk`, `metal`, or `both` route finishes, combine its required suite
results and then aggregate the counters:

```bash
execute_step_combine_verification_results
execute_step_aggregate_results
```

For production runs, the combiner writes the compatibility verdict and counters
at `arch_results.<arch>` while preserving each tester's result under
`suite_results`. Audit runs instead ignore agent-authored summaries and reduce
the current manifest's structured leaves from
`${LOG_DIR}/verification-results/<attempt>/`. A missing, duplicate, malformed,
foreign, zero-count, identity-mismatched, artifact-mismatched, or incomplete
leaf cannot become `SUCCESS`. For `both`, the combined functional outcome is:

- failing if either suite fails;
- `SUCCESS` if at least one suite passes and the other is non-failing;
- otherwise `COMPILED_ONLY` or `UNVERIFIABLE_IN_LLK_SUITE`.

For `none`, call `execute_step_mark_unverifiable` and skip the combiner.
`none` means runtime verification is genuinely not applicable; it never means
that the repository lacked a test.

Handle each suite verdict as follows:

| Verdict | Action |
|---|---|
| `SUCCESS` | continue |
| `COMPILED_ONLY`, `UNVERIFIABLE_IN_LLK_SUITE` | continue with a compiled/unverified outcome |
| `COMPILE_FAILED`, `TESTS_FAILED` | enter the debug loop; `MISSING_TEST_COVERAGE` requires adding/registering a test |
| `ENV_ERROR`, `SIM_ISA_GAP` | record the evidence and finalize failed without a worker retry |
| `SKIPPED` | valid only for analyzer-owned out-of-scope work |

### Debug Loop

Retry only while `DEBUG_CYCLES < MAX_DEBUG_CYCLES`:

1. Call `execute_step_debug_feedback` with the first meaningful failure.
2. Spawn `issue-worker.md` with the concrete failure class and raw-log path.
   A missing selector or zero selected tests uses
   `FAILURE_CLASS=MISSING_TEST_COVERAGE`.
3. On `FIX_UPDATED`, rerun route verification and changed-file recording, then
   call `execute_step_bump_debug`.
4. Return to functional verification using the updated route.

`BLOCKED` or `HYPOTHESIS_REFUTED` ends the run failed with its evidence. If the
budget is exhausted while a repairable failure remains, call
`execute_step_mark_status failed` and finalize.

## 5. Review

Run review when a fix diff exists and functional verification has no terminal
failure. This includes `VERIFY_ROUTE=none`.

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_review
```

Spawn `reviewer.md`, then call `execute_step_record_review`. Read
`blocking_total` from `review_result.json`:

- `0`: continue to performance.
- Greater than zero with retry budget: call `execute_step_review_feedback`,
  spawn `issue-worker.md` with `FAILURE_CLASS=REVIEW_FINDINGS`, and then call
  `execute_step_bump_review`.
- Budget exhausted, `BLOCKED`, or `HYPOTHESIS_REFUTED`: preserve the functional
  outcome, set `OBSTACLE=unresolved_review_findings`, and stop retrying review.

After `FIX_UPDATED`, rerun route verification and changed-file recording, then
return to functional verification. Do not reuse the earlier review.

## 6. Performance

Run only after the current diff has completed the review loop.

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
PERF_ARCHES="$(execute_step_perf_arches)"
```

If empty, call `execute_step_perf_not_measured` and finalize. Otherwise call
`execute_step_advance_perf`, spawn `perf-tester.md` for `TARGET_ARCH`, and call
`execute_step_record_perf`.

| Outcome | Action |
|---|---|
| `PERF_OK` | finalize |
| `PERF_NOT_APPLICABLE`, `PERF_ENV_ERROR` | preserve the functional outcome; do not retry the worker |
| `PERF_TEST_FAILED` | retry the worker with its concrete compile/test/hang failure class |
| `PERF_REGRESSED` | retry with `FAILURE_CLASS=PERF_REGRESSION` |
| `PERF_NOT_IMPROVED` | retry with `FAILURE_CLASS=PERF_NOT_IMPROVED` when the goal is `improve` |

For a performance retry, call `execute_step_perf_feedback`, spawn the worker,
and then call `execute_step_bump_perf`. On `FIX_UPDATED`, rerun route
verification and changed-file recording, then return to functional
verification, review, and performance.

When the performance budget is exhausted:

- `PERF_TEST_FAILED` or a `no_regress` regression fails the run and records the
  obstacle.
- `PERF_NOT_IMPROVED` for an optimization preserves the functional outcome and
  remains visible in `perf_result.json`.

## 7. Finalize

This section is only for in-scope runs. Out-of-scope runs already returned
through `execute_step_finalize_out_of_scope`.

Choose the final functional verdict from the latest valid functional evidence:
`SUCCESS` for real passing verification, or `COMPILED_ONLY` /
`UNVERIFIABLE_IN_LLK_SUITE` only when runtime verification was explicitly not
applicable. Missing required coverage is a failure. A previously marked failure
remains failed.

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_deferred_message
execute_step_status_from_verdict "{final functional verdict}"
execute_step_write_generated_patch
execute_step_finalize_run
execute_step_copy_artifacts
```

On the audit lane, `execute_step_finalize_run` first performs the `all`-scope
reduction. It changes a requested success to failed when any sealed functional
or performance leaf is not successful. The final writer then requires the
reducer's success token and independently hashes the packaged worktree diff
against the verified patch digest. Do not create or patch that token manually.

If `OBSTACLE` is already nonempty, preserve it across
`execute_step_deferred_message`; that helper may clear an obstacle when
verification is deferred. After patch generation, verify that every
fix-related changed path is present in the local fix commit or
`generated.patch`. If any path is omitted, mark the run failed and report the
packaging gap instead of claiming success.

Return the summary from `$LOG_DIR/run.json`, including status, commits, patch,
changed files, functional evidence, review, performance, obstacle, and cost.
