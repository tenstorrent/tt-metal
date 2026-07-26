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

Read `in_scope` from the analysis artifact. If false, use final functional
verdict `SKIPPED` and finalize without spawning another agent.

If `needs_arch_research: true`, run `execute_step_advance_arch_lookup`, then
spawn `arch-lookup.md` once. Otherwise leave `PREVIOUS_AGENT=analyzer`.

## 3. Apply the Fix

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_writer
```

Spawn `issue-worker.md` in initial-fix mode.

- `FIX_APPLIED`: continue.
- `BLOCKED` or `HYPOTHESIS_REFUTED`: store the reported reason in `OBSTACLE`,
  mark the run failed, and finalize without verification.
- Any other or missing marker: treat it as an environment/orchestration error,
  not as an applied fix.

After a successful worker result:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_route_verification
execute_step_record_changed_files
```

If there is no fix-related diff, stop as blocked rather than reporting a
successful empty fix.

## 4. Functional Verification

Use `VERIFY_ROUTE`:

| Route | Action |
|---|---|
| `llk` | spawn `tester.md` |
| `metal` | spawn `metal-tester.md` |
| `both` | run `tester.md`, then `metal-tester.md`; retain both outcomes |
| `none` | run `execute_step_mark_unverifiable`; no functional agent |

Before a tester, call its advance helper:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_tester       # pass fix_tests after a retry
# or
execute_step_advance_metal_test
```

Run `execute_step_aggregate_results` after functional testing. For `both`,
never let the second suite hide an earlier failure. The combined functional
outcome is:

- failing if either suite fails;
- `SUCCESS` if at least one suite passes and the other is non-failing;
- otherwise `COMPILED_ONLY` or `UNVERIFIABLE_IN_LLK_SUITE`.

Handle each suite verdict as follows:

| Verdict | Action |
|---|---|
| `SUCCESS` | continue |
| `COMPILED_ONLY`, `UNVERIFIABLE_IN_LLK_SUITE` | continue with a compiled/unverified outcome |
| `COMPILE_FAILED`, `TESTS_FAILED` | enter the debug loop |
| `ENV_ERROR`, `SIM_ISA_GAP` | record the evidence and finalize failed without a worker retry |
| `SKIPPED` | valid only for analyzer-owned out-of-scope work |

### Debug Loop

Retry only while `DEBUG_CYCLES < MAX_DEBUG_CYCLES`:

1. Call `execute_step_debug_feedback` with the first meaningful failure.
2. Spawn `issue-worker.md` with the concrete failure class and raw-log path.
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

Choose the final functional verdict from the latest valid functional evidence:
`SKIPPED` for an out-of-scope issue, `SUCCESS` for real passing verification,
or `COMPILED_ONLY` / `UNVERIFIABLE_IN_LLK_SUITE` when no relevant in-harness
test exists. A previously marked failure remains failed.

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_deferred_message
execute_step_status_from_verdict "{final functional verdict}"
execute_step_write_generated_patch
execute_step_finalize_run
execute_step_copy_artifacts
```

If `OBSTACLE` is already nonempty, preserve it across
`execute_step_deferred_message`; that helper may clear an obstacle when
verification is deferred. After patch generation, verify that every
fix-related changed path is present in the local fix commit or
`generated.patch`. If any path is omitted, mark the run failed and report the
packaging gap instead of claiming success.

Return the summary from `$LOG_DIR/run.json`, including status, commits, patch,
changed files, functional evidence, review, performance, obstacle, and cost.
