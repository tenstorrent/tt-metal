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

- Spawn one agent at a time. The spawn call blocks and returns that child's
  result, so that return is the entire wait: a completion notification means the
  result is ready to read now. When a result is missing, read its authoritative
  artifact once and act on what is there.
- Expand this delegation template before spawning:

  ```text
  Read and follow {WORKTREE_DIR}/tt_metal/tt-llk/codegen/agents/issue-solver/{role}.md.
  WORKTREE_DIR={WORKTREE_DIR}
  ```

- Spawn the reviewer as subagent type `issue-solver-reviewer`, which runs
  review at `xhigh` effort; spawn every other role as a general subagent.
- The child reads its playbook, state (including verbatim issue text), and
  artifacts. Do not pre-read its playbook or copy these inputs into the prompt.
- Append only needed perf `TARGET_ARCH`, retry `FAILURE_CLASS`, evidence paths,
  scoped questions, or explicit user/runner constraints unavailable in state/artifacts.
- Do not turn model assumptions or prior-memory advice into hard rules.
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

## 2. Solve: analyse, research and apply the fix in one session

Measured on this corpus: run duration tracks the number of agent stages at
roughly half an hour each, and analysis, architecture research and the initial
fix shared almost all of their discovery while paying that cost three times
over. They now run as **one** agent session.

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_solve
```

Spawn a single agent with both contracts, in this order, in one session:

```text
Read and follow {WORKTREE_DIR}/tt_metal/tt-llk/codegen/agents/issue-solver/issue-analyzer.md
to produce the analysis artifact. Then, in this same session, read and follow
{WORKTREE_DIR}/tt_metal/tt-llk/codegen/agents/issue-solver/issue-worker.md
in initial-fix mode to apply the fix it calls for.
WORKTREE_DIR={WORKTREE_DIR}
```

Both playbooks keep their own contracts, artifacts and result markers. What
goes away is the boundary between them: one orientation instead of three, no
analysis artifact written only to be re-read and re-derived by the next agent,
and no separate spawn for architecture research.

Handle the analyzer half's outcome before the fix half's:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_refine_perf_goal
```

Read `in_scope` from the analysis artifact. If false, run
`execute_step_finalize_out_of_scope` and stop. Do not enter any later stage.
The session must decide scope before it edits, so an out-of-scope issue still
costs only the analysis.

`needs_arch_research: true` is answered inside the same session, against the
repository and the architecture references, and recorded in the analysis
artifact exactly as `arch-lookup.md` would. Spawn `arch-lookup.md` separately
only when the session returns `BLOCKED` with a hardware question it could not
settle from the repository; that is an escalation, not a routine stage.

Then take the fix half's marker.

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

`execute_step_route_verification` must complete before any tester is
advanced or spawned. It seals the checksummed manifest and writes its current
manifest/attempt IDs to run state. `VERIFY_ROUTE=missing` means normalization
rejected coverage, a path, or a selector; do not execute a test command.

If there is no fix-related diff, stop as blocked rather than reporting a
successful empty fix.

## 4. Functional Verification

When `run.json.functional_executor` is `sealed-llk-v1` (enabled at initialization),
call `execute_step_run_sealed_functional` after routing/sealing and before
spawning a tester. It supports audit silicon leaves for every verification
suite -- llk, metal and ttnn -- and explicitly marked LLK host leaves. Exit 20
means no leaf executed: follow the normal tester route below. Exit 0 means every
sealed functional leaf passed; skip every tester the route selected, not just
the LLK one, and continue the existing combiner, review and performance gates.
Any other
exit preserves partial evidence in `run.json.functional_execution` and raw
leaf logs: diagnose that failure through the existing retry path; do not
resubmit successful or unresolved jobs to obtain a narrative summary. A changed
candidate needs a new sealed attempt and all its required evidence.

`VERIFY_ROUTE` is a canonical `+`-separated subset of `llk`, `metal`, and
`ttnn`. Run every named suite, in that order:

| Route membership | Action |
|---|---|
| contains `llk` | spawn `tester.md` |
| contains `metal` | spawn `metal-tester.md` |
| contains `ttnn` | spawn `ttnn-tester.md` |
| `missing` | send `MISSING_TEST_COVERAGE` to the worker; do not test or review |
| `none` | run `execute_step_mark_unverifiable`; valid only when `verification_required: no` |

For example, `llk+ttnn` runs the Layer-1 and end-to-end suites, while
`llk+metal+ttnn` runs all three. Never stop after the first successful suite.

Do stop at the first *failing* one. When a suite returns `COMPILE_FAILED`,
`TESTS_FAILED`, `ENV_ERROR` or `SIM_ISA_GAP`, its advance helper for the next
suite exits 21 and prints `SUITE_ROUTE_SHORT_CIRCUIT <suite>` with the blocking
`<arch>/<suite>: <verdict>` lines. Treat exit 21 as that failure already routed:
go straight to the outcome rules for the printed verdict — repair, or a stop
condition — and do not run the remaining suites or retry the advance. The
repair re-runs the whole route, so the skipped suites lose no coverage.
On the `audit` pool the helper never short-circuits: its retry classifier needs
a sealed receipt for every required leaf, so run the whole route there.

The analyzer and worker must leave every required suite at coverage
`existing` or `added`. If routing returns `missing`, consume one debug retry:

1. Call `execute_step_coverage_feedback` with the missing LLK/Metal/TTNN coverage
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
# or
execute_step_advance_ttnn_test
```

After every suite named by a functional route finishes, combine its required
suite results and then aggregate the counters:

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
leaf cannot become `SUCCESS`. The combined functional outcome is `SUCCESS`
only when every required suite has a terminal, nonzero, fully passing result;
any failing or malformed required suite fails the architecture.

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
   For a compiler failure before execution, pass optional second argument
   `COMPILE_FAILED` with the raw compiler log; this preserves caller diagnosis
   without claiming an execution receipt or waiving final verification.
   Stop if it rejects the retry. With a current reduction it records
   `FAILURE_CLASS` and `VERIFICATION_RETRY_CONTEXT` in state; do not replace
   those reasons with the compatibility verdict `TESTS_FAILED`.
2. Spawn `issue-worker.md` with that class and the evidence paths. Pure coverage
   errors use `VERIFICATION_PLAN_ERROR`; zero selected tests use
   `MISSING_TEST_COVERAGE`. Actual candidate failures retain their raw evidence,
   including mixed failures. Legacy runs use the concrete failure/log class.
3. On `FIX_UPDATED`, rerun route verification and changed-file recording, then
   call `execute_step_bump_debug`.
4. Return to functional verification using the updated route.

`BLOCKED` or `HYPOTHESIS_REFUTED` ends the run failed with its evidence. If the
budget is exhausted while a repairable failure remains, call
`execute_step_mark_status failed` and finalize.

## 4a. Sealed measurement, before review

Run this immediately after the functional gate is green and before spawning the
reviewer. It is opt-in and applies only when every sealed perf leaf is a
predeclared current-only `cycle_measurement`:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_run_sealed_measurement; rc=$?
```

| `rc` | Meaning | Action |
|---|---|---|
| `0` | Every measurement leaf measured and bound to this candidate | continue to review; section 6 will skip the perf tester |
| `20` | Unsupported before any submission, or a stale record for another candidate | continue to review; section 6 runs the existing perf tester unchanged |
| other | Hardware or evidence failure, evidence preserved | continue to review; section 6 owns the retry through its existing classes |

This never grants success, reduces all requirements, or replaces review. A
non-zero result is not a functional failure and must not be turned into one: the
functional verdict already stands on its own evidence. Do not skip review
because a measurement succeeded, and do not skip the measurement because review
is pending — they are independent reads of the same frozen candidate.

## 5. Review

Run review when a fix diff exists and functional verification has no terminal
failure. This includes `VERIFY_ROUTE=none`.

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_review
```

Spawn `reviewer.md` as subagent type `issue-solver-reviewer`, then call
`execute_step_record_review`. Read
`blocking_total` from `review_result.json` only after recording succeeds.
If validation fails, have the reviewer correct its output; never forward a
malformed result to the worker. A changed candidate requires
`execute_step_advance_review` and a full review again. For `unresolved` evidence,
spend one review retry (`execute_step_bump_review`) on the existing
architecture/test owner, then re-review. If the budget is exhausted or evidence
is unavailable, mark failed with the evidence gap. Do not interpret
uncertainty as a code-change request or continue to success.

For validated findings:

- `0`: continue to performance.
- Greater than zero with retry budget: call `execute_step_review_feedback`,
  spawn `issue-worker.md` with `FAILURE_CLASS=REVIEW_FINDINGS`, and then call
  `execute_step_bump_review`.
- Budget exhausted, `BLOCKED`, or `HYPOTHESIS_REFUTED`: preserve the functional
  evidence, set `OBSTACLE=unresolved_review_findings`, mark the run failed, and
  stop retrying review. Passing tests do not override incomplete requirements
  or other blocking findings.

For issue solves, review must also report `requirements_complete: true` before
the run can succeed. A false or missing value is not a clean completion review;
return to the existing review/worker loop with the missing requirement evidence.
If the reviewer omitted the check, have it finish the original-issue comparison
instead of sending an unspecified code change to the worker.
An explicit null is allowed only when the reviewer identifies a required
measurement sealed for the upcoming performance stage. Continue to performance
and finish the completion review at finalization; null cannot authorize success.

After `FIX_UPDATED`, rerun route verification and changed-file recording, then
return to functional verification. Do not reuse the earlier review.

## 6. Performance

A comparison runs only after the current diff has completed the review loop,
because a changed diff invalidates the measurement it would compare.

A **sealed current-only measurement is different**: it takes one worktree, no
baseline, and its verdict comes from a deterministic evaluator rather than an
agent. Waiting for review to finish before measuring it adds the entire review
duration to the critical path for no evidentiary gain, so section 4a measures it
as soon as the functional gate is green. When that already succeeded for exactly
this candidate, do not spawn a perf tester for it:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
[ "$(execute_step_sealed_measurement_done)" = 1 ] && echo "measurement already bound"
PERF_ARCHES="$(execute_step_perf_arches)"
```

If the measurement is already bound to this candidate, continue to finalize; its
evidence is in `run.json` under `perf` and the all-scope reduction validates it.

Otherwise, if `PERF_ARCHES` is empty, call `execute_step_perf_not_measured` and
finalize. Otherwise call `execute_step_advance_perf`, spawn `perf-tester.md` for
`TARGET_ARCH`, and call `execute_step_record_perf`.

| Outcome | Action |
|---|---|
| `PERF_OK` | finalize |
| `PERF_NOT_APPLICABLE`, `PERF_ENV_ERROR` | preserve the functional outcome; do not retry the worker |
| `PERF_PLAN_ERROR` | retry with `FAILURE_CLASS=PERF_PLAN_ERROR` to correct selectors or measurements and reseal |
| `PERF_TEST_FAILED` | retry the worker with its concrete compile/test/hang failure class |
| `PERF_REGRESSED` | retry with `FAILURE_CLASS=PERF_REGRESSION` |
| `PERF_NOT_IMPROVED` | retry with `FAILURE_CLASS=PERF_NOT_IMPROVED` when the goal is `improve` |

For a performance retry, call `execute_step_perf_feedback`, spawn the worker,
and then call `execute_step_bump_perf`. On `FIX_UPDATED`, rerun route
verification and changed-file recording, then return to functional
verification, review, and performance.

When the performance budget is exhausted:

- `PERF_PLAN_ERROR`, `PERF_TEST_FAILED`, or a `no_regress` regression fails the
  run and records the concrete planning or measurement obstacle.
- `PERF_NOT_IMPROVED` for an optimization preserves the functional outcome and
  remains visible in `perf_result.json`.

## 7. Finalize

This section is only for in-scope runs. Out-of-scope runs already returned
through `execute_step_finalize_out_of_scope`.

If the run has not already failed and review deferred `requirements_complete`
as null for scheduled performance evidence, invoke the same reviewer to finish
only the outstanding requirement comparison against the recorded perf results,
then call `execute_step_record_review`. Do not repeat its full diff review.
If requirements remain incomplete or blocked, mark the run failed and retain
the completed work and remaining requirement IDs in the final message.

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
Name completed and remaining requirement IDs. Never describe a subset as the
whole issue solved.
