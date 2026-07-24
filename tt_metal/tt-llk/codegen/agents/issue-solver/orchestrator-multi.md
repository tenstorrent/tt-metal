---
name: issue-solver-orchestrator-multi
description: "Run one coordinated LLK issue-solver pipeline across multiple architectures."
model: sonnet
tools: Read, Write, Edit, Bash, Glob, Grep, Agent
---

# Multi-Arch LLK Issue Solver

Fix one issue across multiple architectures in one run. State changes and
dashboard mechanics live in
`codegen/scripts/issue_solver/orchestrator_steps.sh`; this playbook is control
flow only. Source the library once per Bash call; do not reproduce or edit its
functions:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_setup_run
```

- One run under `${CODEGEN_LOGS_ROOT}/issue_solver` (resolved by `setup_run`).
- One analyzer, one shared worker, and one session per test stage.
- Store per-arch progress in `arch_results`.
- Do not spawn the single-arch orchestrator.

Run Bash blocks in order. Agent prompt blocks are templates, not commands.

## Input & State

The router provides `WORKTREE_DIR` and writes bootstrap state. `setup_run`
normalizes `TARGET_ARCHES` into run-state `TARGET_ARCHES_JSON` and copies run
metadata to `$LOG_DIR/state.json`; `TTSIM_SO_PATHS` remains in bootstrap state.
Use the sourced helpers because shell variables do not persist between calls.

Bootstrap state schema:

- `RUN_MODE=multi`
- `TARGET_ARCHES`
- `ISSUE_NUMBER`, `ISSUE_TITLE`, `ISSUE_BODY`, `ISSUE_LABELS`,
  `ISSUE_COMMENTS`, `ISSUE_URL`
- `WORKTREE_BRANCH`, `TEST_BACKEND`
- `TTSIM_SO_PATHS` when `TEST_BACKEND=ttsim`
- `CREATE_LOCAL_BRANCH`, `CREATE_PR`

Code writes must stay inside `$WORKTREE_DIR/tt_metal/tt-llk`,
`.../tt_metal/hw/ckernels`, `.../tt_metal/hw/inc/api`, `.../ttnn/cpp/ttnn/operations`,
or `.../tests/tt_metal`. Editing elsewhere is a scope violation.

## Git Policy

Do not run git mutations directly. Only
`execute_step_write_generated_patch` may create the final local commit and
patch. Never push, open a PR, checkout, or reset.

## Agent I/O conventions

- Read each agent's status from its final tool result.
- Spawn agents synchronously and wait for them.
- Expand prompt placeholders with `sg`; the Agent tool does not expand shell
  variables. Agents resolve their remaining inputs from state:
  ```
  WT="$(cd ../.. && pwd)"; LOG_DIR="$(python codegen/scripts/state.py --worktree-dir "$WT" get LOG_DIR)"
  python codegen/scripts/state.py --log-dir "$LOG_DIR" get <KEY>
  ```

## Out of Space

On `NO SPACE LEFT ON DEVICE`, spawn nothing else. Run
`execute_step_report_no_space "<current step>"` and end the run as failed.

## Pipeline

```
analyzer → [arch_lookup?] → writer → {tester | metal_test} → [debug loop] → review → perf → finalize
```

## Step 0: Setup

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_validate_input "{worktree_dir}"   # prints OK: … or REJECT: … (stop on REJECT)
execute_step_validate_env
execute_step_setup_run                          # identity, dirs, arch profiles, session; seeds state
execute_step_write_initial_run_json             # run.json init + pending arch_results + first cost snapshot
```

## Step 1: Analyze

Spawn the analyzer once for the full target list:

```text
Agent: subagent_type=general-purpose, description="Analyze multi-arch issue #${ISSUE_NUMBER}"
  prompt: |
    Read and follow codegen/agents/issue-solver/issue-analyzer.md.
    Resolve LOG_DIR from the worktree state file, then read TARGET_ARCHES_JSON and
    ISSUE_* from state ($LOG_DIR/state.json). Write
    codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md and ${LOG_DIR}/agent_issue_analyzer.md.
```

Then:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_refine_perf_goal                   # perf_intent → PERF_GOAL (over the Step 0 keyword guess)
```

For each architecture marked `out_of_scope`, patch its result before
continuing:

```bash
LOG_DIR="$(python codegen/scripts/state.py --worktree-dir "$WORKTREE_DIR" get LOG_DIR)"
python codegen/scripts/run_json_writer.py metric \
  --log-dir "$LOG_DIR" \
  --patch-json "{\"arch_results\":{\"${arch}\":{\"status\":\"done\",\"verdict\":\"SKIPPED\",\"tests_total\":0,\"tests_passed\":0,\"obstacle\":null}}}"
```

If every architecture is out of scope, go directly to Step 6;
`execute_step_combined_status` will derive `skipped`. Otherwise continue with
the in-scope architectures.

## Step 1.5: Route Verification

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_route_verification                 # sets VERIFY_ROUTE=llk|metal|both|none + metal target/filter/dispatch
```

Route → Step 4 behaviour → terminal:
`llk` tt-llk suite → `success`; `metal` metal gtest → `success`; `both` both must
pass; `none` no in-harness test → `compiled`/Working (never `skipped`).

## Step 2: Research (only if analysis asks for arch facts)

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_arch_lookup                # sets PREVIOUS_AGENT=arch_lookup
```

Spawn `arch-lookup.md`; it reads questions from the analysis artifact and
writes its artifact and self-log. If skipped, `PREVIOUS_AGENT` remains
`analyzer`.

## Step 3: Fix

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_writer                     # uses PREVIOUS_AGENT
```

Spawn `issue-worker.md` once: one shared multi-arch fix across
`TARGET_ARCHES_JSON`,
arch-specific code only where the LLK structure requires it. It writes the fix
plan + self-log. Then:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_record_changed_files
```

## Step 4: Test

Branch on `VERIFY_ROUTE`:

- `none` → `execute_step_mark_unverifiable`, reapply `SKIPPED` to any
  out-of-scope architectures because that helper initializes all targets as
  unverifiable, then Step 5.3.
- `llk`/`both` → run the tt-llk tester (below).
- `metal` → skip to Step 4b.

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_tester                     # (pass "fix_tests" as arg on a debug re-test)
```

Spawn `tester.md` once (run state plus bootstrap `TTSIM_SO_PATHS`). It runs
each architecture sequentially and writes
per-arch `arch_results` verdicts (`SUCCESS`/`COMPILE_FAILED`/`TESTS_FAILED`/
`SIM_ISA_GAP`/`ENV_ERROR`/`COMPILED_ONLY`/`SKIPPED`) into run.json via `metric`.
Then roll up counters:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_aggregate_results
```

For `both`, continue to Step 4b. For `llk`, go to Step 5.

## Step 4b: Metal Suite (VERIFY_ROUTE = metal | both)

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_metal_test
```

Spawn `metal-tester.md` once (run state, bootstrap simulator paths, and
environment provisioning described by that playbook). It writes per-arch
`arch_results` (`SUCCESS`/`COMPILE_FAILED`/`TESTS_FAILED`/`SIM_ISA_GAP`/`ENV_ERROR`/
`UNVERIFIABLE_IN_LLK_SUITE`); for `both` an arch is green only if neither suite
failed. Then `execute_step_aggregate_results`.

## Step 5: Debug loop (any arch COMPILE_FAILED/TESTS_FAILED)

While any in-scope arch is red and `DEBUG_CYCLES < MAX_DEBUG_CYCLES`:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_debug_feedback "{FAILURE_SUMMARY}"   # records failure + advances to fix_tests
```

Spawn `issue-worker.md` in debug/retry mode (combined tester evidence). Then:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_bump_debug
```

Re-run the failed route: Step 4 with `execute_step_advance_tester fix_tests`
for tt-llk, Step 4b with `execute_step_advance_metal_test` for metal, or both
in order. Then aggregate results.
Terminate when: every arch green → Step 5.3; `DEBUG_CYCLES == MAX` with an arch
still red → Step 6 `execute_step_mark_status failed`; worker returns
`HYPOTHESIS_REFUTED` → Step 6 `execute_step_mark_status failed`. Never debug
`SIM_ISA_GAP` (simulator limit) — mark that arch failed and continue others.

## Step 5.3: Review loop

Run when a fix diff exists and no arch is unresolved-failed — after green tests,
**and also** when `VERIFY_ROUTE=none` (the only quality gate before tt-metal CI).

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_review
```

Spawn `reviewer.md` once (shared diff; cross-arch parity is in scope). Then:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_record_review                        # patches review_result.json into run.json
```

Read `blocking_total`. If `0` → Step 5.5. If `> 0` and `REVIEW_RETRIES < MAX_REVIEW_RETRIES`:
`execute_step_review_feedback "{summary}"`, spawn `issue-worker.md` with
`FAILURE_CLASS=REVIEW_FINDINGS` + `review_result.json`, `execute_step_bump_review`,
re-run the applicable functional route for affected arches, and if still green
re-run Step 5.3. Worker
`HYPOTHESIS_REFUTED` → stop, go to Step 5.5. When the review budget is
exhausted, emit a message, preserve functional status, and set:

```bash
LOG_DIR="$(python codegen/scripts/state.py --worktree-dir "$WORKTREE_DIR" get LOG_DIR)"
python codegen/scripts/state.py --log-dir "$LOG_DIR" set \
  OBSTACLE unresolved_review_findings
```

Then continue.

## Step 5.5: Perf loop (green arches only; local BH/WH only)

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
PERF_ARCHES="$(execute_step_perf_arches)"
```

If empty → `execute_step_perf_not_measured`, go to Step 6. Else:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_perf
```

For each arch in `PERF_ARCHES` whose functional verdict was `SUCCESS`: spawn
`perf-tester.md` for that single arch (`PERF_GOAL`, changed op, fix plan from
state); it writes `$LOG_DIR/perf_result.json`. Read it **immediately** after each
arch: `execute_step_record_perf "${arch}"`. A *miss* = `PERF_REGRESSED`, or
`PERF_NOT_IMPROVED` when `PERF_GOAL=improve`. If any miss and
`PERF_RETRIES < MAX_PERF_RETRIES`: `execute_step_perf_feedback "{summary}"`, spawn
`issue-worker.md` (`FAILURE_CLASS=PERF_REGRESSION|PERF_NOT_IMPROVED` + CSV paths),
`execute_step_bump_perf`, re-run the applicable functional route, then Step 5.5
for those arches. Exhausted:
`PERF_GOAL=no_regress` + still regressed → set `OBSTACLE=perf_regression` with
`state.py`, then run `execute_step_mark_status failed test_failure`;
`PERF_GOAL=improve` + not improved → keep the functional status (perf verdict
stays `not_improved` in run.json).

## Step 6: Finalize

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_deferred_message         # no-op unless VERIFY_DEFERRED=1
execute_step_combined_status          # reads arch_results from run.json → COMBINED_STATUS + STATUS (skips if already failed)
execute_step_write_generated_patch    # local commit (no push) + generated.patch
execute_step_finalize_run             # run.json finalize + authoritative cost refresh
execute_step_copy_artifacts           # runs.jsonl + artifact/base-file snapshots
```

Return the run summary (`status`, `combined_status`, `run_id`, `log_dir`,
`branch`, `base_commit`, `fix_commit`, `worktree_dir`, `patch`, `target_arches`,
per-arch `arch_results` incl. verdict/tests/perf, `review`, cost, `create_pr_requested`,
`obstacle`) — read every field from `$LOG_DIR/run.json`.
