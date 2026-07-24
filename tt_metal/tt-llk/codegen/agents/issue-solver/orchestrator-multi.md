---
name: issue-solver-orchestrator-multi
description: "Single-run multi-arch LLK issue-solver. Creates one dashboard run, one shared fix, and per-arch test results."
model: sonnet
tools: Read, Write, Edit, Bash, Glob, Grep, Agent
---

# Multi-Arch LLK Issue Solver

Fixes one GitHub issue across multiple LLK architectures as one coordinated run.
All mechanics live as `execute_step_*` functions in
`codegen/scripts/issue_solver/orchestrator_steps.sh`; this playbook is control
flow only. Source the library once per Bash call, pass per-run values as
arguments, and never hand-edit or read the function bodies:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_setup_run
```

Contract:

- One run under `${CODEGEN_LOGS_ROOT}/issue_solver` (resolved by `setup_run`).
- One analyzer, one shared writer, one tester session; per-arch progress in the
  run's `arch_results`. Do not spawn `orchestrator.md` per arch.
- Do not push or create PRs. `finalize` makes one **local** commit only.

**EVERY BASH BLOCK BELOW MUST BE EXECUTED IN ORDER, except blocks that only
illustrate an Agent prompt.**

## Input & State

The router seeds the run inputs into the worktree state file and tells you only
`WORKTREE_DIR`. Bootstrap keys (worktree file): `RUN_MODE=multi`, `TARGET_ARCHES`,
`ISSUE_NUMBER/TITLE/BODY/LABELS/COMMENTS/URL`, `WORKTREE_BRANCH`, `TEST_BACKEND`,
`TTSIM_SO_PATHS`, `CREATE_LOCAL_BRANCH`, `CREATE_PR`. `setup_run` hydrates the
rest into `$LOG_DIR/state.json`. Every step reads/writes state through the
library — never via `export` (env vars do not survive between Bash calls).

Code writes must stay inside `$WORKTREE_DIR/tt_metal/tt-llk`,
`.../tt_metal/hw/ckernels`, `.../tt_metal/hw/inc/api`, `.../ttnn/cpp/ttnn/operations`,
or `.../tests/tt_metal`. Editing elsewhere is a scope violation.

## Git Policy

Read-only git only (`status/diff/show/log/rev-parse`) in this orchestrator and
all subagents. The single local commit + `generated.patch` is done by
`execute_step_write_generated_patch` at finalize. No push/PR/checkout/reset.

## Agent I/O conventions

- An agent's status/report is its **final message** — read it from the tool
  result, not a file.
- Spawn every agent **synchronously in the foreground**; never set
  `run_in_background`; never end your turn while an agent is running.
- **Prompt placeholders you fill yourself** before spawning (the Agent tool does
  not expand `${...}`). Fill `${ISSUE_NUMBER}` etc. from state (`sg KEY` after
  sourcing the library). Every subagent resolves its own inputs from state:
  ```
  WT="$(cd ../.. && pwd)"; LOG_DIR="$(python codegen/scripts/state.py --worktree-dir "$WT" get LOG_DIR)"
  python codegen/scripts/state.py --log-dir "$LOG_DIR" get <KEY>
  ```

## Out of Space

If any step prints the `NO SPACE LEFT ON DEVICE` banner: stop, spawn nothing
more, run `execute_step_report_no_space "<current step>"`, report the run failed
(no space), and end.

## Pipeline

```
analyzer → [arch_lookup?] → writer → {tester | metal_test} → [debug loop] → review → perf → finalize
```

Playbooks: `codegen/agents/issue-solver/{issue-analyzer,arch-lookup,issue-worker,tester,metal-tester,reviewer,perf-tester}.md`.

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
    Resolve LOG_DIR from the worktree state file, then read TARGET_ARCHES and
    ISSUE_* from state ($LOG_DIR/state.json). Write
    codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md and ${LOG_DIR}/agent_issue_analyzer.md.
```

Then:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_refine_perf_goal                   # perf_intent → PERF_GOAL (over the Step 0 keyword guess)
```

If the analyzer declares the issue out of scope for **every** arch: run Step 6
with `execute_step_mark_status skipped`, then finalize. If only some arches are
out of scope, keep the run alive (those arches carry `SKIPPED` in `arch_results`).

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

Spawn `arch-lookup.md` (reads inputs from state; writes
`codegen/artifacts/issue_${ISSUE_NUMBER}_arch_research.md` + its self-log). If not
needed, skip — `PREVIOUS_AGENT` stays `analyzer`.

## Step 3: Fix

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_writer                     # uses PREVIOUS_AGENT
```

Spawn `issue-worker.md` once: one shared multi-arch fix across `TARGET_ARCHES`,
arch-specific code only where the LLK structure requires it. It writes the fix
plan + self-log. Then:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_record_changed_files
```

## Step 4: Test

Branch on `VERIFY_ROUTE`:

- `none` → `execute_step_mark_unverifiable`, then Step 5.3.
- `llk`/`both` → run the tt-llk tester (below).
- `metal` → skip to Step 4b.

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_tester                     # (pass "fix_tests" as arg on a debug re-test)
```

Spawn `tester.md` once (reads `TARGET_ARCHES`, `TEST_BACKEND`, `TTSIM_SO_PATHS`,
fix plan, changed files from state). It runs each arch sequentially and writes
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

Spawn `metal-tester.md` once (reads `METAL_TARGET/FILTER/DISPATCH`, backend,
changed files from state; build provisioning `CODEGEN_METAL_VERIFY_HOME/BUILD_DIR`
and silicon `HW_TEST_DISPATCH_CMD/HW_TEST_SESSION` from env). It writes per-arch
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

Re-run the suite that failed (Step 4 for tt-llk, Step 4b for metal) with
`execute_step_advance_tester fix_tests`, then `execute_step_aggregate_results`.
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
re-run Step 4 for affected arches, and if still green re-run Step 5.3. Worker
`HYPOTHESIS_REFUTED` → stop, go to Step 5.5. Budget exhausted with blockers:
`execute_step_message` note + `OBSTACLE=unresolved_review_findings` (keep the
functional status) — set it via `execute_step_mark_status` only if you were going
to fail anyway; otherwise leave status to Step 6. Proceed to Step 5.5.

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
`execute_step_bump_perf`, re-run Step 4 then Step 5.5 for those arches. Exhausted:
`PERF_GOAL=no_regress` + still regressed → `execute_step_mark_status failed` with
`OBSTACLE=perf_regression`; `PERF_GOAL=improve` + not improved → keep the
functional status (perf verdict stays `not_improved` in run.json).

## Step 6: Finalize

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_deferred_message         # no-op unless VERIFY_DEFERRED=1
execute_step_combined_status          # reads arch_results from run.json → COMBINED_STATUS + STATUS (skips if already failed)
execute_step_write_generated_patch    # local commit (no push) + generated.patch
execute_step_finalize_run             # run.json finalize + authoritative cost refresh
execute_step_copy_artifacts           # runs.jsonl + artifact/base-file snapshots
```

Then return the run summary (`status`, `combined_status`, `run_id`, `log_dir`,
`branch`, `base_commit`, `fix_commit`, `worktree_dir`, `patch`, `target_arches`,
per-arch `arch_results` incl. verdict/tests/perf, `review`, cost, `create_pr_requested`,
`obstacle`) — read every field from `$LOG_DIR/run.json`.
