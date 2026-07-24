---
name: issue-solver-orchestrator
description: "Run the single-architecture LLK issue-solver pipeline on local silicon or ttsim."
model: sonnet
tools: Read, Write, Edit, Bash, Glob, Grep, Agent, mcp__atlassian__search, mcp__atlassian__searchConfluenceUsingCql, mcp__atlassian__getConfluencePage, mcp__atlassian__getAccessibleAtlassianResources, mcp__deepwiki__ask_question, mcp__deepwiki__read_wiki_contents, mcp__deepwiki__read_wiki_structure
---

# LLK Issue Solver Orchestrator (single-arch)

Fix one issue for one `TARGET_ARCH`. All state changes and dashboard mechanics
live in
`codegen/scripts/issue_solver/orchestrator_steps.sh` (shared with the multi-arch
orchestrator under `RUN_MODE=single`). Use this file only for control flow.
Source the library once per Bash call; do not reproduce or edit its functions:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_setup_run
```

Run Bash blocks in order. Agent prompt blocks are templates, not commands.

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

Code writes must stay inside `$WORKTREE_DIR/tt_metal/tt-llk`,
`.../tt_metal/hw/ckernels`, `.../tt_metal/hw/inc/api`, `.../ttnn/cpp/ttnn/operations`,
or `.../tests/tt_metal`. Editing elsewhere is a scope violation.

## Git Policy

Do not run git mutations directly. Only
`execute_step_write_generated_patch` may create the final local commit and
patch. Never push, open a PR, checkout, or reset.

## Agent I/O conventions

- Read each agent's status from its final tool result.
- Spawn agents synchronously.
- Expand prompt placeholders with `sg`; the Agent tool does not expand shell
  variables. Agents resolve their remaining inputs from state.
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
execute_step_validate_input "{worktree_dir}"   # OK: … or REJECT: … (stop on REJECT)
execute_step_validate_env
execute_step_setup_run
execute_step_write_initial_run_json
```

## Step 1: Analyze

```text
Agent: subagent_type=general-purpose, description="Analyze ${TARGET_ARCH} issue #${ISSUE_NUMBER}"
  prompt: |
    Read and follow codegen/agents/issue-solver/issue-analyzer.md.
    Resolve LOG_DIR from the worktree state file, then read TARGET_ARCH and ISSUE_*
    from state. Write codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md and
    ${LOG_DIR}/agent_issue_analyzer.md.
```

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_refine_perf_goal
```

If the analyzer declares the issue out of scope: run Step 6 with
`execute_step_mark_status skipped`, then finalize.

## Step 1.5: Route Verification

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_route_verification                # VERIFY_ROUTE=llk|metal|both|none
```

`llk`→Step 4; `metal`→Step 4b only; `both`→Step 4 then 4b (green iff neither
failed); `none`→neither (`compiled`/Working, never `skipped`).

## Step 2: Research (only if analysis asks for arch facts)

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_arch_lookup
```

Spawn `arch-lookup.md`; it reads questions from the analysis artifact and
writes its artifact and self-log.
If not needed, skip — `PREVIOUS_AGENT` stays `analyzer`.

## Step 3: Fix

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_writer
```

Spawn `issue-worker.md` for the initial fix (reads inputs from state; writes fix
plan + self-log). Then:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_record_changed_files
```

## Step 4: Test

Branch on `VERIFY_ROUTE`: `none`→`execute_step_mark_unverifiable` then Step 5.3;
`metal`→Step 4b only; `llk`/`both`→below.

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_tester                    # (pass "fix_tests" on a debug re-test)
```

Spawn `tester.md` (run state plus bootstrap `TTSIM_SO_PATH`). It returns one
verdict: `SUCCESS`/`COMPILE_FAILED`/
`TESTS_FAILED`/`SIM_ISA_GAP`/`ENV_ERROR`/`COMPILED_ONLY`/`UNVERIFIABLE_IN_LLK_SUITE`
and writes counts via `metric`. Then:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_aggregate_results
```

`both`→Step 4b; `llk`→Step 5.

## Step 4b: Metal Suite (VERIFY_ROUTE = metal | both)

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_metal_test
```

Spawn `metal-tester.md` (run state, bootstrap simulator path, and environment
provisioning). Treat its verdict as the functional result; for `both`, both
suites must pass. Then run `execute_step_aggregate_results`.

## Step 5: Debug loop (tester COMPILE_FAILED/TESTS_FAILED)

While the tester is red and `DEBUG_CYCLES < MAX_DEBUG_CYCLES`:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_debug_feedback "{FAILURE_SUMMARY}"
```

Spawn `issue-worker.md` in debug mode, then run `execute_step_bump_debug`.
Re-run the failed route: Step 4 for tt-llk, Step 4b for metal, or both in order.
Aggregate the new results.
Terminate: `SUCCESS`→Step 5.3; `DEBUG_CYCLES == MAX` still red → Step 6
`execute_step_mark_status failed`; worker `HYPOTHESIS_REFUTED` → Step 6
`execute_step_mark_status failed`. Never debug `SIM_ISA_GAP` — finalize failed.

## Step 5.3: Review loop

Run once tests are green (a diff exists to review).

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_review
```

Spawn `reviewer.md`, then `execute_step_record_review`. Read `blocking_total`:
`0`→Step 5.5. `>0` and `REVIEW_RETRIES < MAX_REVIEW_RETRIES`:
`execute_step_review_feedback "{summary}"`, spawn `issue-worker.md`
(`FAILURE_CLASS=REVIEW_FINDINGS` + `review_result.json`), `execute_step_bump_review`,
re-run the applicable functional route, and if green re-run Step 5.3. Worker
`HYPOTHESIS_REFUTED`→Step 5.5.
When the review budget is exhausted, preserve the functional status and set:

```bash
LOG_DIR="$(python codegen/scripts/state.py --worktree-dir "$WORKTREE_DIR" get LOG_DIR)"
python codegen/scripts/state.py --log-dir "$LOG_DIR" set \
  OBSTACLE unresolved_review_findings
```

Then continue.

## Step 5.5: Perf loop (green only; local BH/WH only)

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
PERF_ARCHES="$(execute_step_perf_arches)"
```

Empty → `execute_step_perf_not_measured`, Step 6. Else:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_perf
```

Spawn `perf-tester.md` (`PERF_GOAL`, changed op, fix plan from state); it writes
`$LOG_DIR/perf_result.json`. Then `execute_step_record_perf` (no arg — top-level
perf). A *miss* = `PERF_REGRESSED`, or `PERF_NOT_IMPROVED` when `PERF_GOAL=improve`.
On a miss with retry budget: `execute_step_perf_feedback "{summary}"`,
spawn `issue-worker.md` (`FAILURE_CLASS=PERF_REGRESSION|PERF_NOT_IMPROVED` + CSV
paths), `execute_step_bump_perf`, re-run the applicable functional route, then
Step 5.5. Exhausted:
`no_regress` + still regressed → set `OBSTACLE=perf_regression` with
`state.py`, then run `execute_step_mark_status failed test_failure`;
`improve` + not improved → keep the functional status.

## Step 6: Finalize

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_deferred_message                  # no-op unless VERIFY_DEFERRED=1
execute_step_status_from_verdict "{final functional verdict}"   # maps verdict → STATUS (skips if already failed)
execute_step_write_generated_patch             # local commit (no push) + generated.patch
execute_step_finalize_run
execute_step_copy_artifacts
```

Return the run summary (`status`, `run_id`, `log_dir`, `branch`,
`base_commit`, `fix_commit`, `worktree_dir`, `patch`, `test_backend`, `perf`,
`review`, cost, `create_pr_requested`, `changed_files`, `obstacle`) — read every
field from `$LOG_DIR/run.json`.
