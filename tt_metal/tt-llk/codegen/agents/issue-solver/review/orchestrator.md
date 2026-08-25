---
name: issue-solver-review-orchestrator
description: "Address the review comments on one open LLK pull request, verified on hardware."
model: sonnet
tools: Read, Bash, Grep, Agent
---

# LLK Review-Round Orchestrator

Run one **address-comments round** on an already-open PR: turn the reviewers'
feedback into code changes and prove them with the same verification a solve gets.

The solve orchestrator with its front half replaced: a solve *discovers* scope and
verification route by analyzing the issue, a round **inherits** both from the solve
that produced the PR. Everything from verification onwards is that same pipeline.

```text
seed → address → functional verification → review → [performance] → finalize
           ↑________________________________|
                    any addresser edit
```

Steps live in `codegen/scripts/issue_solver/orchestrator_steps.sh`, shared with the
solve orchestrators. Use this file only for control flow.

## Input & State

The router provides `WORKTREE_DIR` and has already run
`execute_step_seed_review_state`, which wrote the bootstrap state from the source
solve run. The worktree is checked out at the **PR's head commit**, so the diff
against `origin/main` is the PR as it stands today.

Bootstrap state schema (all seeded, none of it your responsibility to construct):

- `RUN_KIND=review`, `RUN_MODE` (`single`|`multi`)
- `TARGET_ARCH` or `TARGET_ARCHES` — inherited from the solve, widened by the
  dashboard with any arch the reviewers named
- `ISSUE_NUMBER`, `ISSUE_TITLE`, `ISSUE_BODY`, `ISSUE_LABELS`, `ISSUE_COMMENTS`,
  `ISSUE_URL`
- `WORKTREE_BRANCH`, `TEST_BACKEND`, `CREATE_LOCAL_BRANCH=yes`, `CREATE_PR=no`
- `PR_NUMBER`, `PR_HEAD_SHA`, `REVIEW_INPUT`, `SOURCE_RUN_DIR`, `SOURCE_RUN_ID`

## Git Policy

Do not run git mutations directly. Only `execute_step_write_generated_patch` may
create the local commit and patch. **Never push and never touch the PR** — the
dashboard pushes, and only after this run reports success. `gh` is unavailable by
design: this run has no GitHub credentials.

## Agent and Result Conventions

Short names below resolve to these paths. Only the addresser is new; the four
leaves are the solve's, reused verbatim, which is what makes the round's hardware
verification identical to a solve's.

| Name | Path |
|---|---|
| `addresser.md` | `codegen/agents/issue-solver/review/addresser.md` |
| `tester.md` | `codegen/agents/issue-solver/tester.md` |
| `metal-tester.md` | `codegen/agents/issue-solver/metal-tester.md` |
| `ttnn-tester.md` | `codegen/agents/issue-solver/ttnn-tester.md` |
| `reviewer.md` | `codegen/agents/issue-solver/reviewer.md` |
| `perf-tester.md` | `codegen/agents/issue-solver/perf-tester.md` |
| the solve orchestrator | `codegen/agents/issue-solver/orchestrator.md` |

Conventions, authoritative results per stage, and stop conditions are the solve
orchestrator's, unchanged — including: spawn one agent at a time; after every
`FIX_UPDATED` re-route verification and re-record changed files; on
`NO SPACE LEFT ON DEVICE` run `execute_step_report_no_space` and stop; never send
`ENV_ERROR` / `SIM_ISA_GAP` / `PERF_*` back to the addresser. The addresser's
authoritative result is its final marker plus `review_dispositions.json`.

## 1. Setup

From `$WORKTREE_DIR/tt_metal/tt-llk`:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_validate_input "$WORKTREE_DIR"
execute_step_validate_env
execute_step_setup_run
execute_step_setup_review_run
execute_step_write_initial_run_json
```

`execute_step_setup_review_run` imports the source solve's analysis and fix-plan
artifacts. It **fails the run** if they are absent: without the analysis artifact
there is no verification route, and a round that cannot verify must not run at
all. Stop on any rejection.

## 2. Address the Review

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_advance_addresser
```

Spawn `addresser.md`.

- `FIX_APPLIED`: continue.
- `NO_CHANGES_REQUIRED`: the reviewers' points needed no code. Skip to §6 with
  `execute_step_mark_status success` — the dispositions still have to be recorded
  and validated first (below), because the replies are the entire product of this
  round.
- `BLOCKED`: store the reason in `OBSTACLE`, mark the run failed, finalize without
  verification.
- Any other or missing marker: treat it as an orchestration error, not an applied
  fix.

Then, in both the applied and no-change cases:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_record_review_dispositions
```

This validates one disposition per actionable thread and enforces the reply
contract. A failure here is terminal for the round: record `OBSTACLE`, mark
failed, and finalize. Do not paper over it — an unvalidated disposition set is
exactly the defect this round exists to remove.

After `FIX_APPLIED`:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_route_verification
execute_step_record_changed_files
```

If `FIX_APPLIED` produced no diff, treat it as `NO_CHANGES_REQUIRED`.

## 3. Functional Verification

Follow the solve orchestrator §4 verbatim — the same `VERIFY_ROUTE` table, the same
advance helpers, `execute_step_combine_verification_results` +
`execute_step_aggregate_results` afterwards, and the same per-verdict handling.

One difference: `missing` here almost always means the addresser added a test but
did not register it in the analysis artifact's coverage/filter fields. Spend one
debug retry — `execute_step_coverage_feedback` with the selector evidence, spawn
`addresser.md` with `FAILURE_CLASS=MISSING_TEST_COVERAGE`, `execute_step_bump_debug`,
re-route. Never convert `missing` to `none`.

### Debug Loop

Retry only while `DEBUG_CYCLES < MAX_DEBUG_CYCLES`:

1. `execute_step_review_round_feedback tester "<first meaningful failure>"`
2. Spawn `addresser.md` with the concrete failure class and raw-log path.
3. On `FIX_UPDATED`, rerun `execute_step_record_review_dispositions`,
   `execute_step_route_verification`, and `execute_step_record_changed_files`,
   then `execute_step_bump_debug`.
4. Return to functional verification.

`BLOCKED` ends the round failed. If the budget is exhausted with a repairable
failure outstanding, `execute_step_mark_status failed` and finalize — the
dashboard will not push an unverified change, which is the intended outcome.

## 4. Review

The solve orchestrator §5, unchanged — a review round's fix is still a fix and gets
the same scrutiny. Its retry spawns `addresser.md` with
`FAILURE_CLASS=REVIEW_FINDINGS` instead of the issue worker.

## 5. Performance

The solve orchestrator §6, unchanged — but `execute_step_perf_arches` returns empty
unless a disposition set `perf_relevant`, which is the normal case, so this is
usually just `execute_step_perf_not_measured` and on to finalize.

## 6. Finalize

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_deferred_message
execute_step_status_from_verdict "{final functional verdict}"
execute_step_write_generated_patch
execute_step_finalize_run
execute_step_copy_artifacts
```

For `NO_CHANGES_REQUIRED` with a clean tree, call `execute_step_mark_status
success` instead of `execute_step_status_from_verdict`, and still run the
remaining three: the round's product is the validated dispositions, and the
dashboard needs a finalized `run.json` to post them.

Choose the final verdict from the latest valid functional evidence: `SUCCESS` for
real passing verification, `COMPILED_ONLY` / `UNVERIFIABLE_IN_LLK_SUITE` only when
runtime verification was explicitly not applicable. A previously marked failure
stays failed.

Return the summary from `$LOG_DIR/run.json` — status, PR number, commits,
patch, changed files, per-arch functional evidence, review, and the disposition
count. The dashboard reads that record to decide whether to push, and posts one
reply per disposition only if it did.
