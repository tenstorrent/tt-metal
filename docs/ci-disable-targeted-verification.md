# CI Disable Work: Targeted Verification Strategy

This note applies to the "disable deterministic failing tests" workflow.

## Goal

Avoid running every job in a large workflow after each disable change.
Only re-run jobs that were failing before the change.

## Default Rule

For each pipeline branch, do not run full workflow verification by default.
Run only the previously failing jobs unless there is a reason to re-expand scope.

## PR Verification Budget (Exactly One Run)

Each PR gets exactly one targeted verification run total.

- Never dispatch a second verification run for the same PR.
- Plan and scope that one run carefully before dispatch.
- Keep existing artifact-reuse/no-rebuild requirements for that one run.

## PR Disable Batch Policy (Exactly One Initial Batch)

Each PR gets exactly one initial disable batch.

- After the first disable batch is committed to that PR, do not add new disables to that PR.
- Exception: removal is allowed. If revalidation on `main` shows a previously disabled test is fixed, remove that disable (add the test back).

## Session Start Rebase + Revalidation (Mandatory)

At the beginning of each active session for a pipeline branch:

1. Rebase the working branch onto the latest `main`.
2. Re-check the latest completed run(s) on `main` for that pipeline.
3. Confirm each test currently disabled by the branch still fails deterministically on `main`.

If any test no longer fails on latest `main`:

- Remove its skip/disable from the branch.
- Keep only disables for tests that are still deterministically failing.
- Continue the targeted verification loop with the reduced disable set.

Rationale: PRs may remain open for days, and we must avoid disabling tests that have already been fixed and are now passing.

## Build Reuse Requirement (No Rebuilds)

For this project, verification runs must reuse existing build artifacts.
Do not run fresh build steps for targeted verification.

Before dispatching a verification run:

1. Modify the pipeline workflow file being dispatched so `workflow_dispatch` can accept an artifact-source run ID (for example, `use-artifacts-from-run`).
2. Thread that input into the workflow's `build-artifact` call so it is passed to `.github/workflows/build-artifact.yaml`.
3. Confirm the called workflow supports artifact reuse and skips build when the run ID is set.
4. Dispatch the verification run with that run ID so build artifacts are downloaded instead of rebuilt.

Run ID selection rules:

- Prefer a successful recent `Merge Gate` run.
- The source run should match the same build intent (platform/build-type/LTO/tracy expectations).
- The source run must contain the required build outputs (build tarball and wheel when needed by the pipeline).
- Prefer a source run on the same commit, or the nearest compatible commit prior to the disable-only changes.

If no compatible artifact-source run is available, do not dispatch a rebuild run.
First find another compatible source run or update the plan/branch flow to obtain one.

## In-Scope vs Out-of-Scope Failures

This effort is only for deterministic runtime/code failures.

- In scope:
  - reproducible test failures with concrete failing test IDs (`FAILED ...`) that recur with the same deterministic failure pattern across 3 consecutive runs on `main`
- Out of scope:
  - plain job timeouts (not the same as a proven hung test)
  - flaky/non-consecutive failures
  - infra/runner/network/download/environment faults

Timeout handling rule:

- Timeouts remain out of scope for disable actions and must stay in the timeout-tracking issue workflow.

Do not spend disable/fix cycles on out-of-scope failures in this project.

## Session Scope (Up to Three PRs)

An automation session may actively work on up to three draft PRs (creating, advancing, or verifying them), bounded only by the three-dispatch cap and the four-hour anti-loop throttle.

- Up to three focus PRs per session for heavy analysis (matching the three-dispatch cap).
- Up to three new workflow dispatches per session, counted across all PRs combined.
- The per-PR budgets ("Exactly One Run", "Exactly One Initial Batch") are PER PR, NOT per session — they do not reduce the session-level allowance.
- Concurrent runs across different PRs are allowed; the cap is on total dispatches, not on simultaneity.
- The four-hour anti-loop throttle applies on top of the three-dispatch cap to prevent rapid re-dispatch loops on the same PR.

## Operating Procedure

1. Identify failing jobs from the latest relevant run:
  - Initial pass: latest completed run on `main`
  - Iteration pass: latest completed run on the feature branch
2. Build the target job set from those failures.
3. Create a temporary verification branch from the current feature branch.
4. In the temporary branch, modify workflow/job selection so only target jobs run.
5. Dispatch workflow on the temporary branch.
6. Inspect results and apply test-disable fixes on the real feature branch.
7. Repeat until targeted failures are either fixed, disabled, or classified as infra/flaky.

## Safety Constraints

- Cap total workflow dispatches at three per session, counted across all PRs. Concurrent runs across different PRs are allowed; the cap is on total dispatches, not on simultaneity. Respect the four-hour anti-loop throttle on top of the three-dispatch cap.
- Never dispatch unrelated workflows.
- Keep PRs as draft until final validation.
- Do not include temporary workflow-pruning edits in the final PR branch.
- After every workflow dispatch, immediately share the run URL in the status update.

## Issue Tracking Sync (Mandatory)

Keep the linked disable-tracking issue in sync with the draft PR at all times.

- Every time a new test/parameterization is disabled, immediately update the issue with that test ID.
- The issue must always reflect the full current set of disables in the PR (not just the latest addition).
- If a disable is removed from the PR, remove it from the issue list in the same session.
- After the initial disable batch is committed, issue updates should only reflect removals/revalidation outcomes (no new disables).

## Exit Criteria for Full Workflow Run

Run full workflow only when:

- targeted failing jobs are stable/green, and
- a final confidence pass is needed before undrafting/hand-off.
