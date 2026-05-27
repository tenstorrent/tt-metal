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
- Do not dispatch another verification run on the PR branch; removals are post-run maintenance only.

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

## Operating Procedure (One Run Per PR)

1. Identify failing jobs from the latest relevant completed run on `main`.
2. Build one initial disable batch and commit it to the PR branch.
3. Build the target job set from those failures for a single pruned verification pass.
4. Create a temporary verification branch from the current feature branch.
5. In the temporary branch, modify workflow/job selection so only target jobs run.
6. Dispatch exactly one targeted verification run for the PR.
7. Inspect results and apply deterministic disable/fix decisions on the real feature branch.
8. After that run, do not run another verification pass for the same PR.
9. Subsequent PR updates are removal-only when latest `main` proves a previously disabled test is fixed.

## Automation Efficiency Guardrails

- Do not perform deep re-analysis for draft PRs updated less than 4 hours ago.
- For those recently updated draft PRs, allow only lightweight checks unless an exception applies.
- Lightweight checks include:
  - detecting active -> completed run state transitions
  - confirming explicit blocker resolution
  - handling a PR selected as the current focus item because its run just completed
- In each automation cycle, perform heavy log/deep failure analysis for at most one focus PR.
- Keep all non-focus PRs on lightweight status checks only.

## Safety Constraints

- Never dispatch more than one workflow run at a time.
- Never dispatch unrelated workflows.
- Keep PRs as draft until final validation.
- Do not include temporary workflow-pruning edits in the final PR branch.
- After every workflow dispatch, immediately share the run URL in the status update.

## Draft PR / Issue / Status File Management (Mandatory)

- The disable-tracking issue is the source of truth for the current disable set.
- Keep timeout-involved failures in a separate timeout-tracking issue; do not mix timeout tracking into the disable-tracking issue.
- Keep `disabling-work-so-far.md` in sync with both PR status and workflow run status.
- Every disable removal in the PR must be reflected in both the disable-tracking issue and `disabling-work-so-far.md` in the same session.
- After the initial disable batch is committed, updates should be removal/revalidation only (no new disables on that PR).
