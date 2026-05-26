# CI Disable Work: Single-Run Targeted Verification Policy

This note applies to the "disable deterministic failing tests" workflow.

## Policy Shift

The goal is no longer to make pipelines perfectly green.
For each PR/pipeline session, perform exactly one verification run total after applying disables.

## In-Scope Failures Only

This effort is only for deterministic runtime/code failures.

- In scope:
  - reproducible test failures with concrete failing test IDs (`FAILED ...`)
  - failures that occur in 3 consecutive runs on `main`
- Out of scope:
  - failures that are at least partially due to job timeouts (including plain job timeouts; not the same as a proven hung test)
  - flaky/non-consecutive failures
  - infra/runner/network/download/environment faults

Do not disable tests for out-of-scope failures.

## Timeout-Tracking Issue Workflow (Mandatory)

For each out-of-scope failure that is at least partially due to job timeouts:

- Track it in a separate timeout-tracking issue (distinct from the pipeline disable-tracking issue).
- Explicitly designate that timeout-tracking issue for pipeline re-org follow-up.
- Whenever the timeout-tracking issue is created or updated, record its link and current status in `disabling-work-so-far.md`.

## Session Start Requirements (Mandatory)

At the beginning of each active session for a pipeline branch:

1. Rebase the working branch onto the latest `main`.
2. Re-check latest completed run(s) on `main` for that pipeline.
3. Identify all tests with 3 consecutive deterministic failures on `main` within the workflow scope.
4. Ensure the branch disables all such tests found in scope.

If a previously disabled test no longer fails deterministically on latest `main`, remove its disable from the branch and issue list.

## Single Verification Run Workflow

After the disable set is updated for the current session:

1. Build the changed-job verification scope (do not run full workflow matrices by default).
2. Create a temporary verification branch from the feature branch.
3. Prune workflow/job selection in the temporary branch so only changed jobs are dispatched.
4. Dispatch exactly one verification run for the session.
5. Inspect results, then remove temporary workflow-pruning edits from the real feature branch.

Do not run iterative or repeated verification loops in the same session.

## Build Reuse Requirement (No Rebuilds)

Verification must reuse existing build artifacts.
Do not run fresh builds for targeted verification.

Before dispatching the single verification run:

1. Ensure `workflow_dispatch` accepts an artifact-source run ID (for example, `use-artifacts-from-run`).
2. Thread that input into the workflow's `build-artifact` call so it is passed to `.github/workflows/build-artifact.yaml`.
3. Confirm the called workflow supports artifact reuse and skips build when the run ID is set.
4. Dispatch with that run ID so builds are downloaded instead of rebuilt.

Run ID selection rules:

- Prefer a successful recent `Merge Gate` run.
- Match build intent (platform/build-type/LTO/tracy expectations).
- Source run must contain required build outputs.
- Prefer same commit or nearest compatible commit prior to disable-only changes.

If no compatible artifact-source run is available, do not dispatch a rebuild run.

## Automation Follow-Up Policy

On subsequent automation runs for the same PR/pipeline session:

- If the single verification run has completed, and
- jobs that were previously passing did not regress,

then mark the PR ready to merge even if new failures appear.

New failures discovered after the single run should be recorded and triaged, but do not trigger repeated verification for the same session.

## Safety Constraints

- Keep one active workflow run at a time.
- Do not dispatch unrelated workflows.
- Keep PRs in Draft while disable changes are still being applied.
- Use skip reason format: `Disabled by issue #XXXXX`.
- Do not include temporary workflow-pruning edits in the final PR diff.
- After every workflow dispatch, immediately report the run URL.

## Issue Tracking Sync (Mandatory)

Keep the linked disable-tracking issue in sync with the draft PR at all times.

- Every time a test/parameterization is disabled, update the issue immediately.
- The issue must reflect the full current disable set in the PR.
- If a disable is removed from the PR, remove it from the issue list in the same session.
