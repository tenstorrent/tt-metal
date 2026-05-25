# CI Disable Work: Targeted Verification Strategy

This note applies to the "disable deterministic failing tests" workflow.

## Goal

Avoid running every job in a large workflow after each disable change.
Only re-run jobs that were failing before the change.

## Default Rule

For each pipeline branch, do not run full workflow verification by default.
Run only the previously failing jobs unless there is a reason to re-expand scope.

## In-Scope vs Out-of-Scope Failures

This effort is only for deterministic runtime/code failures.

- In scope:
  - reproducible test failures with concrete failing test IDs (`FAILED ...`) that recur across consecutive runs
- Out of scope:
  - plain job timeouts (not the same as a proven hung test)
  - flaky/non-consecutive failures
  - infra/runner/network/download/environment faults

Do not spend disable/fix cycles on out-of-scope failures in this project.

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

- Never dispatch more than one workflow run at a time.
- Never dispatch unrelated workflows.
- Keep PRs as draft until final validation.
- Do not include temporary workflow-pruning edits in the final PR branch.

## Exit Criteria for Full Workflow Run

Run full workflow only when:
- targeted failing jobs are stable/green, and
- a final confidence pass is needed before undrafting/hand-off.

