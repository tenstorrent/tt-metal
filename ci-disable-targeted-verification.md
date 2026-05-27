# CI Disable Work: Targeted Verification Strategy

This note applies to the "disable deterministic failing tests" workflow.

## Goal

Avoid running every job in a large workflow after each disable change.
Only re-run jobs that were failing before the change.

## Default Rule

For each pipeline PR, the single targeted verification run is a NON-REGRESSION CHECK:
- Dispatch only the jobs that previously contained the disabled tests on `main`.
- The goal is to confirm that tests which were passing on `main` in those jobs are still passing on the PR branch after the disable changes.
- Do not run the full workflow matrix for verification.
- Do not use the verification run to discover new disables.

## PR Verification Budget (Exactly One Run)

Each PR gets exactly one targeted verification run total.

- Never dispatch a second verification run for the same PR, EXCEPT when the prior verification run was infra-inconclusive (see "Interpreting Verification Results" below). Inconclusive runs do not consume the budget.
- The single run exists only to confirm no regressions in jobs that were passing on `main`.
- Do not use PR-branch verification output to discover, justify, or add new disables.
- Plan and scope that one run carefully before dispatch.
- Keep existing artifact-reuse/no-rebuild requirements for that one run.

### Interpreting Verification Results

- "Verification failure" in this policy means: a job that was passing on `main` immediately before the PR is now failing on the PR branch.
- A "failure" conclusion on the verification workflow run as a whole does NOT by itself mean the PR has a regression.
- If a job was already failing on `main` before the PR, its continued failure on the PR branch is NOT a regression and does NOT block merge.
- If a job's failure on the PR is fully attributable to an out-of-scope cause (timeout tracked in the timeout-tracking issue, infra/runner faults, or flaky non-consecutive failures), it does NOT block merge.
- If the verification run could not actually exercise the previously-passing jobs (for example, infra failure during artifact download, container init, runner allocation, or any failure that prevented pytest from running on those jobs), treat the verification as INCONCLUSIVE. An inconclusive run does NOT count against the one-run-per-PR budget.
- For an inconclusive verification, the agent SHOULD dispatch a new verification run on a later session (still subject to the at most THREE new dispatches per session rule). Do not mark the PR as `verified-pass` until a verification run has actually exercised the previously-passing jobs.
- Merge-readiness decision rule: a PR is `verified-pass` (ready to merge) iff every job that was passing on `main` immediately before the PR is still passing on the PR's single verification run.

## PR Disable Batch Policy (Exactly One Initial Batch)

Each PR gets exactly one initial disable batch.

- Before any PR-branch verification run, commit the single allowed initial disable batch based only on deterministic failures already observed on `main`.
- Eligibility for that initial batch requires the same error signature across at least 3 consecutive completed runs on `main`.
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

For this project, verification runs must reuse existing build artifacts. Do not run fresh build steps for targeted verification.

The reusable mechanism is the `use-artifacts-from-run` input on `.github/workflows/build-artifact.yaml`. It is ALREADY DEFINED there — you do NOT need to modify `build-artifact.yaml`. You only modify the PIPELINE workflow file being dispatched (for example `.github/workflows/t3000-e2e-tests.yaml`) so it accepts the input and threads it into its `build-artifact` job.

### Step 1: Add the input to the pipeline workflow's `workflow_dispatch.inputs` block

Open the target pipeline workflow file (e.g. `.github/workflows/t3000-e2e-tests.yaml`). Find the `on:` block. Under `workflow_dispatch.inputs`, add this entry (preserve any existing inputs):

```yaml
on:
  workflow_dispatch:
    inputs:
      # ... existing inputs unchanged ...
      use-artifacts-from-run:
        required: false
        type: string
        default: ""
        description: "Workflow run ID to download build artifacts from (skips build)"
```

### Step 2: Thread the input into the `build-artifact` job's `with:` block

In the same file, find the job that calls `./.github/workflows/build-artifact.yaml`. It usually looks like this:

```yaml
jobs:
  build-artifact:
    uses: ./.github/workflows/build-artifact.yaml
    with:
      build-type: ${{ inputs.build-type || 'Release' }}
      platform: ${{ inputs.platform || 'Ubuntu 22.04' }}
      # ... other existing fields ...
```

Add ONE line at the end of its `with:` block:

```yaml
      use-artifacts-from-run: ${{ inputs.use-artifacts-from-run || '' }}
```

The complete patched job should look like this (existing fields preserved, one new line added at the end of `with:`):

```yaml
jobs:
  build-artifact:
    uses: ./.github/workflows/build-artifact.yaml
    with:
      build-type: ${{ inputs.build-type || 'Release' }}
      platform: ${{ inputs.platform || 'Ubuntu 22.04' }}
      # ... other existing fields ...
      use-artifacts-from-run: ${{ inputs.use-artifacts-from-run || '' }}
```

### Step 3: Commit those workflow edits to the temporary verification branch only

These edits live on the temporary verification branch (the branch you create for pruning jobs); they do NOT belong on the real disable PR branch. Do not include them in the final PR diff.

### Step 4: Dispatch the verification run with the artifact-source run ID

When you dispatch the workflow, pass `use-artifacts-from-run` as a string equal to the chosen source run ID (e.g. `26445104085`). The `build-artifact.yaml` workflow handles the rest: it skips the build job and downloads the build outputs from the source run via its `download-artifacts` job (which only runs when `inputs.use-artifacts-from-run != ''`).

If the source run does not contain the required artifacts (build tarball, and wheel if the pipeline needs one), the downstream test jobs will fail at the "Initialize containers" / artifact-download step — that is the standard signature of an incompatible source run.

### Source run selection rules (unchanged)

- Prefer a successful recent `Merge Gate` run.
- Source run must match the same build intent (platform, build-type, LTO/Tracy expectations).
- Source run must contain the required build outputs (build tarball, and wheel if the pipeline needs one).
- Prefer the same commit, or the nearest compatible commit prior to the disable-only changes.

If no compatible source run exists, do NOT dispatch a fresh build instead. Wait for a compatible source run.

### Verifying the patch worked before dispatch

After committing the workflow edits on the temporary branch, you can sanity-check the patch with a local file read of the workflow file. The patched workflow's `workflow_dispatch.inputs` must list `use-artifacts-from-run`, and the `build-artifact` job's `with:` block must include `use-artifacts-from-run: ${{ inputs.use-artifacts-from-run || '' }}`.

If either is missing, the dispatch will run a fresh build instead of reusing artifacts.

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

1. Identify deterministic failures from completed runs on `main` and confirm each candidate has the same error signature across at least 3 consecutive `main` runs.
2. Before any verification run, build exactly one initial disable batch from those `main`-proven failures and commit it to the PR branch.
3. Do not use PR-branch verification to discover or add new disables; verification is not a disable-discovery pass.
4. Build the one-run target job set only for regression confirmation in jobs that were passing on `main`.
5. Create a temporary verification branch from the current feature branch.
6. In the temporary branch, modify workflow/job selection so only target jobs run.
7. Dispatch exactly one targeted verification run for the PR.
8. After that run, do not add new disables and do not run another verification pass for the same PR.
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

- Each automation session may dispatch up to THREE new workflow runs total.
- Multiple workflow runs may be in progress concurrently across PRs; the cap applies only to new dispatches per session.
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
