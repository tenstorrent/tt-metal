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

### Source run selection rules (STRICT — read carefully)

Most artifact-reuse failures come from picking the wrong source run. Follow these rules in order; any miss is a hard failure that will surface as "Could not find build artifact matching expected pattern" or "Workflow run X has conclusion: failure (expected: success)".

> See also: **End-of-session verification of dispatched runs (mandatory)** below — even after a careful pre-dispatch source-run pick, the agent MUST confirm the dispatched run's artifact-acquisition step finished with `success` before ending its session. Otherwise the run is treated as infra-inconclusive and is retry-eligible.

REQUIREMENT 1 — Source run MUST be a SUCCESS:
- The source run's `conclusion` must be `success`. Not `failure`, not `cancelled`, not `in_progress`.
- Verify via the GitHub API / MCP before dispatch. Do NOT pick a source run blindly by recency.

REQUIREMENT 2 — Source run MUST share build intent with the TARGET workflow:
- Build intent = the values of `tracy`, `build-type`, `platform`, `enable-lto`, `build-wheel` (and any other build-affecting input) used by the TARGET workflow's `build-artifact` job.
- A tracy mismatch is the most common failure: if the target workflow builds with `tracy: true`, the source MUST also have been built with `tracy: true`. If the target builds with `tracy: false`, the source MUST also have been built with `tracy: false`.
- Read the target workflow file. Find its `build-artifact` job's `with:` block. Note the explicit build-intent values. If a build-intent input is not set there, it inherits the default from `.github/workflows/build-artifact.yaml` (tracy defaults to `false`, build-wheel defaults to `false`, etc.).

REQUIREMENT 3 — Easiest way to satisfy Requirement 2 (RECOMMENDED):
- Pick the source run from the SAME WORKFLOW the verification is targeting, on `main`, with `conclusion: success`.
- Example: verifying a Blackhole post-commit PR? → source = most recent `Blackhole post-commit tests` run on `main` with conclusion `success`.
- Example: verifying a (T3K) T3000 e2e tests PR? → source = most recent `(T3K) T3000 e2e tests` run on `main` with conclusion `success`.
- This works because the same workflow always uses the same build intent, so the artifact names match by construction.

REQUIREMENT 4 — Merge Gate is only acceptable when build intent demonstrably matches:
- Merge Gate runs typically build with `tracy: false` and the default build-type. They are NOT suitable for workflows that build with `tracy: true` (such as Blackhole post-commit).
- Before using a Merge Gate run, confirm its build intent matches the target workflow's `build-artifact` `with:` block field-by-field.
- If unsure, fall back to Requirement 3 (same workflow's main runs).

REQUIREMENT 5 — Source run's head SHA MUST equal the feature branch's rebase base on `main` (HARD REQUIREMENT):
- The source run's `head_sha` (with `head_branch == main`) MUST equal the exact `main` commit that the feature branch is currently rebased on. Tracy / build-type / platform matching alone is NOT sufficient — commit-SHA parity is the hard requirement.
- Determine the feature branch's rebase base with:
  ```bash
  git fetch origin main
  FEATURE_BASE=$(git merge-base origin/main <feature-branch>)
  ```
  After the mandatory session-start rebase (see "Session Start Rebase + Revalidation"), `FEATURE_BASE` should equal the `origin/main` HEAD that was used for the rebase.
- Confirm the candidate source run with:
  ```bash
  gh run view <candidate-source-run-id> --json headSha,headBranch,event,conclusion,workflowName
  ```
  The returned `headSha` MUST equal `FEATURE_BASE`, `headBranch` MUST be `main`, and `conclusion` MUST be `success`.
- If no successful run on the exact rebase-base commit exists for the target workflow, the agent MUST do one of:
  (a) rebase the feature branch onto a slightly older `main` commit that DOES have a matching successful run for the target workflow (then re-run the session-start revalidation), OR
  (b) skip the verification dispatch this session and record the reason (no SHA-matching source run available) in `disabling-work-so-far.md`.
- The agent MUST NOT silently fall back to a "close enough" / "recent successful" / "tracy matches" source run on a different commit. A SHA mismatch is a hard failure of this requirement, not a soft preference.

Rationale: when the source run was built from a different `main` commit than the feature branch's base, the downloaded build artifacts encode different source code than the tests being executed on the verification branch. The resulting test diffs cannot be cleanly attributed to the disable change — they may instead reflect drift between the build SHA and the test SHA. Agents have repeatedly picked "successful and tracy-matching" runs on a different commit and produced misleading verification results; SHA parity eliminates that failure mode.

PRE-DISPATCH SANITY CHECK (DO THIS EVERY TIME):
Before dispatching, verify all of the following about the chosen source run:
1. `conclusion == "success"` — confirmed via API.
2. Same workflow as target OR build intent confirmed to match field-by-field.
3. Required artifacts present (build tarball; wheel if target needs one). You can list the source run's artifacts via the API and look for the expected name patterns.
4. `headSha` equals the feature branch's rebase base (Requirement 5). Run the exact check below and abort the dispatch on mismatch:
   ```bash
   git fetch origin main
   FEATURE_BASE=$(git merge-base origin/main <feature-branch>)
   SOURCE_SHA=$(gh run view <candidate-source-run-id> --json headSha --jq .headSha)
   if [ "$FEATURE_BASE" = "$SOURCE_SHA" ]; then
     echo "OK — source run head SHA matches feature branch base: $FEATURE_BASE"
   else
     echo "MISMATCH — do not use this run. FEATURE_BASE=$FEATURE_BASE SOURCE_SHA=$SOURCE_SHA"
     exit 1
   fi
   ```
   `tracy` / `build-type` / `platform` matching alone is NOT sufficient — if this check prints `MISMATCH`, pick a different source run (or follow Requirement 5's (a)/(b) options).

If any check fails, pick a different source run. Do NOT proceed with a dispatch that has even one mismatch — the resulting infra-inconclusive run wastes a dispatch slot for no information.

COMMON FAILURE SIGNATURES AND THEIR CAUSES:
- `Workflow run X has conclusion: failure (expected: success)` → Requirement 1 violated. Pick a successful source.
- `ERROR: Could not find build artifact matching expected pattern` with `TRACY_ENABLED (requested): true` and the source was built tracy=false → Tracy mismatch (Requirement 2 violated). Pick a source built with the same tracy setting, or fall back to Requirement 3.
- `ERROR: No ttm_any.tar.zst or ttm_any.tar found after download` → Source run's build job did not publish the expected build artifact. Pick a different source.
- Test job fails at "Initialize containers" / artifact download → Build intent mismatch or missing artifacts in source. See above.
- Tests on the verification branch report unexpected diffs (failures or new passes) in jobs that were stable on `main` immediately before the disable change → Likely Requirement 5 violated (source run's `head_sha` ≠ feature branch's rebase base). The downloaded build encodes different source code than the tests being executed, so the diffs cannot be cleanly attributed to the disable patch. Re-pick a source run whose `headSha` matches the feature branch's `git merge-base origin/main <feature-branch>` and re-dispatch (subject to the per-session dispatch cap). Do NOT accept a "tracy and build-type matched, must be fine" justification — SHA parity is the hard requirement.

### Verifying the patch worked before dispatch

After committing the workflow edits on the temporary branch, you can sanity-check the patch with a local file read of the workflow file. The patched workflow's `workflow_dispatch.inputs` must list `use-artifacts-from-run`, and the `build-artifact` job's `with:` block must include `use-artifacts-from-run: ${{ inputs.use-artifacts-from-run || '' }}`.

If either is missing, the dispatch will run a fresh build instead of reusing artifacts.

### End-of-session verification of dispatched runs (mandatory)

This section is the post-dispatch counterpart of "Source run selection rules (STRICT)" above. See those rules — and in particular Requirement 5 — for the same-commit (head-SHA == feature-branch rebase base) requirement that MUST already be satisfied *before* dispatch; the checks below only confirm the artifact step landed cleanly after a properly-selected source run.

Before the agent ends its session, for every targeted verification run it dispatched during that session, the agent MUST poll the run until its artifact-acquisition step is no longer queued or in progress. The agent is NOT allowed to end the session by simply reporting "verification dispatched" — it must confirm the artifact step reached a terminal state and report that state.

Specifically, the agent must confirm one of the following:

- When the run was dispatched with `use-artifacts-from-run` (artifact reuse), the `download-artifacts` job inside `build-artifact.yaml` completed with `conclusion: success`.
- When the run is building artifacts fresh, the `build-artifact` job completed with `conclusion: success`.

The agent only needs the artifact-acquisition job to reach a terminal state. The remaining test jobs may still be in progress when the session ends — this rule is specifically about the artifact step not being left in an unknown state.

If the artifact step did NOT succeed — for example `conclusion: failure` or `cancelled`, "Could not find build artifact matching expected pattern", "Workflow run X has conclusion: failure (expected: success)", source run not found, or any build-intent mismatch — the run is treated as **infra-inconclusive**:

- The agent MUST say so explicitly in the PR comment and in `disabling-work-so-far.md`.
- Per "Interpreting Verification Results" above, infra-inconclusive runs do NOT consume the PR's one-verification-run budget and ARE retry-eligible (still subject to the per-session dispatch cap).
- The session is not considered complete until that inconclusive classification has been recorded for the run.

Concrete check the agent should run (substitute the dispatched run ID):

```bash
gh run view <run-id> --json jobs --jq '.jobs[] | select(.name | test("build-artifact|download-artifacts")) | {name, status, conclusion}'
```

The agent may also use `gh run view <run-id> --json jobs,conclusion,status` and inspect the `jobs` array directly. Poll until the relevant job's `status` is `completed`, then read its `conclusion`. Only `conclusion: success` lets the session end as "verification dispatched and artifact step healthy". Anything else means "infra-inconclusive, retry-eligible" and the session ends in that state — not as a normal completion.

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
