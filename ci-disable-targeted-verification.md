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
- Apply the artifact-reuse policy (optional, strongly preferred — see "Build Reuse (Optional, Strongly Preferred)" below) to that one run.

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

## Session Scope (Two Lanes — Focus and Examining)

Each automation session has TWO distinct work lanes: **Focus PRs** (dispatch lane) and **Examining PRs** (lightweight maintenance lane). A single PR is in at most one lane per session.

### Lane A — Examining PRs (lightweight, no dispatch)

Cap: **up to 3 examining PRs per session.**

Examining PRs are existing open draft disable PRs that need maintenance but should NOT consume a workflow-dispatch slot this session. The default eligibility is "PRs in a non-terminal lifecycle"; under **tier 1** terminal state (see `Terminal States (Two Tiers)` → tier 1), eligibility expands to ANY non-merged open draft disable PR — `verified-pass` / `verified-fail` PRs included — because they still need ongoing rebase + disable-revalidation maintenance to stay merge-ready. For each examining PR, the agent performs ALL of:

- Rebase the PR branch onto the latest `origin/main` and push the merge.
- Revalidate every currently-disabled test against the latest `main` runs; remove disables for tests that have been fixed on `main`.
- **Log-analyze any verification run that completed since the previous session**, and transition the PR's lifecycle to `verified-pass`, `verified-fail`, or `verification-inconclusive` based on the result.
- Comment on the PR with what changed this session.
- Sync the disable-tracking issue with the current disable set.
- Update the PR's entry in the state log (`disabling-work-so-far.md`).

**No workflow dispatches occur in the examining lane.** Dispatches happen only in the focus lane below.

The 4-hour throttle's existing carve-outs apply unchanged — a PR in lifecycle `verification-inconclusive`, `behind-main-needs-rebase`, `new`, or `batch-committed`-with-no-verification is NEVER throttled out of the examining lane (see `Automation Efficiency Guardrails` → `Throttle carve-outs`).

### Lane B — Focus PRs (dispatch lane)

Cap: **up to 3 focus PRs per session.** Each focus PR MUST result in exactly one workflow dispatch in the same session. The 3-dispatch session cap and the 3-focus-PR cap are the same budget viewed two ways.

Focus-slot fill priority (in this exact order):

1. **New PRs for uncovered non-Galaxy single-card workflows.** This is the default and the expected primary use of focus slots — the session target is 3 new PRs unless one of the carve-outs in "Fewer than 3 uncovered workflows" below applies.
2. **Legacy `batch-committed` PRs that have never had a verification dispatch.** These predate the same-session-dispatch rule and get their first dispatch when a focus slot is available.
3. **`verification-inconclusive` PRs needing a re-dispatch.**

Per-focus-PR flow:

- **New PR (priority 1):** commit the initial disable batch + push, then dispatch the verification run — same session.
- **Legacy `batch-committed` PR (priority 2)** or **`verification-inconclusive` PR (priority 3):** rebase first (if applicable), then dispatch.

The 3-dispatch session cap is the hard limit. A focus PR whose dispatch would exceed the cap MUST be deferred to the next session; the OUTPUT must explicitly name which other PRs consumed the dispatch slots.

### Target: 3 new PRs per session

The session target is **3 new PRs per session** unless either:

(a) Fewer than 3 uncovered non-Galaxy single-card workflows remain, OR

(b) Legacy `batch-committed` (priority 2) or `verification-inconclusive` (priority 3) PRs absorb focus slots first.

### Fewer than 3 uncovered workflows

When fewer than 3 uncovered non-Galaxy single-card workflows exist, create as many new PRs as there are uncovered workflows (0, 1, or 2). Fill any remaining focus slots from priorities 2 and 3 above if such PRs exist.

If after filling priorities 1, 2, and 3 the agent has fewer than 3 focus PRs because (a) no uncovered workflows remain AND (b) no priority-2 / priority-3 PRs exist to fill the slots, STOP and record the reason in OUTPUT. This is **NOT** terminal state and **NOT** paralysis — it is a legitimate "not enough actionable work this session" outcome. Use the existing `New PR created: N` and `Total dispatches: N` fields with a short reason note, plus a new line:

```
Focus slots filled: N/3 (reason: <e.g. only 1 uncovered workflow remained, no priority-2/3 PRs available>)
```

Tier 2 terminal state (every disable PR `merged` AND every non-Galaxy single-card workflow covered by a merged PR) remains the only legitimate ZERO-action session — see `Terminal States (Two Tiers)` below. Tier 1 (every non-Galaxy single-card workflow covered by an open or merged PR, but not all merged yet) still permits up to 3 examining-PR maintenance passes and any available priority-2/3 focus fills; it is NOT a zero-action session.

### Lane independence

Examining PRs and focus PRs are **distinct sets** — a single PR is in at most one lane per session. Specifically:

- A brand-new PR created in the focus lane does NOT also count as an examining PR (no rebase or revalidation needed; it's fresh).
- A legacy `batch-committed` or `verification-inconclusive` PR being dispatched in the focus lane includes the rebase as part of the focus-lane work, but it does NOT also count against the examining-PR cap.

### Order within a session

1. Refresh state (policy + state log + open draft PR list).
2. Compute examining-PR candidates (existing PRs in non-terminal lifecycle that need rebase / revalidate / log-analysis). Up to 3.
3. Compute focus-PR candidates by priority (new PRs first, then legacy `batch-committed`, then `verification-inconclusive`). Up to 3.
4. Execute examining-PR work (no dispatches).
5. Execute focus-PR work (commit + push for new PRs, then dispatch each one).
6. Update state log and PR comments.
7. Anti-paralysis / terminal-state check before terminating.

### Per-session caps (summary)

- At most THREE new workflow dispatches per session (counted across all PRs combined). Equals the focus-PR cap.
- At most THREE new disable PRs created per session (one per focus slot taken by priority 1).
- At most THREE focus PRs per session.
- At most THREE examining PRs per session.
- The per-PR "exactly one verification run" budget (excluding infra-inconclusive retries) and the per-PR "exactly one initial disable batch" budget are PER PR, NOT per session.

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

## Build Reuse (Optional, Strongly Preferred)

Reusing existing build artifacts via `use-artifacts-from-run` is **strongly preferred** for targeted verification runs because it skips the build step and significantly shortens the iteration loop. However, it is **NOT mandatory**.

If the agent cannot find a successful source run that satisfies REQUIREMENT 5 below (head SHA equals the feature branch's rebase base on `main`) AND the target workflow's build intent (tracy / build-type / platform), the agent MUST fall back to dispatching the verification run WITHOUT `use-artifacts-from-run`. The workflow will then build artifacts fresh — this is an acceptable outcome and is NOT a blocker on the dispatch.

This is a deliberate policy reversal from the previous "If no compatible artifact source run exists: DO NOT dispatch" rule. That rule is gone. Fresh-build dispatch is now an allowed (and expected) outcome when SHA-matching reuse is not possible.

The agent MUST NOT silently substitute a SHA-mismatched run just to "get reuse working". Reuse is allowed ONLY when SHA parity holds. If parity does not hold, fresh build is the correct fallback, NOT a soft block on the dispatch.

### Decision flow

```
1. Compute FEATURE_BASE = $(git merge-base origin/main <feature-branch>)
2. Look for a successful target-workflow run on main with headSha == FEATURE_BASE
   AND matching build intent (tracy/build-type/platform).
3. If found → dispatch with use-artifacts-from-run = <that run id>
   (apply the two YAML edits described below to the temp verification branch).
4. If not found → dispatch WITHOUT use-artifacts-from-run.
   The workflow will build artifacts fresh. This is acceptable and is NOT a blocker.
   Record the reason ("no SHA-matching successful source run for <workflow>") in the
   PR comment and disabling-work-so-far.md.
```

### Worked example: a pipeline whose `main` runs have been failing for days

**A target workflow whose recent `main` post-commit runs have been failing (e.g. the BH post-commit pipeline's consecutive artifact-expiry / build-failure streak that lasts for days) is NOT a verification blocker.** The correct response is unambiguous:

- Take step 4 of the Decision flow above: dispatch the verification workflow WITHOUT `use-artifacts-from-run` (fresh build).
- The verification run still validates whether previously-passing jobs regress on the PR branch. The only thing the fresh-build path changes is build time — the regression check itself is intact.
- Record the reason ("no SHA-matching successful source run for <workflow> — main has been failing since <date>; dispatching fresh build per policy") in the PR comment and `disabling-work-so-far.md`.

**Treating "no recent successful main run" as "waiting for main to recover" is wrong.** Waiting consumes sessions without producing information. Fresh build is always a valid dispatch path; the artifact-reuse optimization is genuinely optional. The Decision flow's step 4 is the answer; it is NOT a fallback to apologize for.

The only scenario in which the agent should refrain from dispatching is when the PR's per-PR verification-run budget is already consumed by a non-inconclusive prior run (see "PR Verification Budget" above) — that is unrelated to whether `main` happens to be passing.

### When reuse IS chosen: required YAML edits

Everything below in this section — the YAML edits, the strict source-run selection rules, the pre-dispatch sanity checks, and the common-failure signatures — applies ONLY when the agent has chosen artifact reuse via step 3 of the decision flow above. When the agent takes the fresh-build fallback (step 4), skip the YAML edits and skip the source-run selection rules (including Requirement 5) entirely for that dispatch.

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
  (b) take the fresh-build fallback: dispatch the verification run WITHOUT `use-artifacts-from-run` (skip these strict source-run rules and skip the YAML edits) so the workflow builds artifacts fresh. Record the reason ("no SHA-matching successful source run for <workflow>") in the PR comment and `disabling-work-so-far.md`. This is acceptable per the "Build Reuse (Optional, Strongly Preferred)" header — it is NOT a blocker.
- The agent MUST NOT silently fall back to a "close enough" / "recent successful" / "tracy matches" source run on a different commit just to keep reuse. A SHA mismatch is a hard failure of THIS requirement (REQUIREMENT 5), not a soft preference. The correct response to SHA mismatch is option (a) or (b) above, NOT a SHA-mismatched reuse.

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

This section applies to every targeted verification run the agent dispatches, regardless of whether the agent chose artifact reuse or the fresh-build fallback. When reuse was chosen, it is the post-dispatch counterpart of "Source run selection rules (STRICT)" above (in particular Requirement 5's SHA-parity check, which must already have been satisfied *before* dispatch). When the fresh-build fallback was chosen, the checks below are still required — they confirm that the fresh build itself completed cleanly.

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

> Focus-PR selection (which PRs the session acts on at all) is governed by "Session Scope (Up to Three PRs)" above — the agent fills up to three focus slots, falling back to creating new disable PRs when fewer than three existing-PR slots are actionable. The steps below describe the per-PR mechanics once a PR is in focus.

> **Newly created PRs flow straight through this procedure in the same session.** A focus PR that was created in this session's second-pass fill (see "Session Scope (Up to Three PRs)") is eligible for — and required to receive — its initial verification dispatch (steps 5–7 below) in the same session as its creation, subject to the 3-dispatch session cap and the artifact-reuse / fresh-build rules. Do not defer a newly created PR's first dispatch to the next session unless the 3-dispatch cap is already exhausted by other focus PRs.

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

The 4-hour throttle exists to prevent thrash on PRs that have already had recent heavy work (a verification was just dispatched, a deep log analysis was just performed). It is NOT a license to idle, and it is NOT a blanket skip for every PR whose `updatedAt` is recent.

### Default throttle

- Do not perform deep re-analysis for draft PRs updated less than 4 hours ago, EXCEPT when one of the carve-outs below applies.
- For draft PRs that are throttled (recent updatedAt AND no carve-out applies), allow only lightweight checks:
  - detecting active -> completed run state transitions
  - confirming explicit blocker resolution
  - handling a PR selected as the current focus item because its run just completed

### Throttle carve-outs (MUST NOT be throttled, regardless of updatedAt)

The throttle MUST NOT apply to a PR in any of the following lifecycle states. These PRs are eligible to be selected as focus PRs and acted on, even if `updatedAt` is under 4 hours old:

- **`new`** — no disable batch has been committed to the PR yet. The next step is committing the initial disable batch; the throttle has nothing to throttle.
- **`batch-committed` with no verification ever dispatched** — the PR is sitting in the "waiting to be verified" state. Dispatching its first verification is exactly the work the session should be doing.
- **`verification-inconclusive`** — these PRs are retry-eligible by policy (see "Interpreting Verification Results"). Throttling them would make them sit forever.
- **PR is behind `main` and needs a rebase** — rebase + push is allowed regardless of `updatedAt`. (Note: a session-start rebase that itself pushes a merge commit will set the PR's `updatedAt` to "now" — that is precisely the kind of self-bump the throttle must not be tricked by.)

### Why the carve-outs matter

The automation pushes a state-log commit and may merge `main` into each PR at session start. Both of those actions bump `updatedAt`. Without carve-outs, the throttle becomes a self-inflicted starvation loop: every session bumps `updatedAt` to "now", and the next session skips every PR for being <4h old, and the session ends with zero work. The carve-outs above are how the throttle stays scoped to its real purpose (prevent thrash on recently-verified or recently-analyzed PRs) instead of freezing the entire pipeline.

### Focus-PR deep analysis budget

- In each automation cycle, perform heavy log / deep failure analysis for at most three focus PRs (matching the three-dispatch session cap). See "Session Scope (Up to Three PRs)" above.
- Keep all non-focus PRs on lightweight status checks only.

## Safety Constraints

- Each automation session may dispatch up to THREE new workflow runs total.
- Multiple workflow runs may be in progress concurrently across PRs; the cap applies only to new dispatches per session.
- Never dispatch unrelated workflows.
- Keep PRs as draft until final validation.
- Do not include temporary workflow-pruning edits in the final PR branch.
- After every workflow dispatch, immediately share the run URL in the status update.

## Terminal States (Two Tiers)

The automation has TWO distinct terminal-style session states. They differ in which conjuncts hold and how much work the session is allowed to do. Tier 2 is strictly stronger than tier 1.

### Tier 1 — "All workflows covered, examining-lane only"

A session is in **tier 1** when this conjunct holds:

1. Every non-Galaxy single-card workflow in the active pipeline list is already covered by an open or merged draft disable PR created by this automation.

When tier 1 holds, the automation MUST:

- **Stop creating new focus PRs** — suppress priority-1 (new PR for uncovered workflow) focus-slot fills for the session. There are no uncovered workflows by definition.
- **Continue examining-lane work on up to 3 PRs.** Examining-lane work is the standard rebase + revalidate-disables-against-`main` + log-analyze-completed-runs + comment + issue sync + state-log-update flow. Under tier 1 the examining lane is allowed on ANY open draft disable PR that has not yet been merged, regardless of lifecycle stage (`new`, `batch-committed`, `verifying`, `verification-inconclusive`, `verified-pass`, `verified-fail` — all eligible), because even `verified-pass`/`verified-fail` PRs need ongoing rebase + disable-revalidation maintenance to stay merge-ready. If fewer than 3 non-merged PRs exist, examine all of them (the lane runs at less than capacity).
- **Continue priority-2 and priority-3 focus-lane fills if any exist.** Legacy `batch-committed`-no-verify PRs (priority 2) and `verification-inconclusive` PRs (priority 3) are existing PRs that still need a first or retry dispatch — those are NOT new-PR creation and remain available in tier 1, subject to the 3-dispatch / 3-focus-PR session caps.
- **Do NOT emit the top-line `Status: no more work left to do`.** Work is still happening in the examining lane (and possibly in the focus lane at priority 2/3); the session is not at terminal stop.
- Emit `Paralysis check: coverage-complete: <K> examining PRs (no new PRs needed; M focus PRs in priorities 2/3 if any)`. This is an acceptable session-end value; NOT a paralysis bug. See `## Anti-Paralysis` for the full enumeration of Paralysis-check outcomes.

### Tier 2 — "Fully done"

A session is in **tier 2** when BOTH of the following are true:

1. Every disable PR ever created by this automation is `merged`. PRs that were closed without merging count as terminal for this conjunct — they are not "open and unmerged" any longer.
2. Every non-Galaxy single-card workflow in the active pipeline list is already covered by a **merged** disable PR (tier 1's coverage requirement, tightened from "open or merged" to "merged only").

When tier 2 holds, the automation MUST do ZERO work — no examining-lane maintenance, no focus-lane dispatches, no new PRs. It MUST emit:

- `Status: no more work left to do` as the top-line OUTPUT field.
- `Paralysis check: terminal: no more work left to do`.

This is the ONLY legitimate zero-action session.

### Relationship between the tiers

- Tier 2 implies tier 1 (every merged PR also counts as an open-or-merged PR for tier 1's coverage conjunct).
- When tier 2 holds, the session takes the tier-2 branch (zero work, terminal status), NOT the tier-1 branch. Check tier 2 first.
- The previous single-tier terminal state ("every open draft PR `verified-pass`/`verified-fail`/`merged` AND every workflow covered") was insufficient: `verified-pass` PRs that sit unmerged still need rebases + disable-revalidation to stay merge-ready, which the new tier 1 captures. Tier 2 is the strict "literally nothing left to do" stop.

Distinguish both tiers from the paralysis failure mode (`Anti-Paralysis` below): paralysis = idled despite having actionable work; tier 1 = no more *new* PRs needed but examining (and priority-2/3 focus) work continues; tier 2 = literally nothing left to do. The OUTPUT FORMAT distinguishes the three via the `Paralysis check` field (`passed` / `limited` / `coverage-complete` / `terminal` / `PARTIAL` / `FAILED`) and the top-line `Status: no more work left to do` (emitted only in tier 2).

## Anti-Paralysis

A session that ends with **0 focus PRs AND 0 examining PRs** is treated as a **paralysis failure mode**, not a normal completion. Empty sessions are bugs unless the terminal-state or limited-work carve-outs below apply.

The vocabulary distinguishes:

- **Focus PRs** = PRs that received a workflow dispatch this session (priority 1: new PRs for uncovered workflows; priority 2: legacy `batch-committed` PRs; priority 3: `verification-inconclusive` PRs). See `Session Scope (Two Lanes — Focus and Examining)`.
- **Examining PRs** = PRs that received lightweight maintenance this session (rebase, revalidation, log analysis of a run that completed since the previous session, comment, issue sync, state-log update). NO dispatch.

When the agent finds itself about to terminate with zero actions in either lane, it MUST do one of the following before ending the session:

- Dispatch a focus-lane verification on any eligible PR (priority 2: legacy `batch-committed`-with-no-verification; priority 3: `verification-inconclusive`; priority 1: a brand-new PR for an uncovered non-Galaxy single-card workflow). "No recent successful `main` run" is not a reason to skip — fresh build is always a valid path (see "Worked example: a pipeline whose `main` runs have been failing for days").
- Create a new disable PR for an uncovered non-Galaxy single-card workflow, up to the per-session focus-PR cap, with its same-session dispatch.
- Examining-lane work on any open draft PR in non-terminal lifecycle: rebase, revalidate, log-analyze a completed verification run, comment, sync the issue, update the state log.

The agent MUST trace which guardrail caused an apparently-empty session and override it where the carve-outs (throttle exceptions, fresh-build fallback, first-class new-PR fill, the examining lane itself) allow.

### Paralysis-check outcomes

The `Paralysis check` field in OUTPUT must use one of these exact prefixes:

- **`passed: N focus PRs (each dispatched) + M examining PRs`** — normal completion. At least one lane produced work and every focus PR received its dispatch.
- **`limited: N focus PRs (only K uncovered workflows + L priority-2/3 PRs available)`** — fewer than 3 focus slots filled because (a) fewer than 3 uncovered non-Galaxy single-card workflows existed AND (b) priority-2 / priority-3 PRs were not sufficient to fill the remaining slots. Acceptable; NOT a paralysis bug. See `Session Scope (Two Lanes — Focus and Examining)` → "Fewer than 3 uncovered workflows".
- **`coverage-complete: <K> examining PRs (no new PRs needed; M focus PRs in priorities 2/3 if any)`** — tier-1 terminal state holds (every non-Galaxy single-card workflow is already covered by an open or merged disable PR) but tier 2 does NOT hold (some PRs are still unmerged). Priority-1 new-PR creation is suppressed for the session, but examining-lane work continued on K PRs and any available priority-2/3 focus fills happened on M PRs. The top-line `Status: no more work left to do` is NOT emitted (work is still happening). Acceptable; NOT a paralysis bug. See `Terminal States (Two Tiers)` → tier 1.
- **`terminal: no more work left to do`** — tier-2 terminal state holds (every disable PR is `merged` AND every non-Galaxy single-card workflow is covered by a merged disable PR). The top-line `Status: no more work left to do` MUST also be emitted. See `Terminal States (Two Tiers)` → tier 2.
- **`PARTIAL: created N focus PRs but only dispatched M (cap reason: …)`** — a focus PR was created but its dispatch was skipped despite the 3-dispatch cap NOT being exhausted by other focus PRs. This is a bug. Acceptable ONLY when the cap is provably exhausted, in which case the reason MUST explicitly name which PRs consumed the dispatch slots.
- **`FAILED: <reason>`** — zero examining PRs AND zero focus PRs, no acceptable terminal-state, coverage-complete, or limited-work justification. Include an explicit guardrail-trace identifying which carve-out, fresh-build fallback, examining-lane action, or new-PR fill the agent failed to invoke. A FAILED paralysis check is a bug report on the automation prompt and should be escalated.

The tier-2 terminal state defined in `## Terminal States (Two Tiers)` above is the canonical zero-action session; the tier-1 `coverage-complete` outcome is the canonical "no new PRs needed but examining-lane work continues" session; the `limited` outcome above is the canonical sub-3-focus-PR-but-non-terminal session. Anything else with zero focus PRs and zero examining PRs is paralysis.

When in doubt between "skip due to throttle" and "do something useful", do something useful. The throttle is a guard against thrash, not a license to idle. Treating BH-artifact-expiry, "main is broken", or "all PRs <4h old" as session-ending blockers is the failure mode this section exists to prevent.

## Draft PR / Issue / Status File Management (Mandatory)

- The disable-tracking issue is the source of truth for the current disable set.
- Keep timeout-involved failures in a separate timeout-tracking issue; do not mix timeout tracking into the disable-tracking issue.
- Keep `disabling-work-so-far.md` in sync with both PR status and workflow run status.
- Every disable removal in the PR must be reflected in both the disable-tracking issue and `disabling-work-so-far.md` in the same session.
- After the initial disable batch is committed, updates should be removal/revalidation only (no new disables on that PR).
