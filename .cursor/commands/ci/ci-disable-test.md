# CI Disable Test

## Overview
Take a failing CI signal from a workflow job run URL, a `tenstorrent/tt-metal` issue, or both, and create a focused **draft PR** that disables only the failing job/test path. If an issue is provided, reference it in the PR without auto-closing it on merge.

## Input
- **Accepted inputs (at least one required):**
  - One or more GitHub Actions job URLs (example: `https://github.com/tenstorrent/tt-metal/actions/runs/<run-id>/job/<job-id>`)
  - GitHub issue URL or issue number in `tenstorrent/tt-metal` (example: `39820`)
- **Optional:** both issue + one or more job URLs together.

### Multiple job URLs from the user
When the user pastes **two or more** job links as the primary signal (not just “everything linked in an issue”):
- Download failed logs for **every** linked job.
- Identify the failure (test name/path, workflow step, or matrix leaf) that appears in **all** of those runs.
- Disable **only** that shared failure—the smallest scope that still matches **every** run. Do not disable a failure that shows up in only some of the linked jobs.
- If the failures differ across runs and nothing clearly matches all of them, **do not** implement a disable from a single run. Summarize per-run failures and ask for clarification or a reduced set of links.

## GitHub Access
- Assume `gh` is already authenticated for this repo.
- Prefer `gh api` for most GitHub reads/writes.
- Use higher-level `gh` commands (`gh issue view`, `gh run view`, `gh pr create`) only when they are materially simpler than equivalent API calls.

If no valid input is provided, stop and ask for at least one of the two.

## Steps
0. **Start with a clean slate in disabling workspace**
   - Ensure `build_ci/disabling` exists (create it if missing).
   - Delete all pre-existing files/subfolders in `build_ci/disabling` before doing anything else.
   - Never reuse stale artifacts from prior runs.

1. **Resolve ticket/run context**
   - If issue is provided:
     - Load issue details (prefer `gh api`; `gh issue view ... --json ...` is acceptable).
     - Read issue comments.
     - Extract all linked workflow run/job URLs from body and comments.
   - If job URL is provided:
     - Parse `run_id` and `job_id` from the URL.
   - Build a candidate run/job list from the explicit input plus URLs discovered in the issue discussion.

2. **Download and analyze logs**
   - For each selected run/job candidate, download failed logs into `build_ci/disabling` (prefer `gh api`/logs endpoints; `gh run view <run_id> --job <job_id> --log-failed` is acceptable fallback).
   - If the user supplied multiple job URLs, first extract the failure from each log and **intersect**: the disable target must be the failure present in **all** runs (see **Multiple job URLs from the user**).
   - Analyze logs to determine the narrowest disable action that unblocks CI:
     - exact failing test path/pytest target,
     - failing matrix entry,
     - or specific job section in workflow.
   - Prefer disabling the smallest blast-radius target (test case > file > matrix leaf > whole job).
   - If both issue and explicit job are provided and conflict, prioritize concrete evidence from downloaded logs.

3. **Implement minimal disable**
   - Create a branch from updated `main` named `ci-disable-test-<short-slug>`.
   - Apply the smallest safe disable, such as:
     - excluding one failing test selector from the job invocation,
     - gating one matrix entry,
     - marking one specific test path as disabled/skipped in CI only.
   - Add a clear TODO comment with issue reference near the disable.
   - Do not include unrelated refactors.

4. **Delete transient logs after analysis**
   - Remove downloaded logs from `build_ci/disabling` once analysis is complete and no longer needed.
   - Keep only non-log artifacts that summarize decisions (if any).

5. **Commit and push**
   - Run relevant local checks available in the environment.
   - Commit focused changes and push branch.

6. **Create a draft PR**
   - Open a draft PR with `gh pr create --draft`.
   - If issue exists, reference it using non-closing language:
     - `Refs #<issue-number>` (or equivalent wording).
   - Do **not** use closing keywords (`Closes`, `Fixes`, `Resolves`) for the issue.
   - Include:
     - failing evidence source(s) (issue and/or run URLs),
     - what was disabled and why,
     - scope minimization rationale,
     - re-enable criteria / follow-up plan.
   - When multiple job URLs were provided, note that the change targets the failure observed in **all** of those jobs (or explain why no PR was opened if they disagreed).

## Output
- Branch name and latest commit hash.
- Draft PR URL.
- Explicit note confirming issue reference is non-closing (`Refs #...`).
- Short statement of what exact test/job scope was disabled.
