# CI Kickoff Workflows

## Overview
Given an existing PR, determine the minimum set of relevant workflows, create a temporary workflow-pruning branch off the PR branch, dispatch those workflows, and post run links as a PR comment.

## Input
- **Required:** PR URL or PR number in `tenstorrent/tt-metal`.
- **Optional:** explicit workflow include/exclude hints from the user.

If no PR is provided, stop and ask for it.

## GitHub Access
- Assume `gh` is already authenticated for this repository.
- Prefer `gh api` for most GitHub reads/writes.
- Use higher-level `gh` commands when they are simpler for dispatch or run-log ergonomics.

## Steps
1. **Resolve PR context**
   - Load PR metadata (number, URL, head ref, base ref, author, title).
   - Fetch changed files for `base...head`.
   - Gather any existing PR comments that already contain workflow kickoff runs to avoid duplicates.

2. **Decide workflows to run**
   - Determine affected areas from changed files.
   - Map affected areas to candidate workflows using `.github/workflows`.
   - Keep only workflows needed to validate PR risk; avoid unrelated broad workflows unless required.
   - Document rationale for each selected workflow.

3. **Create kickoff branch from PR head**
   - Check out the PR head branch locally and update from remote.
   - Create a new branch off it (for example `ci-kickoff/pr-<number>-<short-slug>`).

4. **Prune workflow scope**
   - For each selected workflow file, comment out or disable every matrix/section/test target not relevant to this PR.
   - Preserve syntax validity and `workflow_dispatch` behavior.
   - Use one focused commit with message prefix `[TO REVERT]`.

5. **Push and dispatch**
   - Push kickoff branch.
   - Dispatch selected workflows on the kickoff branch (`--ref <kickoff-branch>`).
   - Capture run URLs/IDs for each dispatched workflow.

6. **Comment on PR with run links**
   - Post a PR comment (do not edit PR description) containing:
     - kickoff branch name,
     - selected workflows,
     - run URL for each workflow,
     - short note that workflows were temporarily pruned for targeted validation.
   - If issue references are needed, keep them non-closing (`Refs #...`), never closing keywords.

7. **Report**
   - Return kickoff branch, `[TO REVERT]` commit hash, dispatched workflows, and PR comment URL.

## Output
- Kickoff branch name.
- Temporary pruning commit hash.
- Workflow run URLs.
- PR comment URL containing the run links.
