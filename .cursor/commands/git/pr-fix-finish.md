# Finish PR Fix Session

## Overview
Finish the PR fix session by cherry-picking commits to the PR branch and leaving comments on GitHub.

## Input
- The tracking file `_local/{PR_ID}_{name}.md`

## Steps

1. **Review all addressed comments**
   - Check all comments with status `⚠️ ADDRESSED, TO MERGE AND LEAVE COMMENT`
   - Ensure each has a `**My comment:**` line

2. **List commits to cherry-pick**
   - Show commits on the `.fix` branch that are not on the PR branch
   - Exclude commits with `LOCAL:` prefix
   - Show: `git log --oneline {pr_branch}..{fix_branch}`

3. **Switch to PR branch and cherry-pick**
   ```bash
   git checkout {pr_branch}
   git cherry-pick {commit1} {commit2} ...
   ```
   - Or use interactive rebase if commits need to be squashed

4. **Push to PR branch**
   ```bash
   git push
   ```
   - This will trigger CI

5. **Leave comments on GitHub**
   - For each addressed comment, use GitHub MCP to reply with the `**My comment:**` text
   - Consider using `pull_request_review_comment_create` or similar

6. **Update tracking file**
   - Change status from `⚠️ ADDRESSED` to `✅ DONE` for each commented item

## Commit Prefixes Reference
- `fix:` - Bug fixes and review comment fixes (cherry-pick to PR)
- `LOCAL:` - Local-only changes like .gitignore updates (do NOT cherry-pick)

## Notes
- Always verify CI passes after pushing
- Keep the `.fix` branch for reference until PR is merged
- Delete `.fix` branch after PR is merged: `git branch -d {fix_branch}`
