# Check PR Status

## Overview
Check the status of all your open pull requests, including review status and CI checks.

## Steps
1. **Get open PRs**
   - Run `gh pr list --author @me --state open` to find all your open PRs
   - If no PRs found, report that and stop

2. **For each PR, check review status**
   - Run `gh pr view <PR_NUMBER> --json reviews,reviewRequests,reviewDecision`
   - Report if PR has required approvals
   - Report if any reviewer has requested changes
   - Report if reviews are still pending

3. **For each PR, check CI status**
   - Run `gh pr checks <PR_NUMBER>`
   - Report if all checks are passing
   - Report any failing or pending checks

4. **Summarize findings**
   - List PRs that are ready to merge (approved + checks passing)
   - List PRs that need attention (changes requested or checks failing)
   - List PRs waiting on reviews
