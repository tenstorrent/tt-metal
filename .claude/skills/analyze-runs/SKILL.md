---
name: analyze-runs
description: Analyze GitHub Actions CI runs for failures, find consistent patterns, and identify code owners. Use when debugging nightly pipeline failures or triaging CI breakages.
argument-hint: <run-url-or-id> [run-url-or-id] ...
---

# Analyze CI Run Failures

You are analyzing GitHub Actions runs to identify failures, find consistency patterns across runs, and determine who to contact.

## Input

The user has provided these run URLs or IDs: $ARGUMENTS

## Instructions

### Phase 1: Extract Run IDs

Parse run IDs from the arguments. They may be:
- Full URLs like `https://github.com/tenstorrent/tt-metal/actions/runs/22173724149`
- Full job URLs like `https://github.com/tenstorrent/tt-metal/actions/runs/22173724149/job/64181184885`
- Bare run IDs like `22173724149`

Extract just the numeric run ID from each.

### Phase 2: Analyze Runs in Parallel

Launch one `general-purpose` subagent per run using the Task tool. **Launch ALL agents in a single message** so they run in parallel. Each agent should:

1. List failed jobs:
   ```
   gh api repos/tenstorrent/tt-metal/actions/runs/{RUN_ID}/jobs --paginate -q '.jobs[] | select(.conclusion == "failure") | {id: .id, name: .name, html_url: .html_url}'
   ```

2. For each failed job, fetch logs and extract:
   - The specific test name(s) that failed
   - Error messages, assertion failures, crash signals
   - Exit codes

   Use: `gh run view {RUN_ID} --repo tenstorrent/tt-metal --log-failed 2>&1 | tail -500`
   Or per-job: `gh api repos/tenstorrent/tt-metal/actions/jobs/{JOB_ID}/logs 2>&1 | tail -200`

3. Return a structured summary: run date, list of failed jobs (name + URL + error details).

### Phase 3: Cross-Reference Failures

Once all agents return, consolidate results:

1. **Group failures by test/job name** across runs
2. **Classify each** as:
   - **Consistent** — fails in most/all runs
   - **Intermittent** — fails in some runs only
3. For each distinct failure, note the root cause pattern (same error? same exit code?)

### Phase 4: Find Code Owners

For each distinct failing test, find owners using git:

1. Find the relevant source file path (from error messages / job names)
2. Check `CODEOWNERS` for matching path patterns:
   ```
   grep -i "{path_pattern}" .github/CODEOWNERS || grep -i "{path_pattern}" CODEOWNERS
   ```
3. Find top contributors:
   ```
   git log --format='%an <%ae>' -- {file_path} | sort | uniq -c | sort -rn | head -5
   ```

### Phase 5: Generate Report

Produce a **compact, informal report** the user can paste into Slack. Format:

For each distinct failure:
```
**{Failure title} ({N}/{total} runs)**
Hey @{top_contact}, {brief description of what's failing and the error}.
Failing runs:
- [Date1](job_url_1)
- [Date2](job_url_2)
```

Group consistent failures first, then intermittent ones.

Keep it concise — the user wants to ping people on Slack, not read a thesis. Use GitHub handles for contacts (the user will swap to Slack handles). Include just enough technical detail (error message, exit code, test name) for the recipient to understand the problem.

At the end, offer to write a more detailed markdown report file if the user wants one.
