---
description: |
  This workflow creates daily repo status reports. It gathers recent repository
  activity (issues, PRs, discussions, releases, code changes) and generates
  engaging GitHub issues with productivity insights, community highlights,
  and project recommendations.

on:
  schedule: daily
  workflow_dispatch:

permissions:
  contents: read
  issues: read
  pull-requests: read
  discussions: read
  copilot-requests: write

network: defaults

tools:
  github:
    # If in a public repo, setting `lockdown: false` allows
    # reading issues, pull requests and comments from 3rd-parties
    # If in a private repo this has no particular effect.
    lockdown: false
    min-integrity: none # This workflow is allowed to examine and comment on any issues

safe-outputs:
  mentions: false
  allowed-github-references: []
  create-issue:
    title-prefix: "[repo-status] "
    labels: [reports]
    close-older-issues: true
engine: copilot

source: githubnext/agentics/workflows/repo-status.md@1c6668b751c51af8571f01204ceffb19362e0f66
---

# Repo Status

Create an upbeat daily status report for the repo as a GitHub issue.

## What to include

- Repository activity from the last 24 hours (ending at the workflow run time, UTC)
- Issues, PRs, discussions, releases, and code changes created or updated in that window
- Progress tracking, goal reminders and highlights
- Project status and recommendations
- Actionable next steps for maintainers

If the last 24 hours contain no qualifying activity, create a brief "no activity"
issue that says so, rather than skipping the run.

## Style

- Be positive, encouraging, and helpful 🌟
- Use emojis moderately for engagement
- Keep it concise - adjust length based on actual activity

## Process

1. Determine the reporting window: the 24 hours ending at the current workflow run time (UTC).
2. Gather activity from that window only: issues, PRs, discussions, releases, and code changes.
3. If there is no activity in the window, create a minimal issue stating that.
4. Otherwise, create a new GitHub issue with your findings and insights.

## Lifecycle

This workflow keeps only one open `[repo-status]` issue at a time. When a new
report is created, older open `[repo-status]` issues are closed automatically.
