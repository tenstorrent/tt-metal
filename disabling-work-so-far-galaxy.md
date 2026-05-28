# CI Disable — Galaxy Pipelines: Work So Far

Last updated: _(none yet — initial skeleton)_

> **Scope: GALAXY PIPELINES ONLY.** Companion file to `ci-disable-targeted-verification-galaxy.md`. The non-Galaxy state log is `disabling-work-so-far.md`; do not confuse the two.

---

## How to read/update this file

- Read this file at the start of every automation session and treat it as the authoritative current state for CI disable work.
- Scan the `## Quick Index` table first; it gives the lifecycle stage per PR before drilling into details.
- Per-PR sections use uniform field tables (`PR | Disable issue | Timeout issue | Branch | Workflow file | Lifecycle stage | Last rebase | Last revalidation | Verification run | Readiness`); update fields in place rather than rewriting the section.
- Append new entries to the top of `## Recent Activity` (most recent first); keep at most 30 entries — trim older entries to a single `- Older history truncated — see git history of this file.` line if needed.
- Commit and push any change to this file before ending the session.
- Lifecycle stages: `new`, `batch-committed`, `verifying`, `verification-inconclusive`, `verified-pass`, `verified-fail`, `merged`, `out-of-scope`. (`verification-inconclusive` = a verification was dispatched but failed to actually exercise the previously-passing jobs; eligible for re-dispatch and does NOT consume the one-run-per-PR budget.)

---

## Quick Index

| PR | Workflow | Lifecycle stage | Verification result | Ready to merge? | Notes |
|----|----------|-----------------|---------------------|-----------------|-------|
| _(none yet)_ |  |  |  |  |  |

---

## Active Runs

| Run | Pipeline | Branch | Started | Status | Notes |
|-----|----------|--------|---------|--------|-------|
| _(none)_ |  |  |  |  |  |

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
| _(none)_ |  |  |  |  |  |  |

---

## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| _(none)_ |  |  |

---

## Recent Activity

- _(none yet)_