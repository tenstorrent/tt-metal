---
name: issue-solver-reviewer
description: Issue-solver review stage. Reviews an issue-solver candidate for correctness and requirement completeness. Spawned by the issue-solver orchestrators; not for general use.
effort: xhigh
---

Your role playbook is `codegen/agents/issue-solver/reviewer.md`; the delegation
prompt gives its full path under the run's worktree. Read and follow it.

This definition exists only to run review at `xhigh` effort while the rest of
the solve runs at the session's effort. Subagents otherwise inherit the
session's effort, and a solve at `high` passed an incomplete fix through its
first review (run 56945). Everything else about the review is in the playbook.
