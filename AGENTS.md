# Codex repository guidance

For ordinary tt-metal work, follow the repository's existing build/test conventions and the nearest
nested `AGENTS.md`, if any.

For a goal-driven TTNN operation implementation/evaluation—especially a request to replace the
nested Python/LLM orchestrator with one long-lived `/goal`—read and follow
`.claude/agents_codex/goal-coordinator.md` before changing code. It routes to the detailed axis,
testing/gating, JIT/performance/tournament, and reporting/recovery references. The agent owns the
rolling task list; `python3 -m eval.goal_runner` owns deterministic state, canonical test execution,
mechanical gates, commit boundaries, and database synchronization.

Never embed developer-specific absolute paths in reusable code, tests, agent instructions, or docs.
Discover the repository, virtual environment, simulator, cache, worktree, and database paths from
arguments, environment variables, git, `sys.prefix`, or portable user-relative defaults.

Do not invoke subagents merely because custom TT agents are installed. Use them only when the user
explicitly requests/authorizes delegation, and give every write-capable candidate an isolated git
worktree.
