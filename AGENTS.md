# AGENTS.md — instructions for coding agents authoring changes

This file is for agents **writing** changes to tt-metal (GitHub Copilot cloud
agent and equivalents).

Related instruction files:

| File | Audience | Purpose |
| --- | --- | --- |
| `AGENTS.md` (this file) | cloud agent | how to author and **verify** a change |
| `.github/copilot-instructions.md` | code review | cross-cutting review criteria |
| `.github/instructions/*.instructions.md` | code review | path-scoped review criteria (all carry `excludeAgent: "cloud-agent"`) |

The review files describe how to critique a PR. They are not a specification
for your own work.

## Your environment

You run on an internal runner. The tt-metal toolchain is **not** installed on
that host — it lives in the CI build image, and you reach it through a wrapper
script. Do not try to install compilers or dependencies; do not run
`install_dependencies.sh`.

What the host does have: `docker`, a checkout with submodules already
initialised, and a shared remote ccache.

## Match the check to the change

| You changed | What to run |
| --- | --- |
| C++, headers, kernels — `.cpp` / `.hpp` under `tt_metal/`, `ttnn/`, `tt_stl/`, `tt-train/` | **Build.** See below. |
| nanobind bindings (the C++ behind the Python API) | **Build** — this is C++. |
| CMake — `CMakeLists.txt`, `sources.cmake`, `cmake/`, `build_metal.sh` | **Build**, or at minimum `--configure-only` if you only need to prove configure still works. |
| Python only | No build. Formatting is enforced by `pre-commit` (black, isort, autoflake). |
| YAML, workflows | No build. `pre-commit` runs yamllint and check-yaml. |
| Docs, markdown, CODEOWNERS | No build. |

`.pre-commit-config.yaml` defines the formatting and lint hooks the repo enforces
(including `clang-format` and `gersemi` for CMake). Run them if available; do not
treat their absence in your environment as a reason to skip the table above.

**If you are unsure whether your change affects the build, build it.**

## Building

If you changed C++ or CMake, compile before opening the PR.

From the repository root:

```bash
.github/scripts/copilot-build.sh
```

That runs `build_metal.sh --enable-ccache` inside the CI build image against
your working tree, and prints a ccache summary when it finishes. Any arguments
you pass go straight through to `build_metal.sh`:

| Command | When |
| --- | --- |
| `.github/scripts/copilot-build.sh` | default — the usual case |
| `… --configure-only` | prove CMake still configures, without compiling (~6 min) |
| `… --build-metal-tests` | you changed something under `tt_metal/` with tests |
| `… --build-ttnn-tests` | likewise for `ttnn/` |
| `… --build-programming-examples` | you touched `tt_metal/programming_examples/` |
| `… --build-tt-train` | you touched `tt-train/` |
| `… -b Debug` | you need assertions to reproduce something |

`--enable-ccache` is always applied for you. Build the narrowest thing that
actually exercises your change; do not reach for `--build-all`.

If the wrapper warns that Garage credentials are missing, you are building
against a cold cache and it will most likely not finish. Say so in the PR
rather than burning the session on it.

## What to do about a build

- **Builds clean** — say so explicitly in the PR description, including the
  exact command you ran.
- **Fails to build** — fix it and rebuild. Do not open the PR and let CI find
  a compile error you could have caught.
- **Did not need a build** (see the table above) — say which check you ran
  instead, e.g. that it is a docs-only change.
- **Genuinely cannot build** (cold cache, docker unavailable, environment
  problem) — open the PR anyway, and state in the description that the change
  is **unverified**, and why.

Do not claim you ran anything you did not run.

## Things you cannot verify here

The runner has no Tenstorrent accelerator attached, so anything requiring real
silicon — device tests, performance measurements, hardware-dependent
behaviour — cannot be checked in your environment. Compilation and host-side
unit tests are in scope; on-device results are not.

If a change's correctness depends on device behaviour, say so.

Never state a performance improvement without measurements.

## Scope discipline

- Change the minimum needed to solve the stated issue.
- New source files go in the relevant `sources.cmake`, not into
  `CMakeLists.txt` build structure.
- Adding an external dependency (`find_package`, `CPMAddPackage`,
  `FetchContent_Declare`, a new `third_party/` submodule) requires infra team
  review. If the issue seems to need one, stop and say so in the PR rather than
  adding it.

<!-- BEGIN BEADS INTEGRATION v:1 profile:full hash:19cc25d9 -->
## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Dolt-powered version control with native sync
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion

### Quick Start

**Check for ready work:**

```bash
bd ready --json
```

**Create new issues:**

```bash
bd create "Issue title" --description="Detailed context" -t bug|feature|task -p 0-4 --json
bd create "Issue title" --description="What this issue is about" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**

```bash
bd update <id> --claim --json
bd update bd-42 --priority 1 --json
```

**Complete work:**

```bash
bd close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for AI Agents

1. **Check ready work**: `bd ready` shows unblocked issues
2. **Claim your task atomically**: `bd update <id> --claim`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bd create "Found bug" --description="Details about what was found" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bd close <id> --reason "Done"`

### Quality
- Use `--acceptance` and `--design` fields when creating issues
- Use `--validate` to check description completeness

### Lifecycle
- `bd defer <id>` / `bd supersede <id>` for issue management
- `bd stale` / `bd orphans` / `bd lint` for hygiene
- `bd human <id>` to flag for human decisions
- `bd formula list` / `bd mol pour <name>` for structured workflows

### Sync

bd stores issue history in Dolt:

- Each write auto-commits to Dolt history
- Use `bd dolt push`/`bd dolt pull` for remote sync
- Do not treat `.beads/issues.jsonl` as the sync protocol

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

### Important Rules

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bd ready` before asking "what should I work on?"
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems

For more details, see README.md and docs/QUICKSTART.md.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   bd dolt push
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.

<!-- END BEADS INTEGRATION -->

<!-- BEGIN BEADS CODEX SETUP: generated by bd setup codex -->
## Beads Issue Tracker

Use Beads (`bd`) for durable task tracking in repositories that include it. Use the `beads` skill at `.agents/skills/beads/SKILL.md` (project install) or `~/.agents/skills/beads/SKILL.md` (global install) for Beads workflow guidance, then use the `bd` CLI for issue operations.

### Quick Reference

```bash
bd ready                # Find available work
bd show <id>            # View issue details
bd update <id> --claim  # Claim work
bd close <id>           # Complete work
bd prime                # Refresh Beads context
```

### Rules

- Use `bd` for all task tracking; do not create markdown TODO lists.
- Run `bd prime` when Beads context is missing or stale. Codex 0.129.0+ can load Beads context automatically through native hooks; use `/hooks` to inspect or toggle them.
- Keep persistent project memory in Beads via `bd remember`; do not create ad hoc memory files.

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.
<!-- END BEADS CODEX SETUP -->
