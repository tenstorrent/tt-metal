# LLK CodeGen

## Git Policy: Read-Only

Read-only git commands are allowed (`git rev-parse`, `git log`, `git status`, `git diff`, `git show`). **NEVER push, commit, checkout, restore, reset, or otherwise modify** the repo via git.

---

## Step 1: Classify the Request

Determine the request type and extract a **TASK_ID** for worktree naming:

### Generate Kernel (direct request)

When a user asks to **"generate {kernel} for {target_arch}"**:
- `REQUEST_TYPE` = `generate`
- `TARGET_ARCH` = the requested architecture (default: **quasar**)
- `KERNEL_NAME` = the kernel to generate
- `TASK_ID` = `generate-{KERNEL_NAME}-{TARGET_ARCH}` (e.g., `generate-gelu-quasar`)

### Solve a GitHub Issue

When a user references a **GitHub issue** (e.g., "solve issue #123", "fix #456", "work on issue 789"):
- `REQUEST_TYPE` = `issue`
- `TASK_ID` = `issue-{ISSUE_NUMBER}` (e.g., `issue-123`)

Then fetch **all** issue data — title, body, comments, and labels:

```bash
gh issue view {number} --json number,title,body,labels,comments
```

Extract and store verbatim:
- `ISSUE_NUMBER` — the issue number
- `ISSUE_TITLE` — the issue title, unmodified
- `ISSUE_BODY` — the full issue description, unmodified (includes error messages, reproduction steps, code snippets, etc.)
- `ISSUE_LABELS` — all labels as a list
- `ISSUE_COMMENTS` — all comments in full, unmodified (includes follow-up context, clarifications, stack traces, etc.)

**CRITICAL: Never alter, summarize, paraphrase, or truncate any issue content.** The raw title, body, and comments must be passed as-is to every subagent. Agents depend on exact error messages, code snippets, and reproduction steps from the issue to do their work correctly.

#### Determine Architecture(s) (issues only)

Collect **all** relevant architectures into `TARGET_ARCHES` (a list). Issues labeled for more than one arch are real — an API change to an LLK function usually needs to land on every arch that implements it — and must be handled as a single coordinated fix, not N independent runs.

1. **Check labels** — collect every matching label into a list: `blackhole`, `quasar`, `wormhole`. All are equally valid entries.
2. **Fallback: scan content** — if no architecture labels are found, scan the issue title and body for:
   - `blackhole`, `bh`, `tt_llk_blackhole` → add **blackhole**
   - `quasar`, `qs`, `tt_llk_quasar`, `trinity` → add **quasar**
   - `wormhole`, `wh`, `tt_llk_wormhole_b0` → add **wormhole**
3. **Default** — if `TARGET_ARCHES` is still empty, default to `[blackhole]`.

Then set:
- `TARGET_ARCHES` — the list (always at least one element).
- `TARGET_ARCH` — single-arch convenience: `TARGET_ARCHES[0]` when `len(TARGET_ARCHES) == 1`, else **unset** (the multi-arch orchestrator uses `TARGET_ARCHES`).
- `MULTI_ARCH` — `true` if `len(TARGET_ARCHES) > 1`, else `false`. Used for routing in Step 3.

#### Determine Task Type (issues only)

1. **Check labels** — look for:
   - Creation: `new-kernel`, `enhancement`, `feature`, `implement`, `port`
   - Issue fix: `bug`, `fix`, `defect`, `regression`, `compile-error`, `test-failure`
2. **Fallback: keyword heuristics** — if labels are inconclusive, scan title and body for:
   - **New kernel** signals: "implement", "add", "create", "port", "new kernel", "missing", "generate"
   - **Issue fix** signals: "fix", "broken", "error", "fail", "crash", "wrong", "incorrect", "regression", "compile"
3. **Default** — if still ambiguous, treat as **issue fix**.

---

## Step 2: Create Branch and Worktree

Set up an isolated worktree so all code changes happen on a dedicated branch based on `origin/main`. The codegen infrastructure (agents, scripts, references, config) is symlinked into the worktree from the current branch.

```bash
source codegen/scripts/setup_worktree.sh
setup_worktree {TASK_ID}
# Exports: WORKTREE_DIR, WORKTREE_BRANCH
```

After this step:
- `WORKTREE_DIR` — absolute path to the worktree (e.g., `/tmp/codegen_worktree_issue-123`)
- `WORKTREE_BRANCH` — the branch name (e.g., `ai-code-gen/issue-123-v1`)
- All agent playbooks are accessible via symlinks in the worktree
- `codegen/artifacts/` is a real (non-symlinked) directory for per-task artifacts

---

## Step 3: Route to Orchestrator

### Generate Kernel

| Architecture | Orchestrator |
|-------------|-------------|
| **quasar** | `codegen/agents/quasar/orchestrator.md` |
| **blackhole** | Not yet supported — inform the user |

Pass to the orchestrator:
- `KERNEL_NAME`, `TARGET_ARCH`
- `WORKTREE_DIR`, `WORKTREE_BRANCH`

### Solve Issue

Route by task type and by `len(TARGET_ARCHES)`:

| Architecture(s) | Task Type | Orchestrator | Arch input |
|-----------------|-----------|--------------|------------|
| single (any of blackhole / quasar / wormhole) | issue fix | `codegen/agents/issue-solver/orchestrator.md` | `TARGET_ARCH` |
| **multiple** (e.g. `blackhole + wormhole`) | issue fix | `codegen/agents/issue-solver/orchestrator-multi.md` | `TARGET_ARCHES` (JSON array) |
| **quasar** | new kernel | `codegen/agents/quasar/orchestrator.md` | — |
| **blackhole** | new kernel | Not yet supported — inform the user | — |

The single-arch path (`orchestrator.md`) is **unchanged** from before — today's callers keep working bit-for-bit. The multi-arch path (`orchestrator-multi.md`) runs a **shared analyzer + planner** once, then forks per-arch **fixer + tester** subagents against a single shared design doc. This prevents each arch from inventing its own API shape for the same conceptual change.

Pass **all** fetched issue context verbatim to the selected orchestrator: `ISSUE_NUMBER`, `ISSUE_TITLE`, `ISSUE_BODY`, `ISSUE_LABELS`, `ISSUE_COMMENTS`. Never summarize or alter any of these fields — agents need the raw content to parse error messages, stack traces, and reproduction steps.

Also pass:
- `TARGET_ARCH` (single-arch path) **or** `TARGET_ARCHES` (multi-arch path) — never both.
- `WORKTREE_DIR` — the absolute path to the worktree where agents must make all code changes. For multi-arch issues the worktree is shared: every arch's fixer edits files under its own `tt_llk_{arch}/` subdirectory, so there's no conflict surface and the final branch carries all changes in one place.
- `WORKTREE_BRANCH` — the branch name for commits and PRs.

---

## Step 4: Cleanup

After the orchestrator completes (regardless of success or failure), clean up:

```bash
source codegen/scripts/setup_worktree.sh
cleanup_worktree {TASK_ID}
```

If the orchestrator succeeded and changes should be preserved, commit and push from the worktree **before** cleanup.

---

## Orchestrators

Three flows: kernel generation (arch-specific), single-arch issue solving, and multi-arch issue solving (shared design + per-arch fork).

| Flow | Orchestrator | Agents | Notes |
|------|--------------|--------|-------|
| Kernel gen | `codegen/agents/quasar/orchestrator.md` | `codegen/agents/quasar/llk-*.md` | Quasar only today. Unaffected by multi-arch issue-solver work. |
| Issue solver (single-arch) | `codegen/agents/issue-solver/orchestrator.md` | `codegen/agents/issue-solver/*.md` | Used when `len(TARGET_ARCHES) == 1`. Parameterized by `TARGET_ARCH` — see `codegen/references/arch-profiles.md`. |
| Issue solver (multi-arch) | `codegen/agents/issue-solver/orchestrator-multi.md` | same `codegen/agents/issue-solver/*.md` agents, spawned with a shared plan | Used when `len(TARGET_ARCHES) > 1`. Shared `issue_analyzer` + `fix_planner` produce one `issue_{N}_shared_design.md` with a locked `## API Contract`, then `fixer` + `tester` fork per-arch. One worktree, one branch, one PR. |
