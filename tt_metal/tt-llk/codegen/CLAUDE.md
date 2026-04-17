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

#### Determine Architecture (issues only)

1. **Check labels** — look for `blackhole`, `quasar`, `wormhole` in the issue labels.
2. **Fallback: scan content** — if no architecture label found, scan the issue title and body for:
   - `blackhole`, `bh`, `tt_llk_blackhole` → **blackhole**
   - `quasar`, `qs`, `tt_llk_quasar`, `trinity` → **quasar**
3. **Default** — if still ambiguous, default to **blackhole**.

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

| Architecture | Task Type | Orchestrator | TARGET_ARCH |
|-------------|-----------|-------------|-------------|
| **blackhole** | issue fix | `codegen/agents/issue-solver/orchestrator.md` | `blackhole` |
| **quasar** | issue fix | `codegen/agents/issue-solver/orchestrator.md` | `quasar` |
| **wormhole** | issue fix | `codegen/agents/issue-solver/orchestrator.md` | `wormhole` |
| **quasar** | new kernel | `codegen/agents/quasar/orchestrator.md` | — |
| **blackhole** | new kernel | Not yet supported — inform the user | — |

Pass **all** fetched issue context verbatim to the selected orchestrator: `ISSUE_NUMBER`, `ISSUE_TITLE`, `ISSUE_BODY`, `ISSUE_LABELS`, `ISSUE_COMMENTS`. Never summarize or alter any of these fields — agents need the raw content to parse error messages, stack traces, and reproduction steps.

Also pass:
- `TARGET_ARCH` — the value from the table above (omit for kernel-gen — the quasar orchestrator hardcodes its own arch)
- `WORKTREE_DIR` — the absolute path to the worktree where agents must make all code changes
- `WORKTREE_BRANCH` — the branch name for commits and PRs

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

Two flows: kernel generation (arch-specific) and issue solving (shared, arch-parameterized).

| Flow | Orchestrator | Agents | Notes |
|------|--------------|--------|-------|
| Kernel gen | `codegen/agents/quasar/orchestrator.md` | `codegen/agents/quasar/llk-*.md` | Quasar only today |
| Issue solver | `codegen/agents/issue-solver/orchestrator.md` | `codegen/agents/issue-solver/*.md` | Multi-arch via `TARGET_ARCH` — see `codegen/references/arch-profiles.md` |
