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
- `SFPI_MODE` = `true` if the user **explicitly** asked for an SFPI version (phrases like "as SFPI", "sfpi version", "in sfpi", "write it in sfpi"); otherwise `false`. This is an SFPU-only optimizer directive — the writer still mirrors the Blackhole reference's style (raw `TTI_` or SFPI); when `SFPI_MODE=true` the optimizer reimplements the working raw-`TTI_` kernel in SFPI and keeps it only if it generates no more instructions than the intrinsics. Pass `SFPI_MODE` to the orchestrator.

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

Set up an isolated worktree so all code changes happen on a dedicated branch based on `origin/main`. The codegen infrastructure (agents, scripts, references, config) is symlinked into the worktree from the current branch — that is why we can base the fix on a clean `origin/main` (clean PR diff) yet still read the codegen playbooks/skills that only live on the feature branch.

```bash
source codegen/scripts/setup_worktree.sh
setup_worktree {TASK_ID}
# Exports: WORKTREE_DIR, WORKTREE_BRANCH
```

After this step:
- `WORKTREE_DIR` — absolute path to the worktree on **durable disk** (e.g., `$HOME/.codegen/worktrees/issue-123-v1`), removed after the run (Step 4). The directory carries the branch version, so concurrent runs never collide. Override the parent with `CODEGEN_WORKTREE_ROOT` (default `$HOME/.codegen/worktrees`). Durable rather than `/tmp` so a reboot or crash mid-run doesn't lose an in-flight checkout — finished work is preserved as the commit + patch, not the worktree.
- `WORKTREE_BRANCH` — the branch name (e.g., `llk_code_gen/issue-123-v1`)
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
- `SFPI_MODE` (`true`/`false` from Step 1)
- `WORKTREE_DIR`, `WORKTREE_BRANCH`

### Solve Issue

Route by task type and by `len(TARGET_ARCHES)`:

| Architecture(s) | Task Type | Orchestrator | Arch input |
|-----------------|-----------|--------------|------------|
| single (any of blackhole / quasar / wormhole) | issue fix | `codegen/agents/issue-solver/orchestrator.md` | `TARGET_ARCH` |
| **multiple** (e.g. `blackhole + wormhole`) | issue fix | `codegen/agents/issue-solver/orchestrator-multi.md` | `TARGET_ARCHES` (JSON array) |
| **quasar** | new kernel | `codegen/agents/quasar/orchestrator.md` | — |
| **blackhole** | new kernel | Not yet supported — inform the user | — |

The single-arch path (`orchestrator.md`) is **unchanged** from before — today's callers keep working bit-for-bit. The multi-arch path (`orchestrator-multi.md`) creates **one dashboard run**, runs one shared analyzer, one shared fixer, and one tester that executes each selected architecture in sequence. This prevents each arch from inventing its own API shape for the same conceptual change.

Pass **all** fetched issue context verbatim to the selected orchestrator: `ISSUE_NUMBER`, `ISSUE_TITLE`, `ISSUE_BODY`, `ISSUE_LABELS`, `ISSUE_COMMENTS`. Never summarize or alter any of these fields — agents need the raw content to parse error messages, stack traces, and reproduction steps.

Also pass:
- `TARGET_ARCH` (single-arch path) **or** `TARGET_ARCHES` (multi-arch path) — never both.
- `WORKTREE_DIR` — the absolute path to the worktree where agents must make all code changes. For multi-arch issues the worktree is shared and one fixer owns the combined change across every selected architecture.
- `WORKTREE_BRANCH` — the branch name for commits and PRs.

---

## Step 4: Preserve & Cleanup

By the time the orchestrator returns, the fix is already preserved **without the
worktree** — its Step 6 **commits the fix locally to `WORKTREE_BRANCH`** (never
pushes) and archives an apply-able `generated.patch` plus its `base_commit`
beside `run.json` in the durable `LOG_DIR`.

So after the run, **remove the worktree to reclaim disk** (~400 MB/run). That is
the default (`CODEGEN_KEEP_WORKTREE=false`); set it to `true` only when you want
to keep the live checkout around for inspection.

```bash
source codegen/scripts/setup_worktree.sh
cleanup_worktree {TASK_ID}          # removes ONLY this run's worktree (safe under concurrency)
./codegen/scripts/setup_worktree.sh prune 14   # GC worktrees left behind by crashed runs (>14d)
```

What persists after cleanup (all tiny — none of it in the removed worktree):
- `generated.patch` + `base_commit` in `LOG_DIR`. `LOG_DIR` is `${CODEGEN_LOGS_ROOT}/<arch>_issue_solver/<run_id>/`, where `CODEGEN_LOGS_ROOT` is the shared dashboard tree `/proj_sw/user_dev/llk_code_gen` **when it exists**, otherwise an **in-repo, gitignored** `codegen/logs/` in the main checkout. Set `CODEGEN_LOGS_ROOT` to force a location.
- The local fix commit on `WORKTREE_BRANCH` (git objects live in the shared local `.git`, not the worktree).

Recovering a run's work later:
- `git checkout <base_commit> && git apply <LOG_DIR>/generated.patch` — from the log dir, independent of repo state.
- `git worktree add <path> <WORKTREE_BRANCH>` — re-materialize the fix from the branch.

**Pushing / PR creation is a separate, explicit action** (still requires the
user's go-ahead). When `CREATE_PR=yes`, push `WORKTREE_BRANCH` and open the PR
only after the user confirms.

## Running multiple issue-solvers concurrently

The mechanism is concurrency-safe — launch as many as the machine can handle:

- Each run gets a **unique branch and directory** (`llk_code_gen/<task>-v<N>` +
  `.../<task>-v<N>`); `setup_worktree` reserves the version under a `flock`, so
  even two runs of the *same* issue never collide.
- Fixes are committed to **separate branches**, so concurrent local commits
  never touch each other.
- Simulator access is serialized per-arch by `.claude/scripts/run_test.sh` via
  `/tmp/tt-llk-test-<arch>.lock`, so parallel runs compile in parallel and only
  queue at the (single) simulator step.

Launch pattern (mirrors `batch_generate.sh` for kernels): run one
`claude -p "solve issue #<N> ..."` per issue, passing every input the Startup
Contract needs (`TEST_BACKEND`, arch, etc.) in the prompt so no run blocks on an
interactive question. Optionally cap parallelism with a job limiter.

---

## Orchestrators

Three flows: kernel generation (arch-specific), single-arch issue solving, and multi-arch issue solving (one coordinated multi-arch run).

| Flow | Orchestrator | Agents | Notes |
|------|--------------|--------|-------|
| Kernel gen | `codegen/agents/quasar/orchestrator.md` | `codegen/agents/quasar/llk-*.md` | Quasar only today. Unaffected by multi-arch issue-solver work. |
| Issue solver (single-arch) | `codegen/agents/issue-solver/orchestrator.md` | `codegen/agents/issue-solver/*.md` | Used when `len(TARGET_ARCHES) == 1`. Parameterized by `TARGET_ARCH` — see `codegen/references/arch-profiles.md`. |
| Issue solver (multi-arch) | `codegen/agents/issue-solver/orchestrator-multi.md` | same `codegen/agents/issue-solver/*.md` agents, run once with `TARGET_ARCHES` | Used when `len(TARGET_ARCHES) > 1`. One analyzer, one fixer, one tester, one dashboard run, one worktree, one branch, one optional PR. |
