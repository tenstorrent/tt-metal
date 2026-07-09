# LLK CodeGen

## Git Policy:

This router uses git READ-ONLY (`rev-parse`, `log`, `status`, `diff`, `show`) —
**no push, commit, checkout, reset, etc.** The one exception is worktree
lifecycle: Step 2 creates the worktree/branch and Step 4 removes it. The fix
**commit** and `generated.patch` are produced by the orchestrator during its run
(Step 3), not by this router — by Step 4 they already exist on `WORKTREE_BRANCH`.
Push/PR is a separate, user-confirmed action (`CREATE_PR=yes`).

---

## Orchestrators

Three flows: kernel generation (arch-specific), single-arch issue solving, and multi-arch issue solving (one coordinated multi-arch run).

| Flow | Orchestrator | Agents | Notes |
|------|--------------|--------|-------|
| Kernel gen | `codegen/agents/quasar/orchestrator.md` | `codegen/agents/quasar/llk-*.md` | Quasar only today. Unaffected by multi-arch issue-solver work. |
| Issue solver (single-arch) | `codegen/agents/issue-solver/orchestrator.md` | `codegen/agents/issue-solver/*.md` | Used when `len(TARGET_ARCHES) == 1`. Parameterized by `TARGET_ARCH` — see `codegen/references/arch-profiles.md`. |
| Issue solver (multi-arch) | `codegen/agents/issue-solver/orchestrator-multi.md` | same `codegen/agents/issue-solver/*.md` agents, run once with `TARGET_ARCHES` | Used when `len(TARGET_ARCHES) > 1`. One analyzer, one fixer, one tester, one dashboard run, one worktree, one branch, one optional PR. |

## Step 1: Classify the Request

Determine the request type and extract a **TASK_ID** for worktree naming:

### Generate Kernel (direct request)

When a user asks to **"generate {kernel} for {target_arch}"**:
- `REQUEST_TYPE` = `generate`
- `TARGET_ARCH` = the requested architecture (default: **quasar**)
- `KERNEL_NAME` = the kernel to generate
- `TASK_ID` = `generated-{KERNEL_NAME}-{TARGET_ARCH}` (e.g., `generated-gelu-quasar`)
- `SFPI_MODE` = `true` if the user **explicitly** asked for an SFPI version (phrases like "as SFPI", "sfpi version", "in sfpi", "write it in sfpi"); otherwise `false`

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

Set up an isolated worktree so all code changes happen on a dedicated branch based on `origin/main`.

```bash
source codegen/scripts/setup_worktree.sh
setup_worktree {TASK_ID}
# Exports: WORKTREE_DIR, WORKTREE_BRANCH
```

Exports two variables, passed to the orchestrator in Step 3:
- `WORKTREE_DIR`
- `WORKTREE_BRANCH`

---

## Step 3: Route to Orchestrator

### Generate Kernel (`REQUEST_TYPE` = `generate`)

| Architecture | Orchestrator |
|-------------|-------------|
| **quasar** | `codegen/agents/quasar/orchestrator.md` |
| **wormhole/blackhole** | Not yet supported — inform the user |

**Mandatory:** pass the orchestrator this exact JSON structure:
```json
{
  "KERNEL_NAME": "{kernel}",
  "TARGET_ARCH": "{target_arch}",
  "SFPI_MODE": "{SFPI_MODE}",
  "WORKTREE_DIR": "{worktree_dir}",
  "WORKTREE_BRANCH": "{worktree_branch}"
}
```

### Solve Issue (`REQUEST_TYPE` = `issue`)

Route by task type and by `len(TARGET_ARCHES)`:

| Architecture(s) | Task Type | Orchestrator | Arch input |
|-----------------|-----------|--------------|------------|
| single (any of blackhole / quasar / wormhole) | issue fix | `codegen/agents/issue-solver/orchestrator.md` | `TARGET_ARCH` |
| **multiple** (e.g. `blackhole + wormhole`) | issue fix | `codegen/agents/issue-solver/orchestrator-multi.md` | `TARGET_ARCHES` (JSON array) |
| Any | Generate Kernel | NOT SUPPORTED | - |

**Mandatory:** pass the selected orchestrator this exact JSON structure:
```json
{
  "ISSUE_NUMBER": "{issue_number}",
  "ISSUE_TITLE": "{issue_title}",
  "ISSUE_BODY": "{issue_body}",
  "ISSUE_LABELS": ["{label}", "..."],
  "ISSUE_COMMENTS": ["{comment}", "..."],
  "TARGET_ARCH": "{target_arch}",         // single-arch path ONLY — omit if TARGET_ARCHES is present
  "TARGET_ARCHES": ["{arch}", "..."],     // multi-arch path ONLY — omit if TARGET_ARCH is present; never both
  "WORKTREE_DIR": "{worktree_dir}",       // multi-arch: shared — one fixer owns the combined change across every selected architecture
  "WORKTREE_BRANCH": "{worktree_branch}"  // used for commits and PRs
}
```

---

## Step 4: Preserve & Cleanup

Wait for orchestrator to finish the work.

ONLY then run the following:

The worktree is removed after the run; set `CODEGEN_KEEP_WORKTREE=true` if the user asks to keep it.

```bash
source codegen/scripts/setup_worktree.sh
cleanup_worktree {TASK_ID}          # removes ONLY this run's worktree (safe under concurrency)
./codegen/scripts/setup_worktree.sh prune 14   # GC worktrees left behind by crashed runs (>14d)
```

After the cleanup we are left with:
- `LOG_DIR` is the path the orchestrator set during its run — take the concrete path from the orchestrator's report.
- `generated.patch` + `base_commit` in `LOG_DIR`.
- The local fix commit on `WORKTREE_BRANCH`.

Recovering a run's work later:
- `git checkout <base_commit> && git apply <LOG_DIR>/generated.patch` — from the `LOG_DIR`, independent of repo state.
- `git worktree add <path> <WORKTREE_BRANCH>` — re-materialize the fix from the branch.

**Pushing / PR creation is a separate, explicit action** (still requires the
user's go-ahead). Perform when `CREATE_PR=yes`, push `WORKTREE_BRANCH` and open the PR
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

- If you are NOT EXPLICITLY INSTRUCTED TO ASK a QUESTION then DON'T.
