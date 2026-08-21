# LLK CodeGen

## Execution Policy:

Run every step synchronously in the foreground. Never set `run_in_background` on a Bash or Agent call (worktree setup, the orchestrator, `run_test.sh`), and never end your turn while a step is still running — the tool's synchronous return is your wait. `run_test.sh` is a single blocking call that returns a terminal code (0/1/2/3/5); it has no resume loop. Pass the Bash-tool maximum `timeout: 600000` as a backstop — the run bounds itself via hang detection.

## Git Policy:

This router uses git READ-ONLY (`rev-parse`, `log`, `status`, `diff`, `show`) —
**no push, commit, checkout, reset, etc.** The one exception is worktree
lifecycle: Step 2 creates the worktree/branch and Step 4 removes it. The fix
**commit** and `generated.patch` are produced by the orchestrator during its run
(Step 3), not by this router — by Step 4 they already exist on `WORKTREE_BRANCH`.
Push/PR is a separate, user-confirmed action (`CREATE_PR=yes`).

When `CODEGEN_NO_PUSH=1`, the run is an infrastructure/audit run. It must still
create and commit its isolated local feature branch, but it must never push,
create or modify a PR, or dispatch a remote review, regardless of prompt text.
Treat `CREATE_PR` as `no`; the local branch and `generated.patch` are the output.

---

## Orchestrators

Four flows: kernel generation (arch-specific), single-arch issue solving, multi-arch issue solving (one coordinated multi-arch run), and a review round on an already-open PR.

| Flow | Orchestrator | Agents | Notes |
|------|--------------|--------|-------|
| Kernel gen | `codegen/agents/quasar/orchestrator.md` | `codegen/agents/quasar/llk-*.md` | Quasar only today. Unaffected by multi-arch issue-solver work. |
| Issue solver (single-arch) | `codegen/agents/issue-solver/orchestrator.md` | `codegen/agents/issue-solver/*.md` | Used when `len(TARGET_ARCHES) == 1`. Parameterized by `TARGET_ARCH` — see `codegen/references/arch-profiles.md`. |
| Issue solver (multi-arch) | `codegen/agents/issue-solver/orchestrator-multi.md` | same `codegen/agents/issue-solver/*.md` agents, run once with `TARGET_ARCHES` | Used when `len(TARGET_ARCHES) > 1`. One analyzer, one fixer, one tester, one dashboard run, one worktree, one branch, one optional PR. |
| Review round | `codegen/agents/issue-solver/review/orchestrator.md` | `review/addresser.md` plus the same `tester.md` / `metal-tester.md` / `reviewer.md` / `perf-tester.md` | Addresses the review comments on an **already-open** PR. No analyze stage — scope, fix layer, and verification route are inherited from the solve that produced the PR. |

## Step 1: Classify the Request

Determine the request type and extract a **TASK_ID** for worktree naming:

### Generate Kernel (direct request)

When a user asks to **"generate {kernel} for {target_arch}"**:
- `REQUEST_TYPE` = `generate`
- `TARGET_ARCH` = the requested architecture (default: **quasar**)
- `KERNEL_NAME` = the kernel to generate
- `TASK_ID` = `generated-{KERNEL_NAME}-{TARGET_ARCH}` (e.g., `generated-gelu-quasar`)
- `SFPI_MODE` = `true` if the user **explicitly** asked for an SFPI version (phrases like "as SFPI", "sfpi version", "in sfpi", "write it in sfpi"); otherwise `false`
- `QSR_SIM_BACKEND` = the inherited environment value for Quasar (`emu` by
  default, or `vcs`). Validate it without prompting. Dashboard-scheduled runs
  always provide this value and the matching UMD paths.

### Address Review Comments on an Open PR

When a user asks to **address the review comments on a pull request** (e.g.,
"address review comments on PR #51772", "resolve the review feedback on PR 123"):
- `REQUEST_TYPE` = `review`
- `PR_NUMBER` = the pull-request number
- `TASK_ID` = `pr-{PR_NUMBER}-review` (e.g., `pr-51772-review`; `setup_worktree`
  appends its own `-v<N>`, so repeat runs on the same PR never collide)

**Do not** load the issue with `gh` for this request type. A review round has no
GitHub credentials by design — every input is pre-seeded. Skip straight to Step 2;
`execute_step_seed_review_state` (Step 3) reads the issue text, target arches, and
verification route out of the solve run that produced the PR.

This request type requires `CODEGEN_SOURCE_RUN_DIR`, `CODEGEN_REVIEW_INPUT`, and
`CODEGEN_PR_NUMBER` in the environment. If any is missing, stop and report that —
do not fall back to solving the issue.

### Solve a GitHub Issue

When a user references a **GitHub issue** (e.g., "solve issue #123", "fix #456", "work on issue 789"):
- `REQUEST_TYPE` = `issue`
- `TASK_ID` = `issue-{ISSUE_NUMBER}` (e.g., `issue-123`)

Then load **all** issue data — title, body, comments, and labels. For a frozen
regression run (`CODEGEN_ISSUE_SNAPSHOT` is set), use:

```bash
python codegen/scripts/load_issue.py {number}
```

Otherwise, preserve the existing live issue flow:

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

For `REQUEST_TYPE=generate` on quasar, put the run in motion before creating the worktree:

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_begin_setup {kernel} {target_arch} "/proj_sw/user_dev/${USER}/llk_code_gen"
# Echoes LOG_DIR, RUN_ID, START_TIME — carry these to Step 3.
```

Set up an isolated worktree so all code changes happen on a dedicated branch
based on `CODEGEN_BASE_COMMIT` when set, or `origin/main` otherwise.

For `REQUEST_TYPE=review` the dashboard sets `CODEGEN_BASE_COMMIT` to the PR's
head commit and has already fetched it, so the worktree comes up **as the PR
stands today** and the round's commit fast-forwards that branch.

```bash
source codegen/scripts/setup_worktree.sh
setup_worktree {TASK_ID}
# Exports: WORKTREE_DIR, WORKTREE_BRANCH
```

For `REQUEST_TYPE=generate` on quasar, mark setup finished (use the `LOG_DIR` from `execute_step_begin_setup`):

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_setup_ready {log_dir}
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

**Mandatory:** before invoking the orchestrator, write its startup parameters
via `state.py --worktree-dir` — do not hand-construct the state file path
(`--worktree-dir` resolves it the same way for every caller):
```bash
python codegen/scripts/state.py --worktree-dir "{worktree_dir}" set KERNEL_NAME     "{kernel}"
python codegen/scripts/state.py --worktree-dir "{worktree_dir}" set TARGET_ARCH     "{target_arch}"
python codegen/scripts/state.py --worktree-dir "{worktree_dir}" set SFPI_MODE       "{SFPI_MODE}" --json
python codegen/scripts/state.py --worktree-dir "{worktree_dir}" set QSR_SIM_BACKEND "{emu|vcs}"
python codegen/scripts/state.py --worktree-dir "{worktree_dir}" set WORKTREE_BRANCH "{worktree_branch}"
python codegen/scripts/state.py --worktree-dir "{worktree_dir}" set LOG_DIR_BASE    "/proj_sw/user_dev/${USER}/llk_code_gen"
# From execute_step_begin_setup (Step 2) so the orchestrator reuses the same run identity:
python codegen/scripts/state.py --worktree-dir "{worktree_dir}" set LOG_DIR    "{log_dir}"
python codegen/scripts/state.py --worktree-dir "{worktree_dir}" set RUN_ID     "{run_id}"
python codegen/scripts/state.py --worktree-dir "{worktree_dir}" set START_TIME "{start_time}"
```
Optional per-run override flags — set the same way (`state.py --worktree-dir … set <FLAG> true`) only when the request asks for them:
- `LOCK_TESTS` — the tester runs test-locked: it treats the existing test as the immutable source of truth, authors or modifies no test, and only runs it and debugs the kernel; the writer→tester→refiner loop is otherwise unchanged.
- `REMOVE_TESTS` — the orchestrator's Step 2c git-removes-and-commits the op's dedicated test files on the worktree branch, then the tester authors the test fresh from the analysis spec after writing the kernel; overrides `LOCK_TESTS`. Never `rm` the files in the prompt; set this flag and let the orchestrator do it.
- `HIDE_EXISTING_KERNEL` — the orchestrator's Step 2b git-removes-and-commits every layer of the target op's existing implementation on the worktree branch — the metal LLK-API wrapper, the tt-llk lib impl anywhere under the arch tree (`common/inc/sfpu/`, `common/inc/experimental/`, or any other subfolder), and the compute-level API entry point (`tt_metal/hw/inc/api/compute/*/{op}.h`) — so it regenerates blind. Only the metal LLK-API dest (`GENERATED_KERNEL`) is written back, by the writer; the hidden tt-llk lib impl and compute-level entry point get no new version. Never `rm` the files in the prompt; set this flag and let the orchestrator do it.

Then invoke the orchestrator, telling it only `WORKTREE_DIR={worktree_dir}` —
it reads everything else back the same way, via `state.py --worktree-dir`.

### Address Review Comments (`REQUEST_TYPE` = `review`)

One helper writes every bootstrap key, reading them out of the solve run that
produced the PR — do not hand-write them:

```bash
source codegen/scripts/issue_solver/orchestrator_steps.sh
execute_step_seed_review_state "{worktree_dir}"
python codegen/scripts/state.py --worktree-dir "{worktree_dir}" set WORKTREE_BRANCH "{worktree_branch}"
```

Stop on a `REJECT:`. Then invoke
`codegen/agents/issue-solver/review/orchestrator.md`, telling it only
`WORKTREE_DIR={worktree_dir}`.

### Solve Issue (`REQUEST_TYPE` = `issue`)

Route by task type and by `len(TARGET_ARCHES)`:

| Architecture(s) | Task Type | Orchestrator | Arch input |
|-----------------|-----------|--------------|------------|
| single (any of blackhole / quasar / wormhole) | issue fix | `codegen/agents/issue-solver/orchestrator.md` | `TARGET_ARCH` |
| **multiple** (e.g. `blackhole + wormhole`) | issue fix | `codegen/agents/issue-solver/orchestrator-multi.md` | `TARGET_ARCHES` (JSON array) |
| Any | Generate Kernel | NOT SUPPORTED | - |

(`REQUEST_TYPE=review` does not use this table — it always routes to
`codegen/agents/issue-solver/review/orchestrator.md`, which reads `RUN_MODE` from
the seeded state.)

**Mandatory:** before invoking the orchestrator, seed its inputs via
`state.py --worktree-dir` (do not pass them in the prompt). `RUN_MODE` and the
arch key are chosen by `len(TARGET_ARCHES)` — the same test that picks the
orchestrator above:
```bash
WT="{worktree_dir}"; S=codegen/scripts/state.py
python $S --worktree-dir "$WT" set ISSUE_NUMBER     "{issue_number}"
python $S --worktree-dir "$WT" set ISSUE_TITLE      "{issue_title}"
python $S --worktree-dir "$WT" set ISSUE_BODY       "{issue_body}"          # verbatim
python $S --worktree-dir "$WT" set ISSUE_LABELS     "{label1,label2,...}"   # comma-joined string
python $S --worktree-dir "$WT" set ISSUE_COMMENTS   "{issue_comments}"      # verbatim
python $S --worktree-dir "$WT" set ISSUE_URL        "{issue_url}"           # or "" (setup_run derives the default)
python $S --worktree-dir "$WT" set WORKTREE_BRANCH  "{worktree_branch}"
python $S --worktree-dir "$WT" set TEST_BACKEND     "{local|ttsim}"
python $S --worktree-dir "$WT" set CREATE_LOCAL_BRANCH "{yes|no}"
python $S --worktree-dir "$WT" set CREATE_PR        "{yes|no}"
# single-arch (len(TARGET_ARCHES)==1) → orchestrator.md:
python $S --worktree-dir "$WT" set RUN_MODE      single
python $S --worktree-dir "$WT" set TARGET_ARCH   "{target_arch}"
python $S --worktree-dir "$WT" set TTSIM_SO_PATH "{path}"                    # only when TEST_BACKEND=ttsim
# multi-arch (len>1) → orchestrator-multi.md:
python $S --worktree-dir "$WT" set RUN_MODE       multi
python $S --worktree-dir "$WT" set TARGET_ARCHES  '["{arch}","..."]'         # JSON array string
python $S --worktree-dir "$WT" set TTSIM_SO_PATHS '{"{arch}":"{path}",...}'  # only when TEST_BACKEND=ttsim
```
Then invoke the selected orchestrator, telling it only `WORKTREE_DIR={worktree_dir}`
— it reads everything else back via `state.py`.

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
- `generated.patch` + `base_commit` in `LOG_DIR` — always present, always sufficient on its own.
- The local fix commit on `WORKTREE_BRANCH` — issue-solver only. Quasar kernel-gen never commits the generated kernel (writer/optimizer/prettifier leave it as an uncommitted working-tree diff, captured only by `generated.patch`); with `HIDE_EXISTING_KERNEL=true` the branch instead carries commits that *delete* the target op's prior files, so checking out that branch alone yields a regression, not a recovery.

Recovering a run's work later:
- `git checkout <base_commit> && git apply <LOG_DIR>/generated.patch` — works for every flow, independent of repo state. Use this for kernel-gen.
- `git worktree add <path> <WORKTREE_BRANCH>` — issue-solver only, re-materializes its fix commit. For quasar kernel-gen this recovers nothing (or, under `HIDE_EXISTING_KERNEL`, something actively worse than nothing).

**Pushing / PR creation is a separate, explicit action** (still requires the
user's go-ahead). Perform when `CREATE_PR=yes`, push `WORKTREE_BRANCH` and open the PR
only after the user confirms.
`CODEGEN_NO_PUSH=1` overrides both of those signals: keep the branch local and do
not perform any GitHub write.

## Running multiple issue-solvers concurrently

The mechanism is concurrency-safe — launch as many as the machine can handle:

- Each run gets a **unique branch and directory** (`llk_code_gen/<task>-v<N>` +
  `.../<task>-v<N>`); `setup_worktree` reserves the version under a `flock`, so
  even two runs of the *same* issue never collide.
- Fixes are committed to **separate branches**, so concurrent local commits
  never touch each other.
- Device access is serialized by `.claude/scripts/run_test.sh`. Local devices
  use `/tmp/tt-llk-test.lock`; Quasar uses `QSR_AETHER_LOCK` when configured so
  separate compute hosts also serialize against the same remote Aether
  resource. Parallel runs compile lock-free and queue only at the
  `simulate`/`run` step; whoever holds the lock rebuilds if a peer's compile
  invalidated the build cache.

Launch pattern (mirrors `batch_generate.sh` for kernels): run one
`claude -p "solve issue #<N> ..."` per issue, passing every input the Startup
Contract needs (`TEST_BACKEND`, arch, etc.) in the prompt so no run blocks on an
interactive question. Optionally cap parallelism with a job limiter.

---

- If you are NOT EXPLICITLY INSTRUCTED TO ASK a QUESTION then DON'T.
