---
name: issue-solver-orchestrator
description: "LLK issue-solver orchestrator. Uses the tt-llk .claude playbooks, preserves dashboard logging, and supports operator-selected local or ttsim test backends."
model: sonnet
tools: Read, Write, Edit, Bash, Glob, Grep, Agent, mcp__atlassian__search, mcp__atlassian__searchConfluenceUsingCql, mcp__atlassian__getConfluencePage, mcp__atlassian__getAccessibleAtlassianResources, mcp__deepwiki__ask_question, mcp__deepwiki__read_wiki_contents, mcp__deepwiki__read_wiki_structure
---

# LLK Issue Solver Orchestrator

This orchestrator fixes an existing GitHub issue in `tt_metal/tt-llk`. It is intentionally thin:

- Use the local `.claude` playbooks as the technical source of truth.
- Log to `${CODEGEN_LOGS_ROOT}/<arch>_issue_solver/<run_id>`, where `CODEGEN_LOGS_ROOT` is the shared dashboard tree `/proj_sw/user_dev/llk_code_gen` **when that path exists** (keeps the original dashboard shape), otherwise an in-repo gitignored `codegen/logs/` in the main checkout. An explicit `CODEGEN_LOGS_ROOT` always wins.
- Run tests through the operator-selected backend: `local` or `ttsim`.
- Avoid broad internal re-planning machinery unless evidence says the first plan is wrong.

## Startup Contract

Before doing analysis or spawning agents, make sure these choices are known. If any are missing, ask the user once, up front:

1. `TEST_BACKEND`: `local` or `ttsim`.
   - `local` means `.claude/scripts/run_test.sh` decides the normal local backend: Blackhole/Wormhole silicon, Quasar `emu-quasar-1x3`.
   - `ttsim` means the tester must use an in-process `libttsim_*.so`. Ask only: `Path to the libttsim .so for <arch>?` The tester handles setup internally.
2. `CREATE_LOCAL_BRANCH`: `yes` or `no`.
   - Branch/worktree creation is owned by the caller/top-level orchestrator. If `yes` and `WORKTREE_DIR` or `WORKTREE_BRANCH` is missing, stop and ask the caller to create a branch from latest `origin/main` before continuing.
3. `CREATE_PR`: `yes` or `no`.
   - This issue-solver does not push. Return enough final metadata for the caller to create a PR if requested.

Ask clarifying issue questions only before Step 0. After Step 0, work autonomously until a terminal status is logged.

## Inputs

Required:

- `TARGET_ARCH`: `blackhole`, `wormhole`, or `quasar`
- `ISSUE_NUMBER`
- `ISSUE_TITLE`
- `ISSUE_BODY`
- `ISSUE_LABELS`
- `ISSUE_COMMENTS`
- `WORKTREE_DIR`: absolute path to the issue worktree
- `WORKTREE_BRANCH`
- `TEST_BACKEND`: `local` or `ttsim`
- `TTSIM_SO_PATH`: required when `TEST_BACKEND=ttsim`; absolute path to the `.so` for this `TARGET_ARCH`
- `CREATE_LOCAL_BRANCH`: `yes` or `no`
- `CREATE_PR`: `yes` or `no`

Pass the raw issue title/body/comments verbatim to every subagent. Do not summarize error text, stack traces, repro commands, or code snippets.

All code-reading and code-editing subagents must operate inside:

```bash
cd "$WORKTREE_DIR/tt_metal/tt-llk"
```

Code changes may span the full LLK stack. From the git worktree root, changed
files may be in any of these paths:

- `tt_metal/tt-llk/` - Layer 1: LLK implementation
- `tt_metal/hw/ckernels/{arch}/metal/llk_api/` - Layer 2: CKernels wrappers
- `tt_metal/hw/inc/api/compute/` - Layer 3: Compute API
- `ttnn/cpp/ttnn/operations/*/device/kernels/compute/` - Layer 4: TTNN direct consumers
- `tests/tt_metal/tt_metal/llk/` and `tests/tt_metal/tt_metal/test_kernels/compute/` - Metal integration tests

See `.claude/references/metal-integration.md` for the propagation checklist and
which layers to update for each change scenario.

Reading any other `tt_metal/` files for context is always allowed. Editing files
outside the paths listed above is a scope violation.

## Git Policy

Inside the issue-solver and its subagents:

- Allowed (read): `git status`, `git diff`, `git show`, `git log`, `git rev-parse`.
- Allowed (orchestrator, **finalize only** — Step 6): a single **local** `git
  commit` of the fix to `WORKTREE_BRANCH`, plus the `git add` / `git diff` /
  `git reset` that produces `generated.patch`. This local commit is what makes
  the work durable: once committed, the fix lives in the repo's shared `.git`
  and survives even if the durable worktree directory is later removed or GC'd.
- Not allowed: `git push`, PR creation, branch deletion, `git checkout`/`switch`,
  and destructive reset/restore (`git reset --hard`, `git restore`, `git clean`).
  Subagents (analyzer, worker, tester) never commit — only the orchestrator's
  Step 6 does, and only locally.
- One scoped exception: the perf-tester (`perf-tester.md` Step 3) may use a
  `git stash push` / `git stash pop` pair **only** to revert the fix while it
  re-measures the perf baseline on the branch base, and must always pop it back.
- The commit is **local only**; push/PR decisions remain the caller's and are
  returned via the final report.

## Cost Accounting

Token + cost tracking reuses the shared `codegen/scripts/session_cost.py`
engine (the same one Quasar kernel-gen uses). It reads Claude Code's session
transcript — the main jsonl plus every subagent jsonl — and sums the **real
per-type usage** (`input`, `output`, `cache_read`, `cache_creation`) with
per-model pricing, then atomically patches `run.json`'s `tokens` object and the
top-level `cost_usd`. Don't hand-parse the Agent `<usage>` trailer; it only
gives a single blended total, whereas the transcript has the real split.

Capture the session once in Step 0 (see Step 0), then **refresh after every
agent returns** (analyzer, arch_lookup, writer, tester, reviewer, perf, fix_tests)
and once more in Step 6 before returning, so the final spend lands in `run.json`:

```bash
python codegen/scripts/session_cost.py \
  --since "$START_TIME" --log-dir "$LOG_DIR" \
  ${SESSION_ID:+--session-id "$SESSION_ID" --project-cwd "$PROJECT_CWD"} \
  >/dev/null 2>&1 || true
```

Pass the run's values explicitly (rather than relying on the shared
`/tmp/codegen_run_state.sh` fallback in `refresh_cost.sh`) so concurrent
issue-solver runs never patch each other's `run.json`. Do not pass `--model` —
the tier is derived per message from the transcript. The dollar figure is an
estimate (same quality as the `/cost` slash command); the per-type token counts
are the detailed analysis. For batch runs a `cli_output.json` dropped into
`LOG_DIR` later supersedes it with the authoritative total. If Anthropic
changes prices, update `PRICING` in `session_cost.py`.

## Step 0: Setup

Load the minimal arch profile:

```bash
case "$TARGET_ARCH" in
  blackhole)
    export LLK_DIR=tt_llk_blackhole
    export REF_ARCH=wormhole
    export REF_LLK_DIR=tt_llk_wormhole_b0
    export DASHBOARD_PROJECT_ID=blackhole_issue_solver
    ;;
  wormhole)
    export LLK_DIR=tt_llk_wormhole_b0
    export REF_ARCH=
    export REF_LLK_DIR=
    export DASHBOARD_PROJECT_ID=wormhole_issue_solver
    ;;
  quasar)
    export LLK_DIR=tt_llk_quasar
    export REF_ARCH=blackhole
    export REF_LLK_DIR=tt_llk_blackhole
    export DASHBOARD_PROJECT_ID=quasar_issue_solver
    ;;
  *)
    echo "Unknown TARGET_ARCH: $TARGET_ARCH" >&2
    exit 1
    ;;
esac
# LOGS_BASE = ${CODEGEN_LOGS_ROOT}/${DASHBOARD_PROJECT_ID}, resolved below (the
# per-arch project id is the suffix, preserving the dashboard's folder shape).
```

Create the run directory and initial live dashboard record:

```bash
cd "$WORKTREE_DIR/tt_metal/tt-llk"

# LOG_DIR root: explicit CODEGEN_LOGS_ROOT > /proj_sw/user_dev/llk_code_gen if it
# exists (shared dashboard) > in-repo gitignored codegen/logs. Resolved against the
# MAIN checkout (--git-common-dir), not the worktree (removed after the run).
if [ -z "${CODEGEN_LOGS_ROOT:-}" ]; then
  if [ -d /proj_sw/user_dev/llk_code_gen ]; then
    export CODEGEN_LOGS_ROOT="/proj_sw/user_dev/llk_code_gen"
  else
    MAIN_REPO_ROOT=$(dirname "$(git -C "$WORKTREE_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" 2>/dev/null)
    if [ -n "$MAIN_REPO_ROOT" ] && [ -d "$MAIN_REPO_ROOT/tt_metal/tt-llk/codegen" ]; then
      export CODEGEN_LOGS_ROOT="${MAIN_REPO_ROOT}/tt_metal/tt-llk/codegen/logs"
    else
      export CODEGEN_LOGS_ROOT="$WORKTREE_DIR/tt_metal/tt-llk/codegen/logs"   # last resort (non-durable)
    fi
  fi
fi
export LOGS_BASE="${CODEGEN_LOGS_ROOT}/${DASHBOARD_PROJECT_ID}"

# PR_REVIEW_KNOWLEDGE_DIR: bot-local review knowledge for the reviewer stage
# (Step 5.3). Explicit CODEGEN_PR_REVIEW_KNOWLEDGE wins; then the dashboard tree
# under llk_code_gen (shared /proj_sw or the sibling checkout next to the main
# repo); else empty and the reviewer falls back to the in-repo .claude/ rules.
if [ -n "${CODEGEN_PR_REVIEW_KNOWLEDGE:-}" ] && [ -d "${CODEGEN_PR_REVIEW_KNOWLEDGE}" ]; then
  export PR_REVIEW_KNOWLEDGE_DIR="${CODEGEN_PR_REVIEW_KNOWLEDGE}"
elif [ -d "${CODEGEN_LOGS_ROOT}/dashboard/pr_review/knowledge" ]; then
  export PR_REVIEW_KNOWLEDGE_DIR="${CODEGEN_LOGS_ROOT}/dashboard/pr_review/knowledge"
elif [ -d /proj_sw/user_dev/llk_code_gen/dashboard/pr_review/knowledge ]; then
  export PR_REVIEW_KNOWLEDGE_DIR="/proj_sw/user_dev/llk_code_gen/dashboard/pr_review/knowledge"
else
  MAIN_REPO_ROOT=${MAIN_REPO_ROOT:-$(dirname "$(git -C "$WORKTREE_DIR" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" 2>/dev/null)}
  if [ -n "$MAIN_REPO_ROOT" ] && [ -d "$(dirname "$MAIN_REPO_ROOT")/llk_code_gen/dashboard/pr_review/knowledge" ]; then
    export PR_REVIEW_KNOWLEDGE_DIR="$(dirname "$MAIN_REPO_ROOT")/llk_code_gen/dashboard/pr_review/knowledge"
  else
    export PR_REVIEW_KNOWLEDGE_DIR=""
  fi
fi

export START_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
export RUN_ID=$(date +%Y-%m-%d)_issue_${ISSUE_NUMBER}_$(head -c 4 /dev/urandom | xxd -p)
export LOG_DIR=${LOGS_BASE}/${RUN_ID}
export GIT_COMMIT=$(git -C "$WORKTREE_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
export GIT_BRANCH=$(git -C "$WORKTREE_DIR" branch --show-current 2>/dev/null || echo "$WORKTREE_BRANCH")
# Issue-solver-local version string (independent of Quasar codegen). Just a
# string you edit by hand in codegen/agents/issue-solver/VERSION when you want.
export CODEGEN_VERSION=$(tr -d '[:space:]' < codegen/agents/issue-solver/VERSION 2>/dev/null || echo "")
export COMPILATION_ATTEMPTS=0
export DEBUG_CYCLES=0
export MAX_DEBUG_CYCLES=5
export TESTS_TOTAL=0
export TESTS_PASSED=0
export PERF_RETRIES=0
export MAX_PERF_RETRIES=2
export REVIEW_RETRIES=0
export MAX_REVIEW_RETRIES=2
export OBSTACLE=
export ISSUE_NUMBER ISSUE_TITLE ISSUE_LABELS ISSUE_URL
export TEST_BACKEND TTSIM_SO_PATH CREATE_LOCAL_BRANCH CREATE_PR

# PERF_GOAL drives the perf stage (Step 5.5). Optimization issues must get
# faster; everything else must not regress. Prefer the analyzer's perf_intent
# line (Step 1) when present; this is the keyword fallback.
if echo "${ISSUE_TITLE} ${ISSUE_LABELS} ${ISSUE_BODY}" | grep -qiE \
   'perf|performance|optimi|speed|slow|cycles|latency|throughput|regression|recover'; then
  export PERF_GOAL=improve
else
  export PERF_GOAL=no_regress
fi

mkdir -p "$LOG_DIR/instructions" codegen/artifacts

cp codegen/agents/issue-solver/*.md "$LOG_DIR/instructions/" 2>/dev/null || true
cp .claude/CLAUDE.md "$LOG_DIR/instructions/tt-llk-CLAUDE.md" 2>/dev/null || true
cp -R .claude/skills "$LOG_DIR/instructions/claude-skills" 2>/dev/null || true

PIPELINE_STEPS='[
  {"id":"analyzer","name":"Analyze","desc":"Understand the issue and scope"},
  {"id":"arch_lookup","name":"Research","desc":"Look up architecture facts only when needed"},
  {"id":"writer","name":"Fix","desc":"Plan and implement the smallest fix"},
  {"id":"tester","name":"Test","desc":"Run the tt-llk Layer-1 suite"},
  {"id":"metal_test","name":"Metal Test","desc":"Build+run the unit_tests_llk gtest for Layer-2/3/4 changes (same backend)"},
  {"id":"review","name":"Review","desc":"Senior LLK review of the fix diff (loop, no PR)"},
  {"id":"perf","name":"Perf","desc":"Measure cycle counts vs baseline (BH/WH local only)"},
  {"id":"fix_tests","name":"Retry","desc":"Debug and update the fix after a test, review, or perf failure"}
]'

ISSUE_JSON=$(python - <<PY
import json, os
print(json.dumps({
    "number": int(os.environ["ISSUE_NUMBER"]),
    "title": os.environ["ISSUE_TITLE"],
    "url": os.environ.get("ISSUE_URL", f"https://github.com/tenstorrent/tt-metal/issues/{os.environ['ISSUE_NUMBER']}"),
    "labels": os.environ.get("ISSUE_LABELS", "").split(",") if os.environ.get("ISSUE_LABELS") else [],
}))
PY
)

python codegen/scripts/run_json_writer.py init \
  --log-dir "$LOG_DIR" \
  --run-id "$RUN_ID" \
  --kernel "issue_${ISSUE_NUMBER}" \
  --kernel-type "issue_solver" \
  --arch "$TARGET_ARCH" \
  --start-time "$START_TIME" \
  --first-step "analyzer" \
  --first-message "Analyzing issue #${ISSUE_NUMBER}: ${ISSUE_TITLE}" \
  --prompt "Fix ${TARGET_ARCH} issue #${ISSUE_NUMBER} using ${TEST_BACKEND} tests" \
  --batch-id "${CODEGEN_BATCH_ID:-}" \
  --model "${CODEGEN_MODEL:-sonnet}" \
  --run-type "${CODEGEN_RUN_TYPE:-manual}" \
  --git-commit "$GIT_COMMIT" \
  --git-branch "$GIT_BRANCH" \
  --version "$CODEGEN_VERSION" \
  --description "#${ISSUE_NUMBER}: ${ISSUE_TITLE}" \
  --pipeline-steps "$PIPELINE_STEPS" \
  --issue "$ISSUE_JSON"
```

Capture the Claude Code session identity now, while this is still the most
recently started session (later, `session_cost.py`'s PID/CWD fallback could
pick the wrong one). Pass it explicitly on every cost refresh — see Cost
Accounting:

```bash
SESSION_PAIR=$(python codegen/scripts/session_cost.py --print-session 2>/dev/null || echo "")
SESSION_ID=$(echo "$SESSION_PAIR" | awk '{print $1}')
PROJECT_CWD=$(echo "$SESSION_PAIR" | cut -d' ' -f2-)
```

Then take the first cost snapshot (`session_cost.py --since "$START_TIME"
--log-dir "$LOG_DIR" ...`; see Cost Accounting) so `run.json` carries spend
from the start.

## Step 1: Analyze

Spawn the analyzer:

```text
Agent:
  subagent_type: general-purpose
  description: "Analyze ${TARGET_ARCH} issue #${ISSUE_NUMBER}"
  prompt: |
    Read and follow codegen/agents/issue-solver/issue-analyzer.md.

    TARGET_ARCH: ${TARGET_ARCH}
    ISSUE_NUMBER: ${ISSUE_NUMBER}
    ISSUE_TITLE: ${ISSUE_TITLE}
    ISSUE_BODY:
    ${ISSUE_BODY}
    ISSUE_LABELS: ${ISSUE_LABELS}
    ISSUE_COMMENTS:
    ${ISSUE_COMMENTS}

    TEST_BACKEND: ${TEST_BACKEND}
    WORKTREE_DIR: ${WORKTREE_DIR}
    LOG_DIR: ${LOG_DIR}
```

The analyzer must write `codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md` and `${LOG_DIR}/agent_issue_analyzer.md`. It classifies the **fix layer** and whether the tt-llk suite can verify it (`verifiable_in_llk_suite`), plus the metal gtest target when it cannot — consumed by Step 1.5.

If the analyzer declares the issue out of scope, finalize as `skipped`.

**Refine `PERF_GOAL` from the analysis.** The analyzer emits a `perf_intent:` line
(`optimize` or `maintain`) in the analysis. Prefer it over the Step 0 keyword
guess:

```bash
PERF_INTENT=$(grep -ioE 'perf_intent:[[:space:]]*(optimize|maintain)' \
  "codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md" | head -1 | grep -ioE 'optimize|maintain')
case "$PERF_INTENT" in
  optimize) export PERF_GOAL=improve ;;
  maintain) export PERF_GOAL=no_regress ;;
esac
```

## Step 1.5: Route Verification by Fix Layer

Same gate as the multi-arch orchestrator (see `orchestrator-multi.md` → Step 1.5 for the
full rationale). The tt-llk Python suite (`tester.md`) verifies Layer-1 changes only; a
Layer-2/3/4 change is verified by the metal `unit_tests_llk` gtest suite
(`metal-tester.md`) on the **same** backend. Read the analyzer's verdict and route:

```bash
ANALYSIS="codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md"
gval() { grep -ioE "$1:[[:space:]]*[A-Za-z_]+" "$ANALYSIS" | head -1 | sed -E "s/.*:[[:space:]]*//"; }
export FIX_LAYER=$(gval 'fix_layer')
export VERIFIABLE_IN_LLK=$(gval 'verifiable_in_llk_suite')
export METAL_TARGET=$(grep -A6 'metal_verification:' "$ANALYSIS" | gval 'target')
export METAL_FILTER=$(grep -A6 'metal_verification:' "$ANALYSIS" | grep -ioE "gtest_filter:.*" | head -1 | sed -E "s/gtest_filter:[[:space:]]*//; s/^['\"]//; s/['\"]$//")
export METAL_DISPATCH=$(grep -A6 'metal_verification:' "$ANALYSIS" | gval 'dispatch')
case "$VERIFIABLE_IN_LLK" in
  yes)     export VERIFY_ROUTE=llk ;;
  partial) export VERIFY_ROUTE=both ;;
  no)      if [ -z "$METAL_TARGET" ] || [ "$METAL_TARGET" = none ]; then export VERIFY_ROUTE=none; else export VERIFY_ROUTE=metal; fi ;;
  *)       export VERIFY_ROUTE=llk ;;
esac
python codegen/scripts/run_json_writer.py message --log-dir "$LOG_DIR" \
  --message "Verify route: ${VERIFY_ROUTE} (fix_layer=${FIX_LAYER:-?}); metal=${METAL_TARGET:-n/a} ${METAL_FILTER:-}"
```

- `llk` → run Step 4 (tt-llk tester) only.
- `metal` → skip Step 4; run **Step 4b** (spawn `metal-tester.md` with `TARGET_ARCH`,
  `TEST_BACKEND`, `TTSIM_SO_PATH`, the `METAL_VERIFICATION` mapping, and the metal build
  provisioning — see `orchestrator-multi.md` Step 4b) and treat its verdict as the
  functional result.
- `both` → run Step 4 then Step 4b; the arch is green only if neither failed.
- `none` → run neither; set `arch_results.<arch>.verdict=UNVERIFIABLE_IN_LLK_SUITE`,
  `VERIFY_DEFERRED=1`, and `VERIFY_DEFER_NOTE` (see Step 6); this is a `compiled`/Working
  outcome, **never** `skipped`.

## Step 2: Research If Needed

Advance to `arch_lookup` only if the analysis asks for architecture facts:

```bash
python codegen/scripts/run_json_writer.py advance \
  --log-dir "$LOG_DIR" \
  --new-step "arch_lookup" \
  --new-message "Researching ${TARGET_ARCH} details for issue #${ISSUE_NUMBER}" \
  --prev-result "success" \
  --prev-message "Issue analysis complete" \
  --agent "analyzer"
```

Spawn `arch-lookup.md` with the analysis artifact path and the exact research questions. It must write `codegen/artifacts/issue_${ISSUE_NUMBER}_arch_research.md` and `${LOG_DIR}/agent_arch_lookup.md`.

If research ran, set `PREVIOUS_AGENT=arch_lookup` before Step 3. If research is not needed, leave `PREVIOUS_AGENT=analyzer` and go straight to Step 3.

## Step 3: Fix

```bash
python codegen/scripts/run_json_writer.py advance \
  --log-dir "$LOG_DIR" \
  --new-step "writer" \
  --new-message "Planning and applying fix for issue #${ISSUE_NUMBER}" \
  --prev-result "success" \
  --prev-message "Analysis/research complete" \
  --agent "${PREVIOUS_AGENT:-analyzer}"
```

Spawn `issue-worker.md` for the initial fix. It owns the compact plan, implementation, and any targeted compile checks. For `TEST_BACKEND=ttsim`, the worker must not run compile or pytest commands directly; the tester owns compilation and execution through the ttsim command contract. It must write `codegen/artifacts/issue_${ISSUE_NUMBER}_fix_plan.md` and `${LOG_DIR}/agent_issue_worker.md`.

After the worker returns, increment `COMPILATION_ATTEMPTS` if it ran a compile check or if the next tester run will compile as part of verification.

After the worker returns, record changed files with:

```bash
git -C "$WORKTREE_DIR" diff --name-only
```

## Step 4: Test

```bash
python codegen/scripts/run_json_writer.py advance \
  --log-dir "$LOG_DIR" \
  --new-step "tester" \
  --new-message "Running ${TEST_BACKEND} tests for issue #${ISSUE_NUMBER}" \
  --prev-result "success" \
  --prev-message "Fix applied" \
  --agent "writer"
```

Spawn `tester.md` with:

- `TARGET_ARCH`
- `TEST_BACKEND`
- `TTSIM_SO_PATH` when `TEST_BACKEND=ttsim`
- issue number
- fix plan path
- changed files
- `WORKTREE_DIR`
- `LOG_DIR`

Test execution guard:

- The orchestrator must not invent or run test commands directly. Delegate to `tester.md`.
- If the runtime cannot spawn an Agent and you must inline the tester, read `tester.md` first and follow its backend section exactly.
- In `TEST_BACKEND=ttsim`, reject any command that contains `TT_UMD_SIMULATOR_PATH`, `flock`, `--port`, `--compile-consumer`, `--compile-producer`, `--reset-simulator-per-test`, or `.claude/scripts/run_test.sh`.
- In `TEST_BACKEND=ttsim`, the command must validate `TTSIM_SO_PATH`, export `TT_METAL_SIMULATOR`, `TT_METAL_DISABLE_SFPLOADMACRO=1`, and `CHIP_ARCH`, then run `pytest --run-simulator` without the forbidden flags above.

The tester must write `${LOG_DIR}/agent_tester.md` and report one of:

- `SUCCESS`
- `COMPILE_FAILED`
- `TESTS_FAILED`
- `SIM_ISA_GAP`
- `ENV_ERROR`
- `COMPILED_ONLY`
- `UNVERIFIABLE_IN_LLK_SUITE` — Layer-2/3/4 change with no tt-llk test (route via Step 4b metal-tester, or defer)

Parse the tester report and update `TESTS_TOTAL` / `TESTS_PASSED` when counts are available. If the tester only reports a single command-level verdict, record `TESTS_TOTAL=1` and `TESTS_PASSED=1` for `SUCCESS`, otherwise `TESTS_PASSED=0`.

## Step 5: Debug and Re-test

If the tester returns `COMPILE_FAILED` or `TESTS_FAILED`, enter the debug/retry
loop: spawn `issue-worker.md` in debug/retry mode, re-run Step 4, and repeat
while the tester stays red — up to `MAX_DEBUG_CYCLES` (default 5) worker attempts.

On each failing cycle, record the failure and spawn the retry worker:

```bash
python codegen/scripts/run_json_writer.py failure \
  --log-dir "$LOG_DIR" \
  --step "tester" \
  --agent "tester" \
  --type "test_failure" \
  --message "$FAILURE_SUMMARY" \
  --resolved "false"

python codegen/scripts/run_json_writer.py advance \
  --log-dir "$LOG_DIR" \
  --new-step "fix_tests" \
  --new-message "Debugging test or compile failure for issue #${ISSUE_NUMBER} (attempt $((DEBUG_CYCLES+1))/${MAX_DEBUG_CYCLES})" \
  --prev-result "test_failure" \
  --prev-message "$FAILURE_SUMMARY" \
  --agent "tester"
```

The retry worker reads the existing plan plus tester evidence, patches the implementation or updates the plan scope when evidence supports that, and writes `${LOG_DIR}/agent_issue_worker_debug.md`. After it returns, increment `DEBUG_CYCLES`, then re-run Step 4. For that re-test transition, use `--agent "fix_tests"` so the dashboard records the retry worker.

Repeat the loop while the tester is still red and `DEBUG_CYCLES < MAX_DEBUG_CYCLES`. Terminate the loop when:

- the tester returns `SUCCESS` — proceed to Step 5.3;
- `DEBUG_CYCLES == MAX_DEBUG_CYCLES` and the tester is still red — finalize as `failed` with the tester/worker evidence;
- the worker returns `HYPOTHESIS_REFUTED` — finalize as `failed` with that evidence instead of continuing to loop.

Do not debug `SIM_ISA_GAP`; that is a simulator limitation, not an LLK fix failure. Finalize as `failed` unless the caller explicitly reruns with `TEST_BACKEND=local`.

## Step 5.3: Review the Fix and Feed Back

A senior-reviewer pass over the fix diff, run as a loop **inside** the pipeline —
same idea as a `code-review` bot, except findings are fed back to the worker
instead of posted to a PR. Unlike perf, review is static (reads the diff only),
so it runs for **every** backend and arch.

Run this only once the functional tests are **green** (the tester returned
`SUCCESS`). A green functional result implies a fix diff exists, so there is
always something to review here.

Advance to the `review` step and run the reviewer:

```bash
python codegen/scripts/run_json_writer.py advance \
  --log-dir "$LOG_DIR" \
  --new-step "review" \
  --new-message "Reviewing fix diff for issue #${ISSUE_NUMBER} (attempt $((REVIEW_RETRIES+1))/$((MAX_REVIEW_RETRIES+1)))" \
  --prev-result "success" \
  --prev-message "Functional tests passed" \
  --agent "tester"
```

Spawn `reviewer.md` with: issue number, `TARGET_ARCH`, changed files,
`WORKTREE_DIR`, `LOG_DIR`, and `PR_REVIEW_KNOWLEDGE_DIR`. It writes
`$LOG_DIR/review_result.json` and `$LOG_DIR/agent_reviewer.md`, and returns
`REVIEW_CLEAN` or `REVIEW_CHANGES_REQUESTED`. Patch its result into `run.json`:

```bash
python codegen/scripts/run_json_writer.py metric \
  --log-dir "$LOG_DIR" \
  --patch-json "{\"review\": $(cat "$LOG_DIR/review_result.json")}"
```

**Review feedback loop.** Read `blocking_total` from `review_result.json`. If it
is `0` (verdict `clean`), proceed to Step 5.5. If it is `> 0` and
`REVIEW_RETRIES < MAX_REVIEW_RETRIES`, send the blocking findings back to the
worker:

```bash
python codegen/scripts/run_json_writer.py failure \
  --log-dir "$LOG_DIR" \
  --step "review" \
  --agent "reviewer" \
  --type "test_failure" \
  --message "$REVIEW_FAILURE_SUMMARY" \
  --resolved "false"

python codegen/scripts/run_json_writer.py advance \
  --log-dir "$LOG_DIR" \
  --new-step "fix_tests" \
  --new-message "Addressing review findings for issue #${ISSUE_NUMBER}; attempt $((REVIEW_RETRIES+1))/${MAX_REVIEW_RETRIES}" \
  --prev-result "test_failure" \
  --prev-message "$REVIEW_FAILURE_SUMMARY" \
  --agent "fix_tests"
```

Spawn `issue-worker.md` in debug/retry mode with `FAILURE_CLASS=REVIEW_FINDINGS`
and the `$LOG_DIR/review_result.json` path. The worker addresses each **blocking**
finding with the smallest fix; advisory findings are recorded only, not looped
on. Then:

```bash
REVIEW_RETRIES=$((REVIEW_RETRIES + 1))
DEBUG_CYCLES=$((DEBUG_CYCLES + 1))
```

Re-run **Step 4 (functional Test)** — a review fix must not break correctness —
and, if it stays green, re-run this Step 5.3. If the worker returns
`HYPOTHESIS_REFUTED` (the finding cannot be resolved without breaking
correctness), stop the loop and go to Step 5.5.

**When the review budget is exhausted** (`REVIEW_RETRIES == MAX_REVIEW_RETRIES`
and blocking findings remain): the run does **not** fail on the review alone.
Keep the functional `STATUS`, leave `review.verdict=changes_requested` in
`run.json`, and set `OBSTACLE=unresolved_review_findings` as the terminal record.
Then proceed to Step 5.5.

## Step 5.5: Measure Perf and Feed Back

Run this only once the functional tests are **green** (the tester returned
`SUCCESS`, either on the first pass or after the Step 5 debug retry). If the
functional result is not green, skip straight to Step 6.

**Gate.** Perf cycle counts are only meaningful on real silicon. Skip the perf
stage (record it as not measured and go to Step 6) when **either**:

- `TEST_BACKEND != local`, or
- `TARGET_ARCH` is not `blackhole` or `wormhole`.

```bash
if [ "$TEST_BACKEND" != "local" ] || { [ "$TARGET_ARCH" != "blackhole" ] && [ "$TARGET_ARCH" != "wormhole" ]; }; then
  python codegen/scripts/run_json_writer.py metric \
    --log-dir "$LOG_DIR" \
    --patch-json "{\"perf\": {\"measured\": false, \"verdict\": \"not_measured\", \"reason\": \"perf only runs on local Blackhole/Wormhole silicon\"}}"
  # proceed to Step 6
fi
```

Otherwise advance to the `perf` step and run the perf-tester:

```bash
python codegen/scripts/run_json_writer.py advance \
  --log-dir "$LOG_DIR" \
  --new-step "perf" \
  --new-message "Measuring ${TARGET_ARCH} perf for issue #${ISSUE_NUMBER} (goal=${PERF_GOAL})" \
  --prev-result "success" \
  --prev-message "Functional tests passed" \
  --agent "perf"
```

Spawn `perf-tester.md` with: `TARGET_ARCH`, `TEST_BACKEND`, `PERF_GOAL`, issue
number, the changed kernel/op (from the analysis), fix plan path, changed files,
`WORKTREE_DIR`, `LOG_DIR`. The perf-tester writes its result to
`$LOG_DIR/perf_result.json`; patch it into `run.json` after it returns:

```bash
python codegen/scripts/run_json_writer.py metric \
  --log-dir "$LOG_DIR" \
  --patch-json "{\"perf\": $(cat "$LOG_DIR/perf_result.json")}"
```

The perf-tester returns one of:

- `PERF_OK` — goal met (faster, or not slower). Proceed to Step 6.
- `PERF_NOT_APPLICABLE` / `PERF_ENV_ERROR` — could not measure/judge (no perf
  test maps to the change, no baseline, or the perf test could not run). **Never
  block** on these; proceed to Step 6 on the functional result.
- `PERF_REGRESSED` — got slower than baseline. A miss for any goal.
- `PERF_NOT_IMPROVED` — `PERF_GOAL=improve` and the fix did not get faster. A
  miss only for optimization issues.

**Perf feedback loop.** If the perf-tester returns a *miss* (`PERF_REGRESSED`, or
`PERF_NOT_IMPROVED` when `PERF_GOAL=improve`) and `PERF_RETRIES < MAX_PERF_RETRIES`:

```bash
python codegen/scripts/run_json_writer.py failure \
  --log-dir "$LOG_DIR" \
  --step "perf" \
  --agent "perf" \
  --type "test_failure" \
  --message "$PERF_FAILURE_SUMMARY" \
  --resolved "false"

python codegen/scripts/run_json_writer.py advance \
  --log-dir "$LOG_DIR" \
  --new-step "fix_tests" \
  --new-message "Recovering perf for issue #${ISSUE_NUMBER} (${PERF_GOAL}); attempt $((PERF_RETRIES+1))/${MAX_PERF_RETRIES}" \
  --prev-result "test_failure" \
  --prev-message "$PERF_FAILURE_SUMMARY" \
  --agent "fix_tests"
```

Spawn `issue-worker.md` in debug/retry mode with
`FAILURE_CLASS=PERF_REGRESSION` (goal `no_regress`) or
`FAILURE_CLASS=PERF_NOT_IMPROVED` (goal `improve`), plus the perf evidence and
the `perf_baseline_*`/`perf_current_*` CSV paths from `LOG_DIR`. The worker must
recover/improve cycles **without breaking correctness**. Then:

```bash
PERF_RETRIES=$((PERF_RETRIES + 1))
DEBUG_CYCLES=$((DEBUG_CYCLES + 1))
```

Re-run **Step 4 (functional Test)** — correctness must still hold — and, if it
stays green, re-run this Step 5.5 perf check. Never accept a perf "fix" that
breaks a functional test. If the worker returns `HYPOTHESIS_REFUTED` (e.g., the
regression is inherent to the correctness fix), stop looping and treat it as an
exhausted miss below.

**When the perf budget is exhausted** (`PERF_RETRIES == MAX_PERF_RETRIES` and
still a miss, or `HYPOTHESIS_REFUTED`):

- `PERF_GOAL=no_regress` + still regressed → set `STATUS=failed`,
  `OBSTACLE=perf_regression`, `FINAL_RESULT=test_failure`. A correctness fix that
  silently regresses perf is not acceptable; surface it for human review.
- `PERF_GOAL=improve` + still not improved → keep the functional `STATUS`
  (`success`) but leave `perf.verdict=not_improved` in `run.json` and note it in
  the report. An otherwise-correct change is not failed solely for missing a
  speedup, but the unmet optimization goal is made visible.

## Step 6: Finalize

Pick status from the tester's (or metal-tester's) verdict:

- `success`: `SUCCESS` from a real functional test — tt-llk **or** metal (a Layer-3 fix
  the metal suite verified lands here)
- `compiled`: `COMPILED_ONLY`, **or** `UNVERIFIABLE_IN_LLK_SUITE` (`VERIFY_ROUTE=none`: the
  fix is applied + committed but no in-harness test exists — verify in tt-metal CI). A real
  fix exists; this is a Working outcome. **Do not report this as `skipped`.**
- `failed`: compile/test failure, `ENV_ERROR`, or `SIM_ISA_GAP`
- `skipped`: **only** when the analyzer found no relevant LLK work (out of scope). Never
  `skipped` when a fix was produced.

`run_json_writer.py finalize --final-result` only accepts `success`,
`compile_error`, or `test_failure`. Use `success` for terminal
`success`/`compiled`/`skipped`, `compile_error` for compile failures, and
`test_failure` for test/runtime/scope/simulator/environment failures.

**Deferred-verification messaging (`VERIFY_DEFERRED=1`).** Keep `OBSTACLE` empty (the
dashboard renders any obstacle as a red box, and this is a Working outcome) and carry the
next step in the final message:

```bash
if [ "${VERIFY_DEFERRED:-0}" = 1 ]; then
  export OBSTACLE=
  export FINAL_MESSAGE="${TARGET_ARCH} issue #${ISSUE_NUMBER}: fix applied — ${VERIFY_DEFER_NOTE:-no in-harness test exercises this ${FIX_LAYER} change; verify in tt-metal CI}"
fi
```

Write final dashboard state, upsert `runs.jsonl`, copy artifacts, and snapshot changed files:

```bash
case "$STATUS" in
  success|compiled) export SOLVER_STATE=working ;;
  failed|skipped) export SOLVER_STATE=not_working ;;
esac
case "$STATUS" in
  success|compiled|skipped) export FINAL_RESULT=success ;;
  failed) : "${FINAL_RESULT:=test_failure}" ;;
esac

export END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# Exclude perf_data/ — the perf stage regenerates those CSVs as a measurement
# artifact; they are not part of the fix and must not land in the diff or PR.
export CHANGED_FILES=$(git -C "$WORKTREE_DIR" diff --name-only | grep -v '/perf_data/' | grep -v '^perf_data/' || true)
export CHANGED_FILES_JSON=$(python -c "import json,os; print(json.dumps([l for l in os.environ['CHANGED_FILES'].splitlines() if l]))")

# ── Preserve the fix durably: local commit (no push) + archived patch ──
# The commit to WORKTREE_BRANCH survives worktree removal; generated.patch in the
# durable LOG_DIR is a second recovery path. Caller owns push/PR (see Git Policy).
export WORKTREE_DIR WORKTREE_BRANCH GIT_BRANCH
export BASE_COMMIT="$GIT_COMMIT"   # branch base == origin/main
export FIX_COMMIT=""
# Stage the fix across all allowed layers (incl. new files), never perf CSVs.
# Symlinked infra is gitignored so add -A skips it (advice off = no exit-1 noise).
FIX_PATHSPEC="tt_metal/tt-llk tt_metal/hw/ckernels tt_metal/hw/inc/api/compute ttnn/cpp/ttnn/operations tests/tt_metal :(exclude,glob)**/perf_data/** :(exclude,glob)**/__pycache__/** :(exclude)tt_metal/tt-llk/tests/.venv :(exclude)tt_metal/tt-llk/tests/sfpi"
git -C "$WORKTREE_DIR" -c advice.addIgnoredFile=false add -A -- $FIX_PATHSPEC 2>/dev/null || true
if ! git -C "$WORKTREE_DIR" diff --cached --quiet 2>/dev/null; then
  git -C "$WORKTREE_DIR" \
    -c user.name="ai-code-gen" -c user.email="ai-code-gen@tenstorrent.com" \
    commit -q -m "AI issue-solver: fix #${ISSUE_NUMBER} ${ISSUE_TITLE}" 2>/dev/null || true
  export FIX_COMMIT=$(git -C "$WORKTREE_DIR" rev-parse HEAD 2>/dev/null || echo "")
fi
# Apply-able patch archived in the durable LOG_DIR. Reapply from the tt-metal
# repo root with:  git checkout $BASE_COMMIT && git apply $LOG_DIR/generated.patch
if [ -n "$FIX_COMMIT" ] && [ "$FIX_COMMIT" != "$BASE_COMMIT" ]; then
  git -C "$WORKTREE_DIR" diff --binary "$BASE_COMMIT" "$FIX_COMMIT" > "$LOG_DIR/generated.patch" 2>/dev/null || true
else
  # Nothing committed (skipped / no changes): capture any working-tree delta.
  git -C "$WORKTREE_DIR" -c advice.addIgnoredFile=false add -AN -- $FIX_PATHSPEC 2>/dev/null || true
  git -C "$WORKTREE_DIR" diff --binary HEAD -- $FIX_PATHSPEC > "$LOG_DIR/generated.patch" 2>/dev/null || true
  git -C "$WORKTREE_DIR" reset -q -- $FIX_PATHSPEC 2>/dev/null || true
fi
[ -s "$LOG_DIR/generated.patch" ] || rm -f "$LOG_DIR/generated.patch"

python codegen/scripts/run_json_writer.py finalize \
  --log-dir "$LOG_DIR" \
  --end-time "$END_TIME" \
  --status "$STATUS" \
  --final-result "$FINAL_RESULT" \
  --final-message "${FINAL_MESSAGE:-${TARGET_ARCH} issue #${ISSUE_NUMBER}: ${STATUS}}" \
  --solver-state "$SOLVER_STATE" \
  --patch-json "$(python - <<PY
import json, os
log_dir = os.environ["LOG_DIR"]
run_path = os.path.join(log_dir, "run.json")
try:
    agents = json.load(open(run_path)).get("agents", [])
except FileNotFoundError:
    agents = []
for agent, filename in [
    ("analyzer", "agent_issue_analyzer.md"),
    ("arch_lookup", "agent_arch_lookup.md"),
    ("writer", "agent_issue_worker.md"),
    ("tester", "agent_tester.md"),
    ("metal_test", "agent_metal_tester.md"),
    ("reviewer", "agent_reviewer.md"),
    ("perf", "agent_perf_tester.md"),
    ("fix_tests", "agent_issue_worker_debug.md"),
]:
    if os.path.exists(os.path.join(log_dir, filename)) and agent not in agents:
        agents.append(agent)
print(json.dumps({
    "compilation_attempts": int(os.environ.get("COMPILATION_ATTEMPTS", "0")),
    "debug_cycles": int(os.environ.get("DEBUG_CYCLES", "0")),
    "tests_total": int(os.environ.get("TESTS_TOTAL", "0")),
    "tests_passed": int(os.environ.get("TESTS_PASSED", "0")),
    "agents": agents,
    "changed_files": json.loads(os.environ.get("CHANGED_FILES_JSON", "[]")),
    "test_backend": os.environ.get("TEST_BACKEND", ""),
    "create_local_branch_requested": os.environ.get("CREATE_LOCAL_BRANCH", ""),
    "create_pr_requested": os.environ.get("CREATE_PR", ""),
    # Durability: base_commit=branch base, fix_commit=local commit, artifact_patch=archived diff.
    "base_commit": os.environ.get("BASE_COMMIT") or None,
    "fix_commit": os.environ.get("FIX_COMMIT") or None,
    "branch": os.environ.get("GIT_BRANCH") or os.environ.get("WORKTREE_BRANCH") or None,
    "worktree_dir": os.environ.get("WORKTREE_DIR") or None,
    "artifact_patch": "generated.patch" if os.path.exists(os.path.join(log_dir, "generated.patch")) else None,
    "obstacle": os.environ.get("OBSTACLE") or None,
}))
PY
)"

python codegen/scripts/issue_solver_run_utils.py upsert-runs-jsonl \
  --log-dir "$LOG_DIR" \
  --runs-jsonl "${LOGS_BASE}/runs.jsonl"

cp codegen/artifacts/issue_${ISSUE_NUMBER}_*.md "$LOG_DIR/" 2>/dev/null || true
while IFS= read -r f; do
  [ -z "$f" ] && continue
  flat=$(echo "$f" | tr '/' '_')
  [ -f "$WORKTREE_DIR/$f" ] && cp "$WORKTREE_DIR/$f" "$LOG_DIR/$flat" 2>/dev/null || true
  git -C "$WORKTREE_DIR" show "origin/main:$f" > "$LOG_DIR/base_$flat" 2>/dev/null || true
  [ -s "$LOG_DIR/base_$flat" ] || rm -f "$LOG_DIR/base_$flat"
done <<EOF
$CHANGED_FILES
EOF
```

Verify expected self-logs exist. If a subagent ran but did not create its log, write a placeholder file in `LOG_DIR`.

Return:

```text
Issue-Solver Result:
  status: success|compiled|failed|skipped
  codegen_version: ${CODEGEN_VERSION}
  run_id: ${RUN_ID}
  log_dir: ${LOG_DIR}
  branch: ${WORKTREE_BRANCH}            # fix committed here (local, NOT pushed)
  base_commit: ${BASE_COMMIT}           # origin/main SHA the branch was cut from
  fix_commit: ${FIX_COMMIT}             # the local fix commit (empty if no change)
  worktree_dir: ${WORKTREE_DIR}         # where the run executed; removed after finish (recover via branch or patch)
  patch: ${LOG_DIR}/generated.patch     # reapply: git checkout <base_commit> && git apply <patch>
  test_backend: ${TEST_BACKEND}
  perf:
    goal: ${PERF_GOAL}            # improve | no_regress
    verdict: ...                  # improved | neutral | regressed | not_improved | no_baseline | not_measured
    test: ...                     # perf module + -k filter, or "n/a"
    baseline_vs_current: ...      # "<base> -> <cur> cycles (median <pct>%, worst <pct>%)", or "not measured"
    retries_used: ${PERF_RETRIES}/${MAX_PERF_RETRIES}
  review:
    verdict: ...                  # clean | changes_requested | not_reviewed
    findings_total: ...           # review.findings_total
    blocking_total: ...           # review.blocking_total (0 once the loop converges)
    retries_used: ${REVIEW_RETRIES}/${MAX_REVIEW_RETRIES}
    advisory:                     # non-blocking findings recorded, not acted on (nits/parity/style)
      - <severity> <file>:<line> — <title>
  cost:
    tokens: ...                   # "in=<n> out=<n> cache_read=<n> cache_creation=<n>" (tokens.*)
    total_tokens: ...             # tokens.total (input + output)
    est_usd: ...                  # tokens.cost_usd (estimate), or "n/a"
  create_local_branch_requested: ${CREATE_LOCAL_BRANCH}
  create_pr_requested: ${CREATE_PR}
  changed_files:
    ...
  obstacle: ...
```

Populate the `perf:` block from the `perf` object in `run.json` (written by the
perf-tester). When the perf stage was gated out, report `verdict: not_measured`
and `test: n/a`.

Populate the `review:` block from the `review` object in `run.json` (written by
the reviewer). List every `blocking: false` finding under `advisory:` — the
nits/parity/style items the loop recorded but did not act on.

Populate the `cost:` block from the `tokens` object in `run.json` (written by
the `session_cost.py` refreshes). If the session could not be discovered,
report `est_usd: n/a` and token totals as `0`.
