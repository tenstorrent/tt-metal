---
name: issue-solver-orchestrator
description: "LLK issue-solver orchestrator. Uses the tt-llk .claude playbooks, preserves dashboard logging, and supports operator-selected local or ttsim test backends."
model: sonnet
tools: Read, Write, Edit, Bash, Glob, Grep, Agent, mcp__atlassian__search, mcp__atlassian__searchConfluenceUsingCql, mcp__atlassian__getConfluencePage, mcp__atlassian__getAccessibleAtlassianResources, mcp__deepwiki__ask_question, mcp__deepwiki__read_wiki_contents, mcp__deepwiki__read_wiki_structure
---

# LLK Issue Solver Orchestrator

This orchestrator fixes an existing GitHub issue in `tt_metal/tt-llk`. It is intentionally thin:

- Use the local `.claude` playbooks as the technical source of truth.
- Keep the existing `/proj_sw/user_dev/llk_code_gen/*_issue_solver` dashboard logging shape.
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

- Allowed: `git status`, `git diff`, `git show`, `git log`, `git rev-parse`.
- Not allowed: `git push`, PR creation, branch deletion, destructive reset/restore.
- Commit/PR decisions are returned to the caller via the final report.

## Step 0: Setup

Load the minimal arch profile:

```bash
case "$TARGET_ARCH" in
  blackhole)
    export LLK_DIR=tt_llk_blackhole
    export REF_ARCH=wormhole
    export REF_LLK_DIR=tt_llk_wormhole_b0
    export LOGS_BASE=/proj_sw/user_dev/llk_code_gen/blackhole_issue_solver
    export DASHBOARD_PROJECT_ID=blackhole_issue_solver
    ;;
  wormhole)
    export LLK_DIR=tt_llk_wormhole_b0
    export REF_ARCH=
    export REF_LLK_DIR=
    export LOGS_BASE=/proj_sw/user_dev/llk_code_gen/wormhole_issue_solver
    export DASHBOARD_PROJECT_ID=wormhole_issue_solver
    ;;
  quasar)
    export LLK_DIR=tt_llk_quasar
    export REF_ARCH=blackhole
    export REF_LLK_DIR=tt_llk_blackhole
    export LOGS_BASE=/proj_sw/user_dev/llk_code_gen/quasar_issue_solver
    export DASHBOARD_PROJECT_ID=quasar_issue_solver
    ;;
  *)
    echo "Unknown TARGET_ARCH: $TARGET_ARCH" >&2
    exit 1
    ;;
esac
```

Create the run directory and initial live dashboard record:

```bash
cd "$WORKTREE_DIR/tt_metal/tt-llk"

export START_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
export RUN_ID=$(date +%Y-%m-%d)_issue_${ISSUE_NUMBER}_$(head -c 4 /dev/urandom | xxd -p)
export LOG_DIR=${LOGS_BASE}/${RUN_ID}
export GIT_COMMIT=$(git -C "$WORKTREE_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
export GIT_BRANCH=$(git -C "$WORKTREE_DIR" branch --show-current 2>/dev/null || echo "$WORKTREE_BRANCH")
export COMPILATION_ATTEMPTS=0
export DEBUG_CYCLES=0
export TESTS_TOTAL=0
export TESTS_PASSED=0
export OBSTACLE=
export ISSUE_NUMBER ISSUE_TITLE ISSUE_LABELS ISSUE_URL
export TEST_BACKEND TTSIM_SO_PATH CREATE_LOCAL_BRANCH CREATE_PR
mkdir -p "$LOG_DIR/instructions" codegen/artifacts

cp codegen/agents/issue-solver/*.md "$LOG_DIR/instructions/" 2>/dev/null || true
cp .claude/CLAUDE.md "$LOG_DIR/instructions/tt-llk-CLAUDE.md" 2>/dev/null || true
cp -R .claude/skills "$LOG_DIR/instructions/claude-skills" 2>/dev/null || true

PIPELINE_STEPS='[
  {"id":"analyzer","name":"Analyze","desc":"Understand the issue and scope"},
  {"id":"arch_lookup","name":"Research","desc":"Look up architecture facts only when needed"},
  {"id":"writer","name":"Fix","desc":"Plan and implement the smallest fix"},
  {"id":"tester","name":"Test","desc":"Run selected backend tests"},
  {"id":"fix_tests","name":"Retry","desc":"Debug and update the fix after test failure"}
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
  --description "#${ISSUE_NUMBER}: ${ISSUE_TITLE}" \
  --pipeline-steps "$PIPELINE_STEPS" \
  --issue "$ISSUE_JSON"
```

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

    WORKTREE_DIR: ${WORKTREE_DIR}
    LOG_DIR: ${LOG_DIR}
```

The analyzer must write `codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md` and `${LOG_DIR}/agent_issue_analyzer.md`.

If the analyzer declares the issue out of scope, finalize as `skipped`.

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

Parse the tester report and update `TESTS_TOTAL` / `TESTS_PASSED` when counts are available. If the tester only reports a single command-level verdict, record `TESTS_TOTAL=1` and `TESTS_PASSED=1` for `SUCCESS`, otherwise `TESTS_PASSED=0`.

## Step 5: Debug Once, Then Re-test

If the tester returns `COMPILE_FAILED` or `TESTS_FAILED`, record the failure and spawn `issue-worker.md` once more in debug/retry mode:

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
  --new-message "Debugging test or compile failure for issue #${ISSUE_NUMBER}" \
  --prev-result "test_failure" \
  --prev-message "$FAILURE_SUMMARY" \
  --agent "tester"
```

The retry worker reads the existing plan plus tester evidence, patches the implementation or updates the plan scope when evidence supports that, and writes `${LOG_DIR}/agent_issue_worker_debug.md`. After it returns, increment `DEBUG_CYCLES`, then re-run Step 4 once. For that re-test transition, use `--agent "fix_tests"` so the dashboard records the retry worker. If the second tester run is still red, finalize as `failed` with the tester/worker evidence. If the worker returns `HYPOTHESIS_REFUTED`, finalize as `failed` with that evidence instead of looping.

Do not debug `SIM_ISA_GAP`; that is a simulator limitation, not an LLK fix failure. Finalize as `failed` unless the caller explicitly reruns with `TEST_BACKEND=local`.

## Step 6: Finalize

Pick status:

- `success`: compiles and selected tests pass
- `compiled`: compiles but no relevant test exists and the plan declared compile-only
- `failed`: compile/test failure, `ENV_ERROR`, or `SIM_ISA_GAP`
- `skipped`: analyzer found no relevant LLK work

`run_json_writer.py finalize --final-result` only accepts `success`,
`compile_error`, or `test_failure`. Use `success` for terminal
`success`/`compiled`/`skipped`, `compile_error` for compile failures, and
`test_failure` for test/runtime/scope/simulator/environment failures.

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
export CHANGED_FILES=$(git -C "$WORKTREE_DIR" diff --name-only)
export CHANGED_FILES_JSON=$(python -c "import json,os; print(json.dumps(os.environ['CHANGED_FILES'].splitlines()))")

python codegen/scripts/run_json_writer.py finalize \
  --log-dir "$LOG_DIR" \
  --end-time "$END_TIME" \
  --status "$STATUS" \
  --final-result "$FINAL_RESULT" \
  --final-message "${TARGET_ARCH} issue #${ISSUE_NUMBER}: ${STATUS}" \
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
  run_id: ${RUN_ID}
  log_dir: ${LOG_DIR}
  test_backend: ${TEST_BACKEND}
  create_local_branch_requested: ${CREATE_LOCAL_BRANCH}
  create_pr_requested: ${CREATE_PR}
  changed_files:
    ...
  obstacle: ...
```
