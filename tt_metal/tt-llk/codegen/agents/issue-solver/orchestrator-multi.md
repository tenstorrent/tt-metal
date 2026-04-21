---
name: issue-solver-orchestrator-multi
description: "Single-run multi-arch LLK issue-solver. Creates one dashboard run, one shared fix, and per-arch test results."
model: sonnet
tools: Read, Write, Edit, Bash, Glob, Grep, Agent
---

# Multi-Arch LLK Issue Solver

This orchestrator fixes one GitHub issue across multiple LLK architectures as one coordinated run.

Core contract:

- Create exactly one dashboard run under `/proj_sw/user_dev/llk_code_gen/issue_solver`.
- Do not delete, rename, or rewrite historical runs under `blackhole_issue_solver`, `wormhole_issue_solver`, or `quasar_issue_solver`.
- Do not spawn `codegen/agents/issue-solver/orchestrator.md` as a child per arch.
- Run one analyzer, one shared issue-worker, and one tester session.
- Store per-arch progress in the single run's `arch_results` field.
- The tester may switch local/ttsim arch environment between test commands; it must set arch-specific env per command.
- Do not push or create PRs here. Return enough metadata for the caller to create a PR if requested.

## Startup Contract

Before doing analysis or spawning agents, make sure these choices are known. Ask once up front for anything missing:

1. `TEST_BACKEND`: `local` or `ttsim`.
   - `local` means `.claude/scripts/run_test.sh` decides the normal local backend for each arch.
   - `ttsim` means the tester uses an in-process `libttsim_*.so`.
   - If `TEST_BACKEND=ttsim`, ask only: `Path to the libttsim .so for <arch>?` for each missing target arch.
   - Accept common paths such as `~/sim/wh/libttsim_wh.so` and `~/sim/bh/libttsim_bh.so`; the tester validates the file and companion `soc_descriptor.yaml`.
2. `CREATE_LOCAL_BRANCH`: `yes` or `no`.
   - Branch/worktree creation is owned by the caller/top-level orchestrator.
   - If `yes` and `WORKTREE_DIR` or `WORKTREE_BRANCH` is missing, stop and ask the caller to create a branch from latest `origin/main`.
3. `CREATE_PR`: `yes` or `no`.
   - This issue-solver does not push. Return final metadata for the caller to create a PR if requested.

Ask clarifying issue questions only before Step 0. After Step 0, work autonomously until a terminal status is logged.

## Inputs

Required:

- `TARGET_ARCHES`: JSON array or comma-separated list containing `blackhole`, `wormhole`, and/or `quasar`
- `ISSUE_NUMBER`
- `ISSUE_TITLE`
- `ISSUE_BODY`
- `ISSUE_LABELS`
- `ISSUE_COMMENTS`
- `WORKTREE_DIR`: absolute path to the issue worktree
- `WORKTREE_BRANCH`
- `TEST_BACKEND`: `local` or `ttsim`
- `TTSIM_SO_PATHS`: required when `TEST_BACKEND=ttsim`; JSON object mapping each target arch to its `.so` path
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

## Step 0: Setup One Run

Normalize `TARGET_ARCHES` to a unique ordered list. Preserve the caller's order.

Valid arches and profiles:

```bash
case "$arch" in
  blackhole)
    LLK_DIR_OF[$arch]=tt_llk_blackhole
    REF_ARCH_OF[$arch]=wormhole
    REF_LLK_DIR_OF[$arch]=tt_llk_wormhole_b0
    ;;
  wormhole)
    LLK_DIR_OF[$arch]=tt_llk_wormhole_b0
    REF_ARCH_OF[$arch]=
    REF_LLK_DIR_OF[$arch]=
    ;;
  quasar)
    LLK_DIR_OF[$arch]=tt_llk_quasar
    REF_ARCH_OF[$arch]=blackhole
    REF_LLK_DIR_OF[$arch]=tt_llk_blackhole
    ;;
  *)
    echo "Unknown target arch: $arch" >&2
    exit 1
    ;;
esac
```

Create the shared dashboard run:

```bash
cd "$WORKTREE_DIR/tt_metal/tt-llk"

export LOGS_BASE=/proj_sw/user_dev/llk_code_gen/issue_solver
export DASHBOARD_PROJECT_ID=issue_solver
export START_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
export RUN_ID=$(date +%Y-%m-%d)_issue_${ISSUE_NUMBER}_multi_$(head -c 4 /dev/urandom | xxd -p)
export LOG_DIR=${LOGS_BASE}/${RUN_ID}
export GIT_COMMIT=$(git -C "$WORKTREE_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
export GIT_BRANCH=$(git -C "$WORKTREE_DIR" branch --show-current 2>/dev/null || echo "$WORKTREE_BRANCH")
export COMPILATION_ATTEMPTS=0
export DEBUG_CYCLES=0
export TESTS_TOTAL=0
export TESTS_PASSED=0
export OBSTACLE=
export ISSUE_NUMBER ISSUE_TITLE ISSUE_LABELS ISSUE_URL
export TEST_BACKEND CREATE_LOCAL_BRANCH CREATE_PR
mkdir -p "$LOG_DIR/instructions" codegen/artifacts

cp codegen/agents/issue-solver/*.md "$LOG_DIR/instructions/" 2>/dev/null || true
cp .claude/CLAUDE.md "$LOG_DIR/instructions/tt-llk-CLAUDE.md" 2>/dev/null || true
cp -R .claude/skills "$LOG_DIR/instructions/claude-skills" 2>/dev/null || true

PIPELINE_STEPS='[
  {"id":"analyzer","name":"Analyze","desc":"Understand the issue and all target arches"},
  {"id":"arch_lookup","name":"Research","desc":"Look up architecture facts only when needed"},
  {"id":"writer","name":"Fix","desc":"Plan and implement one coordinated multi-arch fix"},
  {"id":"tester","name":"Test","desc":"Run selected backend tests for each target arch"},
  {"id":"fix_tests","name":"Retry","desc":"Debug and update the shared fix after test failure"}
]'
```

Build `TARGET_ARCHES_JSON` and initialize pending `arch_results`:

```bash
TARGET_ARCHES_JSON=$(python - <<'PY'
import json, os
raw = os.environ["TARGET_ARCHES"]
if raw.strip().startswith("["):
    values = json.loads(raw)
else:
    values = [part.strip() for part in raw.split(",") if part.strip()]
aliases = {"bh": "blackhole", "wh": "wormhole", "qsr": "quasar"}
seen = set()
arches = []
for value in values:
    arch = aliases.get(str(value).strip().lower(), str(value).strip().lower())
    if arch not in {"blackhole", "wormhole", "quasar"}:
        raise SystemExit(f"unknown target arch: {value}")
    if arch not in seen:
        seen.add(arch)
        arches.append(arch)
print(json.dumps(arches))
PY
)
export TARGET_ARCHES_JSON

ARCH_COUNT=$(python - <<'PY'
import json, os
print(len(json.loads(os.environ["TARGET_ARCHES_JSON"])))
PY
)

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

INIT_PATCH_JSON=$(python - <<'PY'
import json, os
arches = json.loads(os.environ["TARGET_ARCHES_JSON"])
print(json.dumps({
    "multi_arch_run": True,
    "target_arches": arches,
    "combined_status": "running",
    "arch_results": {
        arch: {
            "status": "pending",
            "verdict": None,
            "tests_total": 0,
            "tests_passed": 0,
            "obstacle": None,
        }
        for arch in arches
    },
    "test_backend": os.environ.get("TEST_BACKEND", ""),
    "create_local_branch_requested": os.environ.get("CREATE_LOCAL_BRANCH", ""),
    "create_pr_requested": os.environ.get("CREATE_PR", ""),
}))
PY
)

python codegen/scripts/run_json_writer.py init \
  --log-dir "$LOG_DIR" \
  --run-id "$RUN_ID" \
  --kernel "issue_${ISSUE_NUMBER}" \
  --kernel-type "issue_solver" \
  --arch "multi" \
  --start-time "$START_TIME" \
  --first-step "analyzer" \
  --first-message "Analyzing issue #${ISSUE_NUMBER} for ${TARGET_ARCHES_JSON}" \
  --prompt "Fix multi-arch issue #${ISSUE_NUMBER} using ${TEST_BACKEND} tests" \
  --batch-id "${CODEGEN_BATCH_ID:-}" \
  --model "${CODEGEN_MODEL:-sonnet}" \
  --run-type "${CODEGEN_RUN_TYPE:-manual}" \
  --git-commit "$GIT_COMMIT" \
  --git-branch "$GIT_BRANCH" \
  --description "#${ISSUE_NUMBER}: ${ISSUE_TITLE}" \
  --phases-total "$ARCH_COUNT" \
  --pipeline-steps "$PIPELINE_STEPS" \
  --issue "$ISSUE_JSON" \
  --patch-json "$INIT_PATCH_JSON"
```

## Step 1: Analyze Once

Spawn the analyzer once for the full target list:

```text
Agent:
  subagent_type: general-purpose
  description: "Analyze multi-arch issue #${ISSUE_NUMBER}"
  prompt: |
    Read and follow codegen/agents/issue-solver/issue-analyzer.md.

    TARGET_ARCHES: ${TARGET_ARCHES_JSON}
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

If the analyzer declares the issue out of scope for every requested arch, finalize as `skipped`. If only some arches are out of scope, keep the run alive and mark those arches `skipped` in `arch_results`.

## Step 2: Research If Needed

Advance to `arch_lookup` only if the analysis asks for architecture facts:

```bash
python codegen/scripts/run_json_writer.py advance \
  --log-dir "$LOG_DIR" \
  --new-step "arch_lookup" \
  --new-message "Researching architecture details for issue #${ISSUE_NUMBER}" \
  --prev-result "success" \
  --prev-message "Issue analysis complete" \
  --agent "analyzer"
```

Spawn `arch-lookup.md` with `TARGET_ARCHES`, the analysis artifact path, and the exact research questions. It must write `codegen/artifacts/issue_${ISSUE_NUMBER}_arch_research.md` and `${LOG_DIR}/agent_arch_lookup.md`.

If research ran, set `PREVIOUS_AGENT=arch_lookup` before Step 3. If research is not needed, leave `PREVIOUS_AGENT=analyzer` and go straight to Step 3.

## Step 3: Fix Once

```bash
python codegen/scripts/run_json_writer.py advance \
  --log-dir "$LOG_DIR" \
  --new-step "writer" \
  --new-message "Planning and applying one shared fix for issue #${ISSUE_NUMBER}" \
  --prev-result "success" \
  --prev-message "Analysis/research complete" \
  --agent "${PREVIOUS_AGENT:-analyzer}"
```

Spawn `issue-worker.md` once. It owns the compact plan, implementation, and any targeted local compile checks.

Important worker requirements:

- Use `TARGET_ARCHES`, not one isolated `TARGET_ARCH`.
- Prefer one shared code/API decision, with arch-specific code only when required by existing LLK structure.
- For `TEST_BACKEND=ttsim`, do not run compile or pytest commands directly; the tester owns compilation and execution through the ttsim command contract.
- Write `codegen/artifacts/issue_${ISSUE_NUMBER}_fix_plan.md` and `${LOG_DIR}/agent_issue_worker.md`.

After the worker returns, record changed files with:

```bash
git -C "$WORKTREE_DIR" diff --name-only
```

## Step 4: Test Once Across Arches

```bash
python codegen/scripts/run_json_writer.py advance \
  --log-dir "$LOG_DIR" \
  --new-step "tester" \
  --new-message "Running ${TEST_BACKEND} tests for issue #${ISSUE_NUMBER} across ${TARGET_ARCHES_JSON}" \
  --prev-result "success" \
  --prev-message "Shared fix applied" \
  --agent "writer"
```

Spawn `tester.md` once with:

- `TARGET_ARCHES`
- `TEST_BACKEND`
- `TTSIM_SO_PATHS` when `TEST_BACKEND=ttsim`
- issue number
- fix plan path
- changed files
- `WORKTREE_DIR`
- `LOG_DIR`

Test execution guard:

- The orchestrator must not invent or run test commands directly. Delegate to `tester.md`.
- The tester runs each selected arch sequentially in one session.
- In `TEST_BACKEND=ttsim`, every arch-specific command must set `TT_METAL_SIMULATOR`, `TT_METAL_DISABLE_SFPLOADMACRO=1`, and `CHIP_ARCH` in the command environment.
- In `TEST_BACKEND=ttsim`, reject any command containing `TT_UMD_SIMULATOR_PATH`, `flock`, `--port`, `--compile-consumer`, `--compile-producer`, `--reset-simulator-per-test`, or `.claude/scripts/run_test.sh`.
- The tester updates the single `run.json` using `message`, `phase-start`,
  `phase-end`, and `metric` as each arch progresses. `metric` accepts
  `--patch-json` only; do not use `--key`/`--value`.

The tester must write `${LOG_DIR}/agent_tester.md` and report a per-arch `arch_results` summary with verdicts:

- `SUCCESS`
- `COMPILE_FAILED`
- `TESTS_FAILED`
- `SIM_ISA_GAP`
- `ENV_ERROR`
- `COMPILED_ONLY`
- `SKIPPED`

Parse the tester report. Update aggregate `TESTS_TOTAL`, `TESTS_PASSED`, `COMPILATION_ATTEMPTS`, and `arch_results` in the top-level run.

## Step 5: Debug Once, Then Re-test Once

If any arch returns `COMPILE_FAILED` or `TESTS_FAILED`, record the failure and spawn `issue-worker.md` once more in debug/retry mode:

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
  --new-message "Debugging multi-arch test failure for issue #${ISSUE_NUMBER}" \
  --prev-result "test_failure" \
  --prev-message "$FAILURE_SUMMARY" \
  --agent "tester"
```

The retry worker reads the existing plan plus the combined tester evidence, patches the shared implementation or narrows arch-specific scope when evidence supports that, and writes `${LOG_DIR}/agent_issue_worker_debug.md`. After it returns, increment `DEBUG_CYCLES`, then re-run Step 4 once. For that re-test transition, use `--agent "fix_tests"` so the dashboard records the retry worker.

Do not debug `SIM_ISA_GAP`; that is a simulator limitation, not an LLK fix failure. Mark the affected arch failed with a simulator obstacle. Continue evaluating other arches when possible.

## Step 6: Finalize One Run

Pick `combined_status`:

- `success`: every in-scope arch passed or explicitly compiled-only
- `partial`: at least one arch passed/compiled and at least one arch failed
- `failed`: every in-scope arch failed, hit `ENV_ERROR`, or hit `SIM_ISA_GAP`
- `skipped`: analyzer found no relevant LLK work for any requested arch

`run_json_writer.py finalize` only accepts `success`, `compiled`, `failed`, or `skipped`. Use:

- `status=success` when `combined_status=success` and all successful arches ran functional tests
- `status=compiled` when every in-scope arch is `COMPILED_ONLY`
- `status=failed` when `combined_status=partial` or `combined_status=failed`
- `status=skipped` when `combined_status=skipped`

`run_json_writer.py finalize --final-result` only accepts `success`,
`compile_error`, or `test_failure`. Use:

- `final_result=success` for `status=success`, `status=compiled`, or
  `status=skipped`
- `final_result=test_failure` for `status=failed` caused by test/runtime/scope
  failures or simulator/environment blockers
- `final_result=compile_error` only when the terminal failure is a compile error

Set `solver_state=working` only for `status=success` or `status=compiled`; otherwise set `solver_state=not_working`.

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
  --final-message "multi-arch issue #${ISSUE_NUMBER}: ${COMBINED_STATUS}" \
  --solver-state "$SOLVER_STATE" \
  --patch-json "$(python - <<PY
import json, os
log_dir = os.environ["LOG_DIR"]
run_path = os.path.join(log_dir, "run.json")
try:
    current = json.load(open(run_path))
except FileNotFoundError:
    current = {}
agents = current.get("agents", [])
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
    "multi_arch_run": True,
    "target_arches": json.loads(os.environ["TARGET_ARCHES_JSON"]),
    "combined_status": os.environ.get("COMBINED_STATUS", ""),
    "arch_results": json.loads(os.environ.get("ARCH_RESULTS_JSON", "{}")),
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
Multi-Arch Issue-Solver Result:
  status: success|compiled|failed|skipped
  combined_status: success|partial|failed|skipped
  run_id: ${RUN_ID}
  log_dir: ${LOG_DIR}
  target_arches:
    - ...
  test_backend: ${TEST_BACKEND}
  create_local_branch_requested: ${CREATE_LOCAL_BRANCH}
  create_pr_requested: ${CREATE_PR}
  arch_results:
    - arch: blackhole|wormhole|quasar
      verdict: SUCCESS|COMPILE_FAILED|TESTS_FAILED|SIM_ISA_GAP|ENV_ERROR|COMPILED_ONLY|SKIPPED
      tests_total: N
      tests_passed: N
      obstacle: ...
  changed_files:
    ...
  obstacle: ...
```
