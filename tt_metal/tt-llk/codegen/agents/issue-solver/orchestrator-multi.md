---
name: issue-solver-orchestrator-multi
description: "Single-run multi-arch LLK issue-solver. Creates one dashboard run, one shared fix, and per-arch test results."
model: sonnet
tools: Read, Write, Edit, Bash, Glob, Grep, Agent
---

# Multi-Arch LLK Issue Solver

This orchestrator fixes one GitHub issue across multiple LLK architectures as one coordinated run.

Core contract:

- Create exactly one run under `${CODEGEN_LOGS_ROOT}/issue_solver`, where `CODEGEN_LOGS_ROOT` is the shared dashboard tree `/proj_sw/user_dev/llk_code_gen` **when that path exists**, otherwise an in-repo gitignored `codegen/logs/` in the main checkout. An explicit `CODEGEN_LOGS_ROOT` always wins.
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
   - Accept common paths such as `~/sim/wh/libttsim_wh.so`, `~/sim/bh/libttsim_bh.so`, and `~/sim/qsr/libttsim_qsr.so`; the tester validates the file and auto-stages the companion `soc_descriptor.yaml` (from `tt_metal/soc_descriptors/`) when it is missing and the `.so` directory is writable.
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

- Allowed (read): `git status`, `git diff`, `git show`, `git log`, `git rev-parse`.
- Allowed (orchestrator, **finalize only** — Step 6): a single **local** `git
  commit` of the shared fix to `WORKTREE_BRANCH`, plus the `git add` / `git diff`
  / `git reset` that produces `generated.patch`. This local commit is what makes
  the work durable: once committed, the fix lives in the repo's shared `.git`
  and survives even if the durable worktree directory is later removed or GC'd.
- Not allowed: `git push`, PR creation, branch deletion, `git checkout`/`switch`,
  and destructive reset/restore (`git reset --hard`, `git restore`, `git clean`).
  Subagents never commit — only the orchestrator's Step 6 does, and only locally.
- One scoped exception: the perf-tester (`perf-tester.md` Step 3) may use a
  `git stash push` / `git stash pop` pair only to revert the fix while it
  re-measures the perf baseline, and must always pop it back.
- The commit is **local only**; push/PR decisions remain the caller's and are
  returned via the final report.

## Cost Accounting

Identical to single-arch (`orchestrator.md` → Cost Accounting). Capture the
session once in Step 0, then refresh after every agent returns using the shared
`session_cost.py` engine — it sums the real per-type usage (`input`, `output`,
`cache_read`, `cache_creation`) from the session transcript and patches
`run.json`'s `tokens` + `cost_usd`:

```bash
# Step 0, once: capture the session identity.
SESSION_PAIR=$(python codegen/scripts/session_cost.py --print-session 2>/dev/null || echo "")
SESSION_ID=$(echo "$SESSION_PAIR" | awk '{print $1}')
PROJECT_CWD=$(echo "$SESSION_PAIR" | cut -d' ' -f2-)

# Every step boundary: refresh (pass values explicitly; no shared /tmp state).
python codegen/scripts/session_cost.py \
  --since "$START_TIME" --log-dir "$LOG_DIR" \
  ${SESSION_ID:+--session-id "$SESSION_ID" --project-cwd "$PROJECT_CWD"} \
  >/dev/null 2>&1 || true
```

There is one run and one session across all arches, so the transcript-summed
`tokens` cover the whole multi-arch run. Don't pass `--model` (derived per
message). `cost_usd` is an estimate; per-type token counts are the detail.

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

export DASHBOARD_PROJECT_ID=issue_solver
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
export RUN_ID=$(date +%Y-%m-%d)_issue_${ISSUE_NUMBER}_multi_$(head -c 4 /dev/urandom | xxd -p)
export LOG_DIR=${LOGS_BASE}/${RUN_ID}
export GIT_COMMIT=$(git -C "$WORKTREE_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
export GIT_BRANCH=$(git -C "$WORKTREE_DIR" branch --show-current 2>/dev/null || echo "$WORKTREE_BRANCH")
# Issue-solver-local version string (independent of Quasar codegen). Edit it by
# hand in codegen/agents/issue-solver/VERSION when you want.
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
export TEST_BACKEND TTSIM_SO_PATHS CREATE_LOCAL_BRANCH CREATE_PR

# PERF_GOAL drives the perf stage (Step 5.5). Optimization issues must get
# faster; everything else must not regress. Refined from the analyzer's
# perf_intent line in Step 1.
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
  {"id":"analyzer","name":"Analyze","desc":"Understand the issue and all target arches"},
  {"id":"arch_lookup","name":"Research","desc":"Look up architecture facts only when needed"},
  {"id":"writer","name":"Fix","desc":"Plan and implement one coordinated multi-arch fix"},
  {"id":"tester","name":"Test","desc":"Run the tt-llk Layer-1 suite for each target arch"},
  {"id":"metal_test","name":"Metal Test","desc":"Build+run the unit_tests_llk gtest suite for Layer-2/3/4 changes (same backend)"},
  {"id":"review","name":"Review","desc":"Senior LLK review of the shared fix diff (loop, no PR)"},
  {"id":"perf","name":"Perf","desc":"Measure cycle counts vs baseline per BH/WH arch (local only)"},
  {"id":"fix_tests","name":"Retry","desc":"Debug and update the shared fix after a test, review, or perf failure"}
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
  --version "$CODEGEN_VERSION" \
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

    TEST_BACKEND: ${TEST_BACKEND}
    WORKTREE_DIR: ${WORKTREE_DIR}
    LOG_DIR: ${LOG_DIR}
```

The analyzer must write `codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md` and `${LOG_DIR}/agent_issue_analyzer.md`. It classifies the **fix layer** and whether the tt-llk Python suite can verify it (`verifiable_in_llk_suite`), plus the metal gtest target when it cannot — consumed by Step 1.5.

If the analyzer declares the issue out of scope for every requested arch, finalize as `skipped`. If only some arches are out of scope, keep the run alive and mark those arches `skipped` in `arch_results`.

**Refine `PERF_GOAL`** from the analyzer's `perf_intent:` line (`optimize` →
`improve`, `maintain` → `no_regress`), preferring it over the Step 0 keyword guess:

```bash
PERF_INTENT=$(grep -ioE 'perf_intent:[[:space:]]*(optimize|maintain)' \
  "codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md" | head -1 | grep -ioE 'optimize|maintain')
case "$PERF_INTENT" in
  optimize) export PERF_GOAL=improve ;;
  maintain) export PERF_GOAL=no_regress ;;
esac
```

## Step 1.5: Route Verification by Fix Layer

Decide **before the writer** how the fix will be verified, so the run doesn't reach the
tester only to find the chosen suite can't exercise the change. The tt-llk suite
(`tester.md`) verifies Layer-1 only; the metal `unit_tests_llk` suite (`metal-tester.md`)
verifies Layer-2/3/4 on the same backend. Read the analyzer's verdict and set the route:

```bash
ANALYSIS="codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md"
gval() { grep -ioE "$1:[[:space:]]*[A-Za-z_]+" "$ANALYSIS" | head -1 | sed -E "s/.*:[[:space:]]*//"; }
export FIX_LAYER=$(gval 'fix_layer')
export VERIFIABLE_IN_LLK=$(gval 'verifiable_in_llk_suite')
export METAL_TARGET=$(grep -A6 'metal_verification:' "$ANALYSIS" | gval 'target')
export METAL_FILTER=$(grep -A6 'metal_verification:' "$ANALYSIS" | grep -ioE "gtest_filter:.*" | head -1 | sed -E "s/gtest_filter:[[:space:]]*//; s/^['\"]//; s/['\"]$//")
export METAL_DISPATCH=$(grep -A6 'metal_verification:' "$ANALYSIS" | gval 'dispatch')

case "$VERIFIABLE_IN_LLK" in
  yes)     export VERIFY_ROUTE=llk ;;                      # Layer-1 → tt-llk suite (Step 4)
  partial) export VERIFY_ROUTE=both ;;                     # L1 slice → tt-llk; higher slice → metal
  no)
    if [ -z "$METAL_TARGET" ] || [ "$METAL_TARGET" = none ]; then
      export VERIFY_ROUTE=none                             # no in-harness test exists → honest defer
    else
      export VERIFY_ROUTE=metal                            # metal gtest suite (Step 4 → metal-tester)
    fi ;;
  *)       export VERIFY_ROUTE=llk ;;                      # unknown → default to today's path
esac

# Surface the route immediately so the dashboard shows the truth at ~analysis time,
# not 30 minutes later at the tester.
python codegen/scripts/run_json_writer.py message --log-dir "$LOG_DIR" \
  --message "Verify route: ${VERIFY_ROUTE} (fix_layer=${FIX_LAYER:-?}); metal=${METAL_TARGET:-n/a} ${METAL_FILTER:-}"
```

`VERIFY_ROUTE` values, and what each means for Step 4 and Step 6:

| Route | Meaning | Step 4 | Terminal (Step 6) |
|-------|---------|--------|-------------------|
| `llk` | Layer-1; tt-llk suite exercises it | `tester.md` | `success` on pass |
| `metal` | Layer-2/3/4; a metal gtest exercises it | `metal-tester.md` (build+run `unit_tests_llk`) | `success` on pass |
| `both` | mixed; both slices testable | `tester.md` **then** `metal-tester.md` | `success` iff both pass |
| `none` | no in-harness test exists anywhere | no real run; mark `UNVERIFIABLE_IN_LLK_SUITE` | `compiled` + Working (fix ready for tt-metal CI), **never** `skipped` |

`VERIFY_ROUTE=metal`/`both` require the metal build provisioning for Step 4: pass
`METAL_VERIFY_HOME`/`METAL_VERIFY_BUILD_DIR` when a warm pre-built tree is available
(`CODEGEN_METAL_VERIFY_HOME`/`CODEGEN_METAL_VERIFY_BUILD_DIR` env, else the metal-tester
builds in the worktree with ccache).

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

Branch on `VERIFY_ROUTE` from Step 1.5:

- `llk` → run the tt-llk suite (this Step 4) only.
- `metal` → **skip** the tt-llk tester; go straight to **Step 4b** (metal suite) only.
- `both` → run this Step 4 (tt-llk, Layer-1 slice), then **Step 4b** (metal, higher slice).
- `none` → run **neither**. No in-harness test can exercise this change. Mark every
  in-scope arch `UNVERIFIABLE_IN_LLK_SUITE` and record the deferral, then go to Step 5.3:

  ```bash
  # VERIFY_ROUTE=none: fix is applied + committed, but no tt-llk OR metal test exercises it.
  # This is a WORKING outcome (compiled), NOT a failure and NOT skipped. Leave `obstacle`
  # empty so the dashboard does not red-flag it; the actionable note rides in final_message.
  for arch in $(echo "$TARGET_ARCHES_JSON" | python -c "import json,sys;print(' '.join(json.load(sys.stdin)))"); do
    python codegen/scripts/run_json_writer.py metric --log-dir "$LOG_DIR" \
      --patch-json "{\"arch_results\":{\"${arch}\":{\"status\":\"done\",\"verdict\":\"UNVERIFIABLE_IN_LLK_SUITE\",\"tests_total\":0,\"tests_passed\":0,\"obstacle\":null}}}"
  done
  export VERIFY_DEFERRED=1
  export VERIFY_DEFER_NOTE="fix applied + committed; no tt-llk or metal test exercises this ${FIX_LAYER} change — verify in tt-metal CI"
  ```

For `llk`/`both`, advance to the tester step and run the tt-llk Python suite:

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
- `SKIPPED` — this arch is genuinely **out of scope** for the fix. Do **not** use it for
  "the fix isn't reachable by this suite"; that is the metal route (Step 4b) or
  `UNVERIFIABLE_IN_LLK_SUITE` (`VERIFY_ROUTE=none`), which are Working outcomes.

Parse the tester report. Update aggregate `TESTS_TOTAL`, `TESTS_PASSED`, `COMPILATION_ATTEMPTS`, and `arch_results` in the top-level run.

## Step 4b: Metal Suite (Layer-2/3/4 verification)

Run this when `VERIFY_ROUTE` is `metal` or `both`: build and run the metal `unit_tests_llk`
gtest on the same backend to verify a Layer-2/3/4 change the tt-llk suite can't reach.

```bash
python codegen/scripts/run_json_writer.py advance \
  --log-dir "$LOG_DIR" \
  --new-step "metal_test" \
  --new-message "Building+running unit_tests_llk (${METAL_FILTER}) for issue #${ISSUE_NUMBER} across ${TARGET_ARCHES_JSON}" \
  --prev-result "success" \
  --prev-message "${VERIFY_ROUTE:-metal} route" \
  --agent "${VERIFY_ROUTE:+writer}"
```

Spawn `metal-tester.md` once with:

- `TARGET_ARCHES`, `TEST_BACKEND`, `TTSIM_SO_PATHS` (when ttsim)
- `METAL_VERIFICATION`: `target=${METAL_TARGET}`, `gtest_filter=${METAL_FILTER}`,
  `dispatch=${METAL_DISPATCH}`, and the kernel from the analysis artifact
- issue number, fix plan path, changed files, `WORKTREE_DIR`, `LOG_DIR`
- build provisioning: `METAL_VERIFY_HOME`/`METAL_VERIFY_BUILD_DIR` (from
  `CODEGEN_METAL_VERIFY_HOME`/`CODEGEN_METAL_VERIFY_BUILD_DIR`, if set)
- `FIX_PATCH`: `${LOG_DIR}/generated.patch` if it exists yet, else the metal-tester derives
  the diff from `git -C "$WORKTREE_DIR" diff`
- silicon (cardless): `HW_TEST_DISPATCH_CMD` + `HW_TEST_SESSION` reach the metal-tester
  from the environment (the dashboard sets them for a silicon solve). When present with
  `TEST_BACKEND=local`, the metal-tester offloads the build+run of each arch to the hw_test
  queue (its Step 0) instead of using a local card — nothing extra to pass here.

Metal-tester guard (mirror of the ttsim guard): every gtest command must set
`TT_METAL_HOME`, a **fresh** `TT_METAL_CACHE`, and (when `dispatch=slow`)
`TT_METAL_SLOW_DISPATCH_MODE=1`; in ttsim mode it must set `TT_METAL_SIMULATOR` to the
arch `.so` and reject `flock`, `--port`, `TT_UMD_SIMULATOR_PATH`, and pytest flags. It must
leave any warm build tree clean (reverse the applied patch).

It writes `${LOG_DIR}/agent_metal_tester.md` and reports per-arch `arch_results` with
verdicts `SUCCESS`, `COMPILE_FAILED`, `TESTS_FAILED`, `SIM_ISA_GAP`, `ENV_ERROR`, or
`UNVERIFIABLE_IN_LLK_SUITE`. Parse it and update `TESTS_TOTAL`, `TESTS_PASSED`, and
`arch_results` (for `both`, a metal `SUCCESS` combines with the tt-llk result — the arch is
green only if neither suite failed).

## Step 5: Debug and Re-test

If any arch returns `COMPILE_FAILED` or `TESTS_FAILED` — from either the tt-llk tester
(Step 4) or the metal-tester (Step 4b) — enter the debug/retry loop: spawn
`issue-worker.md` in debug/retry mode, re-run the suite that failed, and repeat while
any arch stays red — up to `MAX_DEBUG_CYCLES` (default 5) worker attempts.

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
  --new-message "Debugging multi-arch test failure for issue #${ISSUE_NUMBER} (attempt $((DEBUG_CYCLES+1))/${MAX_DEBUG_CYCLES})" \
  --prev-result "test_failure" \
  --prev-message "$FAILURE_SUMMARY" \
  --agent "tester"
```

The retry worker reads the existing plan plus the combined tester evidence, patches the shared implementation or narrows arch-specific scope when evidence supports that, and writes `${LOG_DIR}/agent_issue_worker_debug.md`. After it returns, increment `DEBUG_CYCLES`, then re-run the suite that failed (Step 4 for a tt-llk failure, Step 4b for a metal failure). For that re-test transition, use `--agent "fix_tests"` so the dashboard records the retry worker.

Repeat the loop while any arch is still red and `DEBUG_CYCLES < MAX_DEBUG_CYCLES`. Terminate the loop when every arch is green (proceed to Step 5.3), when `DEBUG_CYCLES == MAX_DEBUG_CYCLES` and an arch is still red (finalize as `failed` with the tester/worker evidence), or when the worker returns `HYPOTHESIS_REFUTED` (the failure is inherent to the fix, not a scope issue — finalize as `failed` with that evidence instead of continuing to loop).

Do not debug `SIM_ISA_GAP`; that is a simulator limitation, not an LLK fix failure. Mark the affected arch failed with a simulator obstacle. Continue evaluating other arches when possible.

## Step 5.3: Review the Shared Fix and Feed Back

A senior-reviewer pass over the shared multi-arch diff, run as a loop **inside**
the pipeline — same idea as a `code-review` bot, except findings are fed back to
the worker instead of posted to a PR. Review is static (reads the diff only), so
it runs once for the whole run regardless of backend, and cross-arch **parity**
gaps are especially relevant here (a change that landed on one arch but not the
others it should).

Run this whenever a shared fix diff exists and no arch is in an unresolved *failed*
state — i.e. after functional tests are **green** (Step 4/4b), **and also** when
`VERIFY_ROUTE=none` (`VERIFY_DEFERRED=1`): a diff still exists and, with no functional
test possible in-harness, this static review is the only quality gate before the fix
goes to tt-metal CI, so it is especially important there.

Advance to the `review` step and run one reviewer over the shared diff:

```bash
python codegen/scripts/run_json_writer.py advance --log-dir "$LOG_DIR" \
  --new-step "review" \
  --new-message "Reviewing shared fix diff for issue #${ISSUE_NUMBER} across ${TARGET_ARCHES_JSON} (attempt $((REVIEW_RETRIES+1))/$((MAX_REVIEW_RETRIES+1)))" \
  --prev-result "success" --prev-message "Functional tests passed" --agent "tester"
```

Spawn `reviewer.md` once with: issue number, `TARGET_ARCHES`, changed files,
`WORKTREE_DIR`, `LOG_DIR`, and `PR_REVIEW_KNOWLEDGE_DIR`. It writes
`$LOG_DIR/review_result.json` and `$LOG_DIR/agent_reviewer.md`. Patch its result
into `run.json`:

```bash
python codegen/scripts/run_json_writer.py metric --log-dir "$LOG_DIR" \
  --patch-json "{\"review\": $(cat "$LOG_DIR/review_result.json")}"
```

**Review feedback loop (shared fix).** Read `blocking_total`. If `0`, proceed to
Step 5.5. If `> 0` and `REVIEW_RETRIES < MAX_REVIEW_RETRIES`, record a `failure`,
advance to `fix_tests`, and spawn `issue-worker.md` once in
`FAILURE_CLASS=REVIEW_FINDINGS` mode with `$LOG_DIR/review_result.json`. The
worker makes one shared correctness-preserving change addressing the blocking
findings across all arches. Then `REVIEW_RETRIES=$((REVIEW_RETRIES+1))`,
`DEBUG_CYCLES=$((DEBUG_CYCLES+1))`, re-run **Step 4 (functional Test)** for the
affected arches, and if still green re-run this Step 5.3. If the worker returns
`HYPOTHESIS_REFUTED` (the finding cannot be resolved without breaking
correctness), stop the loop and proceed to Step 5.5.

**When the review budget is exhausted** with blocking findings remaining: the run
does **not** fail on the review alone. Keep the functional `combined_status`,
leave `review.verdict=changes_requested` in `run.json`, and set
`OBSTACLE=unresolved_review_findings` as the terminal record. Then proceed to
Step 5.5.

## Step 5.5: Measure Perf Per Arch and Feed Back

Run this only for arches whose functional result is **green**. Perf is gated to
**local Blackhole/Wormhole** — skip Quasar and any non-`local` backend.

```bash
PERF_ARCHES=$(python - <<'PY'
import json, os
arches = json.loads(os.environ["TARGET_ARCHES_JSON"])
backend = os.environ.get("TEST_BACKEND", "")
keep = [a for a in arches if backend == "local" and a in ("blackhole", "wormhole")]
print(" ".join(keep))
PY
)
```

If `PERF_ARCHES` is empty, record `perf` as not measured for the run and skip to
Step 6:

```bash
python codegen/scripts/run_json_writer.py metric --log-dir "$LOG_DIR" \
  --patch-json "{\"perf\": {\"measured\": false, \"verdict\": \"not_measured\", \"reason\": \"perf only runs on local Blackhole/Wormhole silicon\"}}"
```

Otherwise advance to the `perf` step and run the perf-tester **once per
`PERF_ARCHES` entry** whose functional verdict was `SUCCESS`:

```bash
python codegen/scripts/run_json_writer.py advance --log-dir "$LOG_DIR" \
  --new-step "perf" \
  --new-message "Measuring perf for issue #${ISSUE_NUMBER} on ${PERF_ARCHES} (goal=${PERF_GOAL})" \
  --prev-result "success" --prev-message "Functional tests passed" --agent "perf"
```

For each `arch` in `PERF_ARCHES`, spawn `perf-tester.md` with that single
`TARGET_ARCH`, `TEST_BACKEND=local`, `PERF_GOAL`, the changed op, fix plan path,
changed files, `WORKTREE_DIR`, and `LOG_DIR`. The perf-tester writes its result
to the fixed file `$LOG_DIR/perf_result.json`. **Read it immediately** after each
arch (the next arch overwrites it) and record that arch's result under
`arch_results.<arch>.perf` via `metric`:

```bash
python codegen/scripts/run_json_writer.py metric --log-dir "$LOG_DIR" \
  --patch-json "{\"arch_results\": {\"${arch}\": {\"perf\": $(cat "$LOG_DIR/perf_result.json")}}}"
```

**Perf feedback loop (shared fix).** Collect the per-arch perf verdicts. A *miss*
is any `PERF_REGRESSED`, or `PERF_NOT_IMPROVED` when `PERF_GOAL=improve`. If at
least one arch missed and `PERF_RETRIES < MAX_PERF_RETRIES`: record a `failure`,
advance to `fix_tests`, and spawn `issue-worker.md` once in perf mode
(`FAILURE_CLASS=PERF_REGRESSION`/`PERF_NOT_IMPROVED`) with the missing arches and
their `perf_baseline_*`/`perf_current_*` CSV paths. The worker makes one shared
correctness-preserving change. Then `PERF_RETRIES=$((PERF_RETRIES+1))`,
`DEBUG_CYCLES=$((DEBUG_CYCLES+1))`, re-run **Step 4 (functional Test)** for the
affected arches, and if still green re-run this Step 5.5 for them.

**When the perf budget is exhausted** (still a miss, or `HYPOTHESIS_REFUTED`):

- `PERF_GOAL=no_regress` + any arch still regressed → that arch's `arch_results`
  verdict becomes a failure with `obstacle=perf_regression`; fold it into
  `combined_status` (a regressed arch counts as failed for Step 6).
- `PERF_GOAL=improve` + not improved → keep the functional verdict; leave
  `perf.verdict=not_improved` on that arch and note it in the report.

## Step 6: Finalize One Run

Compute each in-scope arch's **effective verdict** from `arch_results` (written by the
tt-llk tester and/or the metal-tester):

- *passed* — `SUCCESS` (tt-llk **or** metal), or `COMPILED_ONLY`
- *unverified-but-fixed* — `UNVERIFIABLE_IN_LLK_SUITE`: no in-harness test exists, the fix
  is applied + committed and belongs in tt-metal CI
- *failed* — `COMPILE_FAILED`, `TESTS_FAILED`, `ENV_ERROR`, `SIM_ISA_GAP`, or a perf miss
  folded in by Step 5.5
- *out-of-scope* — `SKIPPED`: the analyzer found no work for this arch

Pick `combined_status`:

- `success`: every in-scope arch *passed*, at least one via a real functional test
  (tt-llk or metal). A Layer-3 fix that the metal suite verified lands here.
- `compiled`: every in-scope arch is *unverified-but-fixed* or *passed-compiled-only*, and
  none *failed* — a real fix exists but no functional test ran in-harness (a Layer-2/3/4
  change with `VERIFY_ROUTE=none`, or compile-only). **This is the bucket that issue #22943
  was wrongly reported as `skipped`.**
- `partial`: at least one arch *passed*/*unverified-but-fixed* and at least one *failed*
- `failed`: every in-scope arch *failed*
- `skipped`: **only** when the analyzer found no relevant LLK work for any requested arch
  (every arch *out-of-scope*). Never `skipped` when a fix was produced.

`run_json_writer.py finalize` only accepts `success`, `compiled`, `failed`, or `skipped`:

- `status=success` when `combined_status=success`
- `status=compiled` when `combined_status=compiled`
- `status=failed` when `combined_status=partial` or `combined_status=failed`
- `status=skipped` when `combined_status=skipped`

`run_json_writer.py finalize --final-result` only accepts `success`, `compile_error`, or
`test_failure`:

- `final_result=success` for `status=success`, `status=compiled`, or `status=skipped`
- `final_result=test_failure` for `status=failed` from test/runtime/scope/sim/env blockers
- `final_result=compile_error` only when the terminal failure is a compile error

Set `solver_state=working` for `status=success` or `status=compiled`; otherwise
`not_working`.

**Deferred-verification messaging (`VERIFY_DEFERRED=1` → `combined_status=compiled` from
`UNVERIFIABLE_IN_LLK_SUITE`).** This is a Working outcome, so keep `OBSTACLE` empty — do
**not** populate the field the dashboard renders as a red "⚠ Obstacle" box — and carry the
actionable next step in the final message instead:

```bash
if [ "${VERIFY_DEFERRED:-0}" = 1 ]; then
  export OBSTACLE=
  export FINAL_MESSAGE="multi-arch issue #${ISSUE_NUMBER}: fix applied — ${VERIFY_DEFER_NOTE}"
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
# Exclude perf_data/ — regenerated by the perf stage as a measurement artifact.
export CHANGED_FILES=$(git -C "$WORKTREE_DIR" diff --name-only | grep -v '/perf_data/' | grep -v '^perf_data/' || true)
export CHANGED_FILES_JSON=$(python -c "import json,os; print(json.dumps([l for l in os.environ['CHANGED_FILES'].splitlines() if l]))")

# ── Preserve the shared fix durably: local commit (no push) + archived patch ──
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
    commit -q -m "AI issue-solver: multi-arch fix #${ISSUE_NUMBER} ${ISSUE_TITLE}" 2>/dev/null || true
  export FIX_COMMIT=$(git -C "$WORKTREE_DIR" rev-parse HEAD 2>/dev/null || echo "")
fi
# Apply-able patch archived in the durable LOG_DIR. Reapply from the tt-metal
# repo root with:  git checkout $BASE_COMMIT && git apply $LOG_DIR/generated.patch
if [ -n "$FIX_COMMIT" ] && [ "$FIX_COMMIT" != "$BASE_COMMIT" ]; then
  git -C "$WORKTREE_DIR" diff --binary "$BASE_COMMIT" "$FIX_COMMIT" > "$LOG_DIR/generated.patch" 2>/dev/null || true
else
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
  --final-message "${FINAL_MESSAGE:-multi-arch issue #${ISSUE_NUMBER}: ${COMBINED_STATUS}}" \
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
    ("metal_test", "agent_metal_tester.md"),
    ("reviewer", "agent_reviewer.md"),
    ("perf", "agent_perf_tester.md"),
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
    # Durability: where the shared fix lives after the run.
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
Multi-Arch Issue-Solver Result:
  status: success|compiled|failed|skipped
  combined_status: success|partial|failed|skipped
  codegen_version: ${CODEGEN_VERSION}
  run_id: ${RUN_ID}
  log_dir: ${LOG_DIR}
  branch: ${WORKTREE_BRANCH}            # shared fix committed here (local, NOT pushed)
  base_commit: ${BASE_COMMIT}           # origin/main SHA the branch was cut from
  fix_commit: ${FIX_COMMIT}             # the local fix commit (empty if no change)
  worktree_dir: ${WORKTREE_DIR}         # where the run executed; removed after finish (recover via branch or patch)
  patch: ${LOG_DIR}/generated.patch     # reapply: git checkout <base_commit> && git apply <patch>
  target_arches:
    - ...
  test_backend: ${TEST_BACKEND}
  perf_goal: ${PERF_GOAL}
  create_local_branch_requested: ${CREATE_LOCAL_BRANCH}
  create_pr_requested: ${CREATE_PR}
  arch_results:
    - arch: blackhole|wormhole|quasar
      verdict: SUCCESS|COMPILE_FAILED|TESTS_FAILED|SIM_ISA_GAP|ENV_ERROR|COMPILED_ONLY|UNVERIFIABLE_IN_LLK_SUITE|SKIPPED
      tests_total: N
      tests_passed: N
      suite: llk|metal           # which suite produced the verdict (metal = unit_tests_llk gtest)
      perf_verdict: improved|neutral|regressed|not_improved|no_baseline|not_measured
      obstacle: ...
  review:                         # one review over the shared diff (run-level, not per-arch)
    verdict: ...                  # clean | changes_requested | not_reviewed
    findings_total: ...           # review.findings_total
    blocking_total: ...           # review.blocking_total (0 once the loop converges)
    retries_used: ${REVIEW_RETRIES}/${MAX_REVIEW_RETRIES}
    advisory:                     # non-blocking findings recorded, not acted on (nits/parity/style)
      - <severity> <file>:<line> — <title>
  cost:
    tokens: ...                   # "in=<n> out=<n> cache_read=<n> cache_creation=<n>"
    total_tokens: ...             # tokens.total (input + output, whole run, all arches)
    est_usd: ...                  # tokens.cost_usd (estimate), or "n/a"
  changed_files:
    ...
  obstacle: ...
```

Populate the `review:` block from the `review` object in `run.json` (written by
the reviewer). List every `blocking: false` finding under `advisory:`.

Populate the `cost:` block from the `tokens` object in `run.json` (written by
the `session_cost.py` refreshes); `est_usd: n/a` if the session could not be
discovered.
