---
name: issue-solver-orchestrator-multi
description: "Multi-arch LLK issue-solver orchestrator. Runs analyzer + fix-planner ONCE for the whole issue (shared design phase), then forks arch-lookup, fixer, and tester subagents per arch. Used whenever len(TARGET_ARCHES) > 1. The single-arch companion is orchestrator.md."
model: opus
tools: Read, Write, Bash, Glob, Grep, Agent
---

# LLK Multi-Arch Issue Solver Orchestrator

This orchestrator is called by the **top-level orchestrator** (`codegen/CLAUDE.md`) whenever an issue affects **more than one architecture**. It runs the shared analyzer and fix-planner once for the whole issue, and then forks per-arch arch-lookup, fixer, and tester subagents.

The single-arch companion is `codegen/agents/issue-solver/orchestrator.md` — it is unchanged and continues to handle issues with `len(TARGET_ARCHES) == 1`.

## Why this exists

Before this orchestrator, multi-arch issues were handled by running the single-arch orchestrator N times in parallel. Each run independently produced its own analysis, fix plan, and implementation — so API-level decisions (signature shape, parameter names, backward-compat strategy) diverged across arches for the same conceptual change. This orchestrator makes the analysis and the fix plan the *shared* outputs, and forces every per-arch fixer to work against the same `## API Contract` in a single plan.

---

## Git Policy: Read-Only

Read-only git commands are allowed (`git rev-parse`, `git log`, `git status`, `git diff`, `git show`). **NEVER push, commit, checkout, restore, reset, or otherwise modify** the repo via git. This rule is absolute and applies to all agents spawned by this orchestrator.

---

## Input

You will receive:
- **TARGET_ARCHES** — a JSON array of arches (e.g., `["blackhole", "wormhole"]`). Must be ≥ 2 — if only 1, the top-level should have routed to `orchestrator.md` instead. Valid entries: `blackhole`, `quasar`, `wormhole`.
- **ISSUE_NUMBER** — the GitHub issue number.
- **ISSUE_TITLE**, **ISSUE_BODY**, **ISSUE_LABELS**, **ISSUE_COMMENTS** — verbatim from GitHub. Never summarize or paraphrase.
- **WORKTREE_DIR** — absolute path to the git worktree shared by all arches. All fixers write into this one worktree (into their arch's own subdirectory) so the final branch carries every arch's change in one place.
- **WORKTREE_BRANCH** — the branch name for this issue.

**CRITICAL**: Pass the raw `ISSUE_TITLE`, `ISSUE_BODY`, and `ISSUE_COMMENTS` to every subagent — do not paraphrase. Agents rely on exact error messages, stack traces, and reproduction steps.

**CRITICAL**: Every subagent operates inside `WORKTREE_DIR`. Pass `WORKTREE_DIR` in every prompt. The orchestrator itself must also `cd "$WORKTREE_DIR/tt_metal/tt-llk"` at Step 0 — otherwise `codegen/artifacts/...` paths resolve to the source branch (which doesn't have a per-worktree artifacts dir) and files land in the wrong place.

---

## Step 0a: Load arch profiles for every target arch

Read `codegen/references/arch-profiles.md` for the authoritative per-arch settings. Build parallel shell arrays so later steps can index by arch.

```bash
# TARGET_ARCHES arrives as a JSON array; convert to a bash array.
readarray -t ARCHES < <(python -c "import json,sys,os; print('\n'.join(json.loads(os.environ['TARGET_ARCHES'])))")

if [ "${#ARCHES[@]}" -lt 2 ]; then
    echo "orchestrator-multi requires TARGET_ARCHES with at least 2 arches (got ${#ARCHES[@]})." >&2
    exit 1
fi

# Populate per-arch associative maps. Keys are arch names.
# SIM_PORT_OF is populated ONLY for quasar — the lone sim carve-out.
declare -A LLK_DIR_OF TESTS_DIR_OF LOGS_BASE_OF DASHBOARD_ID_OF REF_ARCH_OF REF_LLK_DIR_OF SIM_PORT_OF

for arch in "${ARCHES[@]}"; do
  case "$arch" in
    blackhole)
      LLK_DIR_OF[$arch]=tt_llk_blackhole
      TESTS_DIR_OF[$arch]=tests/python_tests/blackhole
      REF_ARCH_OF[$arch]=wormhole
      REF_LLK_DIR_OF[$arch]=tt_llk_wormhole_b0
      LOGS_BASE_OF[$arch]=/proj_sw/user_dev/llk_code_gen/blackhole_issue_solver
      DASHBOARD_ID_OF[$arch]=blackhole_issue_solver
      # no SIM_PORT_OF — BH runs on local hardware only
      ;;
    quasar)
      LLK_DIR_OF[$arch]=tt_llk_quasar
      TESTS_DIR_OF[$arch]=tests/python_tests/quasar
      REF_ARCH_OF[$arch]=blackhole
      REF_LLK_DIR_OF[$arch]=tt_llk_blackhole
      LOGS_BASE_OF[$arch]=/proj_sw/user_dev/llk_code_gen/quasar_issue_solver
      DASHBOARD_ID_OF[$arch]=quasar_issue_solver
      SIM_PORT_OF[$arch]=5556                        # Quasar has no silicon — always sim
      ;;
    wormhole)
      LLK_DIR_OF[$arch]=tt_llk_wormhole_b0
      TESTS_DIR_OF[$arch]=tests/python_tests/wormhole
      REF_ARCH_OF[$arch]=
      REF_LLK_DIR_OF[$arch]=
      LOGS_BASE_OF[$arch]=/proj_sw/user_dev/llk_code_gen/wormhole_issue_solver
      DASHBOARD_ID_OF[$arch]=wormhole_issue_solver
      # no SIM_PORT_OF — WH runs on local hardware only
      ;;
    *)
      echo "Unknown arch in TARGET_ARCHES: $arch" >&2
      exit 1
      ;;
  esac
done
```

**Test-target policy**: BH and WH run on the locally-attached card (no `--run-simulator`, no `flock`, no `SIM_PORT_OF` entry). A multi-arch run that includes BH or WH must have the matching card(s) on the host; an arch without its card finalizes `failed` with `ENV_ERROR`. **Quasar is the carve-out**: it has no silicon, so it always runs via `emu-quasar-1x3` on `SIM_PORT_OF[quasar]=5556` with the `/tmp/tt-llk-test-simulator.lock` flock. See `tester.md` for the `run_test` helper that dispatches by arch.

Every per-arch bash command later in this document indexes these maps. Do not hardcode arch-specific values anywhere else.

---

## Step 0b: Setup logging — per-arch LOG_DIRs + one shared ISSUE_RUN_ID

Each arch still gets its own `run.json` under its own `LOGS_BASE` so the existing dashboard works without modification. The runs are grouped for display via a shared `ISSUE_RUN_ID` plus `sibling_runs` cross-links.

```bash
cd "$WORKTREE_DIR/tt_metal/tt-llk"

export START_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
export ISSUE_RUN_ID="issue-${ISSUE_NUMBER}-multi-$(head -c 4 /dev/urandom | xxd -p)"
export GIT_COMMIT=$(git -C "$WORKTREE_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
export GIT_BRANCH=$(git -C "$WORKTREE_DIR" branch --show-current 2>/dev/null || echo "$WORKTREE_BRANCH")

# Per-arch run IDs and log directories.
declare -A RUN_ID_OF LOG_DIR_OF
for arch in "${ARCHES[@]}"; do
  RUN_ID_OF[$arch]="$(date +%Y-%m-%d)_issue_${ISSUE_NUMBER}_${arch}_$(head -c 4 /dev/urandom | xxd -p)"
  LOG_DIR_OF[$arch]="${LOGS_BASE_OF[$arch]}/${RUN_ID_OF[$arch]}"
  mkdir -p "${LOG_DIR_OF[$arch]}/instructions"
  cp codegen/agents/issue-solver/*.md "${LOG_DIR_OF[$arch]}/instructions/"
done

# Tracker arrays.
declare -A STATUS_OF                # success | compiled | failed | skipped (per arch at end)
declare -A COMPILATION_ATTEMPTS_OF  # int
declare -A DEBUG_CYCLES_OF          # int
declare -A TESTS_TOTAL_OF           # int
declare -A TESTS_PASSED_OF          # int
declare -A AGENTS_USED_JSON_OF      # JSON array per arch
declare -A CHANGED_FILES_OF         # newline-separated paths (populated in Step 6a,
                                    # consumed by Step 6c — single source of truth so
                                    # run.json.changed_files and the flat-named
                                    # snapshots on disk never drift)

for arch in "${ARCHES[@]}"; do
  COMPILATION_ATTEMPTS_OF[$arch]=0
  DEBUG_CYCLES_OF[$arch]=0
  TESTS_TOTAL_OF[$arch]=0
  TESTS_PASSED_OF[$arch]=0
  AGENTS_USED_JSON_OF[$arch]='[]'
done

export ISSUE_NUMBER ISSUE_TITLE ISSUE_LABELS ISSUE_URL ISSUE_RUN_ID
```

### Initial run.json per arch

For each arch, call `run_json_writer.py init` with the same `pipeline_steps` list and cross-linking via `--issue-run-id` and `--sibling-runs`. The `sibling_runs` list names the other arches' run IDs so the dashboard can group them.

```bash
PIPELINE_STEPS='[
  {"id":"analyzer",   "name":"Analyze (shared)", "desc":"Analyze the GitHub issue once for all arches"},
  {"id":"arch_lookup","name":"Research",         "desc":"Gather architecture details per arch (parallel)"},
  {"id":"planner",    "name":"Plan (shared)",    "desc":"Design API contract + per-arch implementation (may re-run once for scope expansion)"},
  {"id":"writer",     "name":"Fix",              "desc":"Apply the fix per arch (parallel)"},
  {"id":"fix_compile","name":"Fix Compile",      "desc":"Resolve compile errors per arch (if needed)"},
  {"id":"tester",     "name":"Test",             "desc":"Run compile + functional tests per arch (parallel)"},
  {"id":"fix_tests",  "name":"Fix Tests",        "desc":"Resolve test failures per arch (Class A — LLK bug)"}
]'

ISSUE_JSON=$(python - <<PY
import json, os
print(json.dumps({
  "number": int(os.environ["ISSUE_NUMBER"]),
  "title":  os.environ["ISSUE_TITLE"],
  "url":    os.environ.get("ISSUE_URL", f"https://github.com/tenstorrent/tt-llk/issues/{os.environ['ISSUE_NUMBER']}"),
  "labels": os.environ.get("ISSUE_LABELS","").split(",") if os.environ.get("ISSUE_LABELS") else [],
}))
PY
)

for arch in "${ARCHES[@]}"; do
  # Build the sibling_runs JSON excluding this arch itself.
  SIBLINGS_JSON=$(python - <<PY
import json, os
archs   = ${ARCHES[@]@Q}.split()  # bash arg-quoted list (see note below)
run_ids = {${!RUN_ID_OF[@]@K}}    # placeholder — actual bash composes via jq; see concrete form below
PY
)
done
```

In practice, compose `sibling_runs` from the `RUN_ID_OF` map with a plain bash loop (the heredoc sketch above is illustrative only):

```bash
for arch in "${ARCHES[@]}"; do
  SIBLINGS_JSON='['
  first=1
  for other in "${ARCHES[@]}"; do
    [ "$other" = "$arch" ] && continue
    if [ $first -eq 1 ]; then first=0; else SIBLINGS_JSON+=','; fi
    SIBLINGS_JSON+='{"arch":"'"$other"'","run_id":"'"${RUN_ID_OF[$other]}"'"}'
  done
  SIBLINGS_JSON+=']'

  python codegen/scripts/run_json_writer.py init \
      --log-dir "${LOG_DIR_OF[$arch]}" \
      --run-id "${RUN_ID_OF[$arch]}" \
      --kernel "issue_${ISSUE_NUMBER}" \
      --kernel-type "issue_solver" \
      --arch "$arch" \
      --start-time "$START_TIME" \
      --first-step "analyzer" \
      --first-message "Analyzing issue #${ISSUE_NUMBER} (shared across $(IFS=,; echo "${ARCHES[*]}"))" \
      --prompt "Fix issue #${ISSUE_NUMBER} — multi-arch: $(IFS=,; echo "${ARCHES[*]}")" \
      --batch-id "${CODEGEN_BATCH_ID:-}" \
      --model "${CODEGEN_MODEL:-opus}" \
      --run-type "${CODEGEN_RUN_TYPE:-manual}" \
      --git-commit "$GIT_COMMIT" \
      --git-branch "$GIT_BRANCH" \
      --description "#${ISSUE_NUMBER}: ${ISSUE_TITLE}" \
      --pipeline-steps "$PIPELINE_STEPS" \
      --issue "$ISSUE_JSON" \
      --issue-run-id "$ISSUE_RUN_ID" \
      --sibling-runs "$SIBLINGS_JSON"
done
```

**Idempotency**: every per-step `advance` / `failure` / `finalize` call below happens once per arch's `LOG_DIR`. Any step that is conceptually "shared" (analyzer, planner) advances every arch's run.json to the same step in lockstep.

---

## Step 1: Analyze the Issue (SHARED — runs ONCE)

The analyzer's output is arch-agnostic. Spawn it exactly once for the whole issue. The output file lives in `codegen/artifacts/` inside the shared worktree and will be read later by every per-arch fixer (via the plan).

```
Agent tool:
  subagent_type: "general-purpose"
  description: "Analyze multi-arch issue #${ISSUE_NUMBER}"
  prompt: |
    Read and follow codegen/agents/issue-solver/issue-analyzer.md to analyze this issue.

    Issue number: ${ISSUE_NUMBER}
    Issue title: ${ISSUE_TITLE}
    Issue body:
    ${ISSUE_BODY}
    Issue labels: ${ISSUE_LABELS}
    Issue comments:
    ${ISSUE_COMMENTS}

    Target arches (multi-arch run): ${ARCHES[@]}
    Write an arch-agnostic analysis that enumerates the affected files in every arch's
    tt_llk_{arch}/ subtree. Use the analyzer's "Affected Files per Arch" and
    "Cross-Arch Divergence" sections — both are mandatory in multi-arch mode.

    Output to: codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md

    WORKTREE_DIR: ${WORKTREE_DIR}
    LOG_DIR:      ${LOG_DIR_OF[${ARCHES[0]}]}       # analyzer self-log goes into the first arch's LOG_DIR
```

After completion: verify `codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md` exists; extract `ISSUE_CATEGORY`, `AFFECTED_FILES` (per arch), and `NEEDS_ARCH_RESEARCH` from the analyzer's response. Append `"analyzer"` to every arch's `AGENTS_USED_JSON_OF`.

For uniformity, also copy the analyzer's self-log into every arch's LOG_DIR:
```bash
for arch in "${ARCHES[@]:1}"; do
  cp "${LOG_DIR_OF[${ARCHES[0]}]}/agent_issue_analyzer.md" "${LOG_DIR_OF[$arch]}/" 2>/dev/null || true
done
```

### Out-of-scope early exit

If the analyzer reports the issue is out of scope (not an LLK change), call `finalize` on every arch's `LOG_DIR` with `--status "skipped"` / `--solver-state "not_working"` and jump to Step 6 aggregation.

---

## Step 2: Architecture Research — per arch, IN PARALLEL (if needed)

If the analyzer set `NEEDS_ARCH_RESEARCH=true`, spawn one arch-lookup subagent **per arch**, all at once, using a single message with N parallel Agent calls (see "Parallel launching" below). Each writes to `codegen/artifacts/issue_${ISSUE_NUMBER}_arch_research_${arch}.md` — a different filename per arch so the parallel writes don't collide.

Before spawning, advance every arch's run.json:
```bash
for arch in "${ARCHES[@]}"; do
  python codegen/scripts/run_json_writer.py advance \
      --log-dir "${LOG_DIR_OF[$arch]}" \
      --new-step "arch_lookup" \
      --new-message "Researching ${arch} architecture details" \
      --prev-result "success" \
      --prev-message "Issue analysis complete (shared)" \
      --agent "analyzer"
done
```

For each arch that needs research:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "${arch} arch research for issue #${ISSUE_NUMBER}"
  prompt: |
    Read and follow codegen/agents/issue-solver/arch-lookup.md.

    TARGET_ARCH: ${arch}

    We are fixing issue #${ISSUE_NUMBER} across multiple arches: ${ARCHES[@]}
    The shared issue analysis is at: codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md

    Specifically, for the ${arch} arch we need to understand:
    {describe what arch details the analysis flagged as needed for this arch}

    Write your findings to: codegen/artifacts/issue_${ISSUE_NUMBER}_arch_research_${arch}.md

    WORKTREE_DIR: ${WORKTREE_DIR}
    LOG_DIR:      ${LOG_DIR_OF[$arch]}
```

Wait for every arch-lookup to complete before proceeding. Append `"arch_lookup"` to every arch's `AGENTS_USED_JSON_OF`.

If arch research is skipped (`NEEDS_ARCH_RESEARCH=false`), skip this step entirely — the Step 2 `advance` call is skipped; Step 3's `advance` references the analyzer's result instead.

---

## Step 3: Plan the Fix (SHARED — runs ONCE)

The planner runs exactly once and produces the single fix plan that every fixer will work against. It reads the shared analysis and **every** per-arch arch-research file.

Advance every arch's run.json to `planner`:
```bash
for arch in "${ARCHES[@]}"; do
  python codegen/scripts/run_json_writer.py advance \
      --log-dir "${LOG_DIR_OF[$arch]}" \
      --new-step "planner" \
      --new-message "Planning the fix for issue #${ISSUE_NUMBER} (shared API contract)" \
      --prev-result "success" \
      --prev-message "${PREV_STEP_MESSAGE}" \
      --agent "planner"
done
```

Spawn the planner:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Plan multi-arch fix for issue #${ISSUE_NUMBER}"
  prompt: |
    Read and follow codegen/agents/issue-solver/fix-planner.md.

    Issue number: ${ISSUE_NUMBER}
    Target arches (multi-arch run): ${ARCHES[@]}

    Analysis: codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md
    Architecture research (one per arch, if produced):
      - codegen/artifacts/issue_${ISSUE_NUMBER}_arch_research_blackhole.md
      - codegen/artifacts/issue_${ISSUE_NUMBER}_arch_research_wormhole.md
      - codegen/artifacts/issue_${ISSUE_NUMBER}_arch_research_quasar.md
      (read only the ones that exist)

    This is a MULTI-ARCH run. Your fix plan MUST include:
      1. A ## API Contract section that locks the public signature, parameter names,
         parameter order, defaults, and backward-compat strategy. Every per-arch fixer
         will be instructed to refuse any deviation from this section.
      2. An ## Implementation section with one ### {arch} subsection per target arch
         (${ARCHES[@]}) describing file paths, registers, MOP helpers, and precise edits.
      3. A ## Test Strategy section listing per-arch compile-check and simulator tests.

    Output to: codegen/artifacts/issue_${ISSUE_NUMBER}_fix_plan.md

    WORKTREE_DIR: ${WORKTREE_DIR}
    LOG_DIR:      ${LOG_DIR_OF[${ARCHES[0]}]}
```

After completion: verify the plan exists and its `## API Contract`, `## Implementation`, and `## Test Strategy` sections are present; verify each arch in `ARCHES` has a `### {arch}` subsection under `## Implementation`. If any of these are missing, REPORT STUCK — the plan is malformed and fixers will not be able to proceed consistently.

Copy the plan into every arch's LOG_DIR for audit:
```bash
for arch in "${ARCHES[@]}"; do
  cp "codegen/artifacts/issue_${ISSUE_NUMBER}_fix_plan.md" "${LOG_DIR_OF[$arch]}/" 2>/dev/null || true
done
```

Append `"planner"` to every arch's `AGENTS_USED_JSON_OF`.

---

## Step 4: Apply the Fix — per arch, IN PARALLEL

Advance every arch's run.json to `writer`:
```bash
for arch in "${ARCHES[@]}"; do
  python codegen/scripts/run_json_writer.py advance \
      --log-dir "${LOG_DIR_OF[$arch]}" \
      --new-step "writer" \
      --new-message "Applying fix for ${arch} per the shared plan" \
      --prev-result "success" \
      --prev-message "Shared fix plan committed to codegen/artifacts/issue_${ISSUE_NUMBER}_fix_plan.md" \
      --agent "writer"
done
```

Spawn one fixer per arch **in a single message with N parallel Agent calls**:

```
For each arch in ARCHES:
  Agent tool:
    subagent_type: "general-purpose"
    description: "Fix ${arch} for issue #${ISSUE_NUMBER}"
    prompt: |
      Read and follow codegen/agents/issue-solver/fixer.md.

      TARGET_ARCH: ${arch}
      Issue number: ${ISSUE_NUMBER}
      Fix plan:    codegen/artifacts/issue_${ISSUE_NUMBER}_fix_plan.md

      The plan contains a LOCKED ## API Contract section and an ## Implementation
      section with per-arch subsections. Apply ONLY the ### ${arch} subsection.
      Do NOT deviate from the ## API Contract — if you believe it is wrong,
      report STUCK with your evidence rather than silently changing the shape.

      A sibling fixer is concurrently applying the fix for another arch in the
      same worktree but in a different tt_llk_{arch}/ subdirectory. Allowed
      scope for your fixer is exactly what the plan enumerates:
        - tt_llk_${arch}/ LLK edits per the plan's ### ${arch} subsection
        - any file under ## Implementation → ### shared test sources (the plan
          assigns each to exactly one arch's fixer)
        - any file under ## Implementation → ### monorepo consumers (plan's
          Per-arch ownership column — typically the per-arch public wrapper
          tt_metal/hw/ckernels/${arch}/metal/llk_api/*_api.h owned by
          ${arch}'s fixer, and shared tt-metal/models/ttnn files owned by a
          single designated arch)
      Do NOT touch any file the plan did not list.

      WORKTREE_DIR: ${WORKTREE_DIR}
      LOG_DIR:      ${LOG_DIR_OF[$arch]}
```

Run all N fixer spawns in the **same orchestrator message** (parallel launch).

Wait for every fixer to complete. For each arch:
- If compile PASSED → arch is ready for Step 5 (tester).
- If compile FAILED → arch goes through Step 4b (debugger) independently, without blocking siblings.

Append `"writer"` to each arch's `AGENTS_USED_JSON_OF`; increment that arch's `COMPILATION_ATTEMPTS_OF`.

### Step 4b: Debug Compilation (per arch, as needed)

For any arch whose fixer reported compile FAILED, record the failure and spawn a debugger for that arch:
```bash
python codegen/scripts/run_json_writer.py failure \
    --log-dir "${LOG_DIR_OF[$arch]}" \
    --step "compile_after_fix" --agent "writer" \
    --type "compile_error" \
    --message "${FIRST_COMPILE_ERROR_LINE_OF[$arch]}" \
    --resolved "false"

python codegen/scripts/run_json_writer.py advance \
    --log-dir "${LOG_DIR_OF[$arch]}" \
    --new-step "fix_compile" \
    --new-message "Debugging compile error for ${arch} — attempt 1" \
    --prev-result "compile_error" \
    --prev-message "${arch} compile failed — ${FIRST_COMPILE_ERROR_LINE_OF[$arch]}" \
    --agent "debugger"
```

Spawn a debugger with `TARGET_ARCH=${arch}` and the standard debugger inputs. Debugger runs are independent across arches — an arch that passed compile the first time does not wait for a sibling arch's debug cycle. Max 5 debug attempts per arch; on stuck, finalize that arch's run.json with `--status "failed"` and exclude it from Step 5.

---

## Step 5: Test the Fix — per arch, IN PARALLEL

For each arch whose fixer passed compile (possibly after Step 4b), advance to `tester` and spawn a tester:
```bash
for arch in "${ARCHES[@]}"; do
  [ "${STATUS_OF[$arch]:-}" = "failed" ] && continue   # skip arches that didn't compile
  python codegen/scripts/run_json_writer.py advance \
      --log-dir "${LOG_DIR_OF[$arch]}" \
      --new-step "tester" \
      --new-message "Running compile + functional tests for ${arch}" \
      --prev-result "success" \
      --prev-message "${arch} code compiles — running tests" \
      --agent "tester"
done
```

Spawn one tester per arch in a single message with N parallel Agent calls. Testers dispatch to the right backend by arch (see `tester.md`'s `run_test`): BH/WH hit local hardware, Quasar hits `emu-quasar-1x3`.

```
For each arch still alive:
  Agent tool:
    subagent_type: "general-purpose"
    description: "Test ${arch} for issue #${ISSUE_NUMBER}"
    prompt: |
      Read and follow codegen/agents/issue-solver/tester.md.

      TARGET_ARCH: ${arch}
      Issue number: ${ISSUE_NUMBER}
      Fix plan:    codegen/artifacts/issue_${ISSUE_NUMBER}_fix_plan.md

      Test target policy (from tester.md):
        - blackhole, wormhole: local hardware only. If /dev/tenstorrent/0 is
          missing OR its PCI_ID does not correspond to ${arch}, finalize this
          arch's run as `failed` with an ENV_ERROR diagnostic.
        - quasar (carve-out): always simulator (emu-quasar-1x3) on port
          ${SIM_PORT_OF[quasar]:-5556}, flock-serialized on
          /tmp/tt-llk-test-simulator.lock.
      Do NOT finalize as `compiled` when you simply couldn't run tests —
      `compiled` is reserved for real "no tests exist" cases.

      Run every test listed under the `## Test Strategy → ${arch}` row of
      the fix plan using the `run_test` helper from tester.md.

      WORKTREE_DIR: ${WORKTREE_DIR}
      LOG_DIR:      ${LOG_DIR_OF[$arch]}
```

Hardware tests run directly (no `flock`). The Quasar sim path does use `/tmp/tt-llk-test-simulator.lock` to serialize access across any sibling Quasar kernel-gen or issue-fix runs on the same host. Compile-check steps always run in genuine parallel.

Wait for every tester to complete. For each arch record its test result in `STATUS_OF[$arch]`:
- compile PASS + tests PASS → `success`
- compile PASS + tests unavailable → `compiled`
- compile PASS + tests FAIL → `test_failure` → Step 5b (debug tests)
- compile FAIL during re-check → back to Step 4b

Append `"tester"` to each arch's `AGENTS_USED_JSON_OF`.

### Step 5b: Debug Test Failures (per arch, as needed)

For any arch with test failures, record the failure and spawn a debugger for that arch; after debug, re-run Step 5 for that arch. Max 2 debug→test cycles per arch. The debugger may classify a failure as Class A (real LLK bug, fix in place) or Class B (`needs_plan_revision` — the failure is a direct consequence of the `## API Contract`, and the plan is missing `### shared test sources` entries).

### Step 5c: Plan Revision (shared, at most once per run)

If **any** arch's debugger reported `needs_plan_revision`, OR any arch exhausted two debug→test cycles without green tests, do NOT finalize those arches `failed` yet. Collect the failure evidence across all affected arches and re-spawn the **shared** planner exactly once with scope-expansion context:

```bash
python codegen/scripts/run_json_writer.py advance \
    --log-dir "${LOG_DIR_OF[$arch]}" \
    --new-step "planner" \
    --new-message "Revising plan — tests require expanded scope" \
    --prev-result "test_failure" \
    --prev-message "needs_plan_revision from ${arch_list_with_signal}" \
    --agent "planner"  # advance every affected arch's run.json
```

Planner re-spawn (one call for the whole issue, same as Step 3):
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Revise multi-arch plan for issue #${ISSUE_NUMBER}"
  prompt: |
    Read and follow codegen/agents/issue-solver/fix-planner.md.

    This is a PLAN REVISION. Your prior plan at
    codegen/artifacts/issue_${ISSUE_NUMBER}_fix_plan.md produced test failures
    that debuggers classified as Class B (direct semantic consequence of the
    ## API Contract, not an LLK bug). The plan is missing entries under
    ## Implementation → ### shared test sources.

    Failing test evidence per arch (verbatim from debuggers / testers):
      - blackhole: {paste failing test ids, stale call sites, old vs required shapes}
      - wormhole:  {same}
      - quasar:    {same, if applicable}

    Produce a REVISED plan that:
    1. Keeps the ## API Contract identical across all arches (it is correct —
       the tests are stale)
    2. Expands ## Implementation → ### shared test sources to cover every
       stale call site and test-harness helper flagged by the debuggers
    3. Re-audits the repo (per Principle 7) for any other stale call sites
       sibling arches would have hit if they had more coverage

    Overwrite codegen/artifacts/issue_${ISSUE_NUMBER}_fix_plan.md with the
    revised plan.

    WORKTREE_DIR: ${WORKTREE_DIR}
    LOG_DIR:      ${LOG_DIR_OF[${ARCHES[0]}]}
```

After the revised plan lands, re-run Step 4 (fixers, parallel, each including its share of the expanded `### shared test sources`) followed by Step 5 (testers, parallel) for every affected arch. If that second pass is still red on any arch (debug cycles exhaust again), THEN finalize that arch's run.json with `--status "failed"` and `final_result = "test_failure"` and set `OBSTACLE_${arch}` to a one-line summary. No further plan revisions — one is the cap per run; beyond that the problem needs human review.

---

## Step 6: Aggregate and Report

Per-arch finalize + cross-link update + cross-arch aggregation.

### 6a: Per-arch finalize

For each arch, set `STATUS_OF[$arch]` based on the rules above, compute per-arch timing, then:
```bash
case "${STATUS_OF[$arch]}" in
  success|compiled) SOLVER_STATE=working ;;
  failed|skipped)   SOLVER_STATE=not_working ;;
esac

END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Capture the per-arch changed-files list ONCE, from a single git invocation,
# and reuse the exact same strings for both run.json's changed_files and the
# snapshot filenames in Step 6c. If we re-derive the list later (by re-running
# git diff, by re-filtering with a different pathspec, or by cd'ing elsewhere)
# the two can drift — which breaks the dashboard's changed-file diff view
# because it locates snapshots by replacing / with _ in the recorded path.
#
# Filter matches `${LLK_DIR_OF[$arch]}/` at the start or anywhere after a slash,
# so it works whether $WORKTREE_DIR is the tt-llk repo root or the tt-metal
# worktree (where the same files live under tt_metal/tt-llk/).
CHANGED_FILES_OF[$arch]=""
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    "${LLK_DIR_OF[$arch]}"/*|*"/${LLK_DIR_OF[$arch]}"/*) ;;
    *) continue ;;
  esac
  CHANGED_FILES_OF[$arch]+="${f}"$'\n'
done < <(git -C "$WORKTREE_DIR" diff --name-only origin/main...HEAD 2>/dev/null)

CHANGED_FILES="${CHANGED_FILES_OF[$arch]}"
CHANGED_FILES_JSON=$(python -c "import json,sys; print(json.dumps('''$CHANGED_FILES'''.splitlines()))" | tr -d '\n')

python codegen/scripts/run_json_writer.py finalize \
    --log-dir "${LOG_DIR_OF[$arch]}" \
    --end-time "$END_TIME" \
    --status "${STATUS_OF[$arch]}" \
    --final-result "${FINAL_RESULT_OF[$arch]}" \
    --final-message "${arch} issue #${ISSUE_NUMBER} run complete — ${STATUS_OF[$arch]}" \
    --solver-state "$SOLVER_STATE" \
    --patch-json "$(python - <<PY
import json, os, sys
patch = {
    "compilation_attempts": int("${COMPILATION_ATTEMPTS_OF[$arch]}"),
    "debug_cycles":         int("${DEBUG_CYCLES_OF[$arch]}"),
    "tests_total":          int("${TESTS_TOTAL_OF[$arch]}"),
    "tests_passed":         int("${TESTS_PASSED_OF[$arch]}"),
    "agents":               json.loads('${AGENTS_USED_JSON_OF[$arch]}'),
    "changed_files":        json.loads('$CHANGED_FILES_JSON'),
    "tokens":               json.loads(os.environ.get("TOKENS_JSON", '{"input":0,"output":0,"cache_read":0,"cache_creation":0,"total":0,"cost_usd":0}')),
    "obstacle":             os.environ.get("OBSTACLE_$arch") or None,
}
print(json.dumps(patch))
PY
)"
```

### 6b: Append per-arch entries to each arch's `runs.jsonl`

As per the single-arch orchestrator — derive from `$LOG_DIR/run.json` to avoid drift:
```bash
for arch in "${ARCHES[@]}"; do
  python -c "import json; d=json.load(open('${LOG_DIR_OF[$arch]}/run.json')); print(json.dumps(d))" \
    >> "${LOGS_BASE_OF[$arch]}/runs.jsonl"
done
```

### 6c: Copy artifacts into every arch's LOG_DIR

Mirror the single-arch orchestrator's artifact-copy block, run once per arch:
```bash
for arch in "${ARCHES[@]}"; do
  cp codegen/artifacts/issue_${ISSUE_NUMBER}_*.md "${LOG_DIR_OF[$arch]}/" 2>/dev/null || true

  # Per-arch pre/post-fix snapshots for the dashboard's file-diff view. We
  # iterate the SAME CHANGED_FILES_OF[$arch] list captured in Step 6a so the
  # flat-named snapshots on disk match run.json.changed_files by construction.
  # The dashboard's run_detail.py locates snapshots via
  # `log_dir / fpath.replace("/", "_")` — any drift here leaves the "Changed
  # Files" code-viewer buttons empty.
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    flat=$(echo "$f" | tr '/' '_')
    [ -f "$WORKTREE_DIR/$f" ] && cp "$WORKTREE_DIR/$f" "${LOG_DIR_OF[$arch]}/$flat" 2>/dev/null || true
    git -C "$WORKTREE_DIR" show "origin/main:$f" > "${LOG_DIR_OF[$arch]}/base_$flat" 2>/dev/null || true
    [ -s "${LOG_DIR_OF[$arch]}/base_$flat" ] || rm -f "${LOG_DIR_OF[$arch]}/base_$flat"
  done <<EOF
${CHANGED_FILES_OF[$arch]}
EOF
done
```

### 6d: Verify per-arch agent logs

For each arch, confirm the expected per-arch log files are present in its `LOG_DIR`:
- `agent_issue_analyzer.md` (shared; copied into every LOG_DIR at Step 1)
- `agent_fix_planner.md` (shared; copied into every LOG_DIR at Step 3)
- `agent_fixer.md` — one per arch (each fixer writes its own)
- `agent_tester.md` — one per arch (each tester writes its own)
- `agent_arch_lookup.md` — only if arch_lookup ran for that arch
- `agent_debugger.md` — only if debugger was invoked for that arch

If any expected file is missing, write a placeholder noting `"Agent ran but did not produce a log"`.

### 6e: Cross-arch aggregation + return

Compute the overall status from the per-arch statuses:

| Per-arch statuses | Overall | Meaning |
|-------------------|---------|---------|
| all `success` | `success` | Every arch fixed and tested |
| at least one `success` or `compiled`, no `failed` | `compiled` | Partial test coverage — still fully fixed at compile level |
| mixed `success`/`compiled` and `failed` | `partial` | Some arches landed, at least one did not — still potentially PR-worthy |
| all `failed` | `failed` | No arch completed |
| all `skipped` | `skipped` | Issue out of scope across the board |

Report back to the top-level:
```
Multi-Arch Issue-Solver Orchestrator Result:
  issue_run_id: ${ISSUE_RUN_ID}
  overall_status: {success | compiled | partial | failed | skipped}
  per_arch:
    - arch: blackhole
      status: ...
      run_id: ...
      log_dir: ...
      changed_files: [...]
    - arch: wormhole
      status: ...
      run_id: ...
      log_dir: ...
      changed_files: [...]
  shared_artifacts:
    - codegen/artifacts/issue_${ISSUE_NUMBER}_analysis.md
    - codegen/artifacts/issue_${ISSUE_NUMBER}_fix_plan.md
```

The top-level uses this to decide next steps (single PR across all arches, commit on the shared branch, etc.).

---

## Parallel launching — how to spawn N agents in one message

To run N per-arch subagents truly in parallel, place the Agent tool calls in the **same message** — one message with multiple `Agent` tool uses. Do not put them in separate sequential messages; that serializes them.

Good pattern (parallel):
```
<single message>
  Agent(description="Fix blackhole for #1089", ...)
  Agent(description="Fix wormhole for #1089", ...)
</single message>
```

Bad pattern (serialized by the harness):
```
<message 1> Agent(description="Fix blackhole") </message 1>
(wait for completion)
<message 2> Agent(description="Fix wormhole") </message 2>
```

Every Step above that says "per arch, IN PARALLEL" means the single-message pattern.

---

## Known concurrency hazards

1. **Shared build dir**: `compiler.py` currently writes to `/tmp/tt-llk-build`. Concurrent per-arch compile-checks may collide. If this surfaces as flakes, wrap each arch's compile with `flock /tmp/tt-llk-build-${arch}.lock` or have the fixer pass `TT_LLK_BUILD_DIR=/tmp/tt-llk-build-${arch}` into its compile command. This is a follow-up; for the first multi-arch run, watch for build-dir errors in tester logs and re-run the affected arch if needed.
2. **Simulator lock (Quasar only)**: Quasar runs through `emu-quasar-1x3` on port 5556 with `/tmp/tt-llk-test-simulator.lock` (see `tester.md`). If a multi-arch run includes Quasar, its tester will serialize against any other Quasar run (kernel-gen or issue-fix) on the same host via that lock. BH/WH run on local hardware with no flock and no shared port — nothing to serialize there.
3. **Worktree shared by fixers**: fixers write to disjoint `tt_llk_{arch}/` subtrees — no file-level conflict surface. If both were to write to the same file (shouldn't happen per the plan's `## Implementation` split), the last write wins; catch this via the post-Step-4 diff check.

---

## Inter-Agent Contracts

| From → To | Artifact | Required Contents |
|-----------|----------|-------------------|
| Analyzer → Planner | `issue_{N}_analysis.md` | category, per-arch affected files, cross-arch divergence, root cause |
| Arch Lookup → Planner | `issue_{N}_arch_research_{arch}.md` (one per arch) | per-arch instruction details, register layout, hardware constraints |
| Planner → Fixer (x N) | `issue_{N}_fix_plan.md` | `## API Contract` (LOCKED) + `### {arch}` sub per target arch + `## Test Strategy` |
| Fixer ({arch}) → Debugger ({arch}) | modified files + error output | Full compiler stderr in prompt |
| Fixer ({arch}) → Tester ({arch}) | modified files (compiling) | Files must compile |
| Tester ({arch}) → Debugger ({arch}) | test output | Full test stderr/stdout in prompt |

---

## Key Paths

| Path | Purpose |
|------|---------|
| `tt_llk_{arch}/` | Per-arch LLK implementations — each fixer stays in its own arch's subtree |
| `codegen/artifacts/issue_{N}_analysis.md` | Shared analysis produced once |
| `codegen/artifacts/issue_{N}_fix_plan.md` | Shared fix plan produced once |
| `codegen/artifacts/issue_{N}_arch_research_{arch}.md` | Per-arch arch research (if produced) |
| `${LOGS_BASE_OF[$arch]}/${RUN_ID_OF[$arch]}/` | Per-arch LOG_DIR (dashboard-compatible layout) |

## Commands

Arch-parameterized versions of the compile + test commands. Functional tests use local hardware for BH/WH and `emu-quasar-1x3` for Quasar (carve-out).
```bash
# Compile check — parameterized by arch
cd codegen
[ -f ../tests/.venv/bin/activate ] && source ../tests/.venv/bin/activate
CHIP_ARCH=${arch} python scripts/compiler.py \
    {path_to_test_source} \
    -t "TEMPLATE_PARAM(...)" -r "RUNTIME_PARAM(...)" -v

# Functional tests — BH/WH on local card, Quasar on emu-quasar-1x3.
# BH/WH preconditions: /dev/tenstorrent/0 exists AND PCI_ID matches ${arch}
#                     (see detect_hw_arch() in tester.md — finalizes ENV_ERROR if not).
# Quasar preconditions: emu-quasar-1x3 build discoverable under
#                     /proj_sw/user_dev/*/tt-umd-simulators/build/ (see run_quasar_sim()).
[ -f ../tests/.venv/bin/activate ] && source ../tests/.venv/bin/activate || \
    export PYTHONPATH="${PYTHONPATH:-}:${HOME}/.local/lib/python3.10/site-packages"
cd ../${TESTS_DIR_OF[$arch]}
if [ "${arch}" = "quasar" ]; then
    flock --timeout 900 /tmp/tt-llk-test-simulator.lock \
        bash -c 'CHIP_ARCH=quasar pytest -x --run-simulator --compile-consumer --port='"${SIM_PORT_OF[quasar]}"' {test_file}'
else
    CHIP_ARCH=${arch} pytest -x {test_file}
fi
```
