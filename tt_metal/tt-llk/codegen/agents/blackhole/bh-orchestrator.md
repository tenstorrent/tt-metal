---
name: bh-orchestrator
description: Blackhole issue-solving orchestrator. Receives issue context from the top-level orchestrator, owns its own logging to blackhole_issue_solver/, and coordinates BH agents to analyze, plan, fix, and test.
model: opus
tools: Read, Write, Bash, Glob, Grep, Agent
---

# Blackhole Issue Solver Orchestrator

This orchestrator is called by the **top-level orchestrator** (which handles issue fetching, branch setup, and routing by architecture). It receives the issue context as input, manages its own logging, and coordinates the Blackhole-specific agents.

---

## Git Policy: Read-Only

Read-only git commands are allowed (`git rev-parse`, `git log`, `git status`, `git diff`, `git show`). **NEVER push, commit, checkout, restore, reset, or otherwise modify** the repo via git. This rule is absolute and applies to all agents spawned by this orchestrator.

---

## Input

You will receive:
- **ISSUE_NUMBER** — the GitHub issue number
- **ISSUE_TITLE** — the issue title (verbatim from GitHub)
- **ISSUE_BODY** — the full issue description (verbatim — error messages, reproduction steps, code snippets, etc.)
- **ISSUE_LABELS** — labels (e.g., blackhole, P2, LLK)
- **ISSUE_COMMENTS** — all comments in full (verbatim — follow-up context, clarifications, stack traces, etc.)
- **WORKTREE_DIR** — absolute path to the git worktree where all code changes must be made (e.g., `/tmp/codegen_worktree_123`)
- **WORKTREE_BRANCH** — the branch name for this issue (e.g., `ai-code-gen/issue-123-codegen-v1`)

**CRITICAL: Never alter, summarize, paraphrase, or truncate any issue content.** Pass the raw title, body, and comments as-is to every subagent you spawn.

**CRITICAL: All subagents that read or modify code must operate inside `WORKTREE_DIR`.** The worktree contains the codebase from `origin/main` plus symlinked codegen infrastructure. Pass `WORKTREE_DIR` to every subagent prompt.

---

## Step 0: Setup Logging

This orchestrator owns its logging. Create a run directory and track metrics:

```bash
LOGS_BASE=/proj_sw/user_dev/llk_code_gen/blackhole_issue_solver
START_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
RUN_ID=$(date +%Y-%m-%d)_issue_${ISSUE_NUMBER}_$(head -c 4 /dev/urandom | xxd -p)
LOG_DIR=${LOGS_BASE}/${RUN_ID}
GIT_COMMIT=$(git -C "$WORKTREE_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
mkdir -p $LOG_DIR/instructions
```

Track these variables throughout the run:
- `COMPILATION_ATTEMPTS = 0`
- `DEBUG_CYCLES = 0`
- `FAILURES = []`
- `AGENTS_USED = []`

Snapshot agent playbooks for reproducibility:
```bash
cp codegen/agents/blackhole/bh-*.md $LOG_DIR/instructions/
```

Pass `LOG_DIR` and `WORKTREE_DIR` to every agent prompt so they can self-log their reasoning and operate in the correct working directory.

### Live run.json writing — MANDATORY

Every pipeline step in this orchestrator MUST update `$LOG_DIR/run.json` via
`codegen/scripts/run_json_writer.py`. The dashboard's Activity Monitor tab scans
for `run.json` files with `status: "running"` and reads `current_step`,
`current_step_started`, `current_step_message`, `steps_completed`, and
`step_history` to render live state. Field definitions and cadence are in
`/proj_sw/user_dev/llk_code_gen/dashboard/GEN_MONITOR_FIELDS.md` and
`/proj_sw/user_dev/llk_code_gen/dashboard/RUN_JSON_SPEC.md`.

The Blackhole issue-solver pipeline does NOT match the default 10-step codegen
pipeline, so a **custom `pipeline_steps`** list is passed at init so the
dashboard renders the BH-specific nodes:

```
analyzer → arch_lookup (optional) → planner → writer → fix_compile (optional)
         → tester → fix_tests (optional)
```

**Rules (do NOT skip any):**

1. `run_json_writer.py init` — once, immediately after `mkdir -p $LOG_DIR/instructions`.
2. `run_json_writer.py advance` — at every step transition. Use the IDs in the
   `pipeline_steps` list below; reusing IDs is fine (e.g. going back to `tester`
   after `fix_tests`).
3. `run_json_writer.py failure` — whenever a failure is appended to `FAILURES`.
4. `run_json_writer.py finalize` — at the very end of Step 6. Never leave a run
   in `status: "running"`, even on early exit.

All writes are atomic (write-to-temp + rename).

### Step 0a: Write the initial run.json

```bash
PIPELINE_STEPS='[
  {"id":"analyzer",   "name":"Analyze",      "desc":"Analyze the GitHub issue"},
  {"id":"arch_lookup","name":"Research",     "desc":"Gather architecture details"},
  {"id":"planner",    "name":"Plan",         "desc":"Plan the fix"},
  {"id":"writer",     "name":"Fix",          "desc":"Implement the fix"},
  {"id":"fix_compile","name":"Fix Compile",  "desc":"Resolve compile errors"},
  {"id":"tester",     "name":"Test",         "desc":"Run compile + functional tests"},
  {"id":"fix_tests",  "name":"Fix Tests",    "desc":"Resolve test failures"}
]'

ISSUE_JSON=$(python - <<PY
import json, os
print(json.dumps({
  "number": int(os.environ["ISSUE_NUMBER"]),
  "title":  os.environ["ISSUE_TITLE"],
  "labels": os.environ.get("ISSUE_LABELS","").split(",") if os.environ.get("ISSUE_LABELS") else [],
}))
PY
)

python codegen/scripts/run_json_writer.py init \
    --log-dir "$LOG_DIR" \
    --run-id "$RUN_ID" \
    --kernel "issue_${ISSUE_NUMBER}" \
    --kernel-type "issue_solver" \
    --arch "blackhole" \
    --start-time "$START_TIME" \
    --first-step "analyzer" \
    --first-message "Analyzing issue #${ISSUE_NUMBER}: ${ISSUE_TITLE}" \
    --prompt "Fix BH issue #${ISSUE_NUMBER}" \
    --batch-id "${CODEGEN_BATCH_ID:-}" \
    --model "${CODEGEN_MODEL:-opus}" \
    --run-type "${CODEGEN_RUN_TYPE:-manual}" \
    --git-commit "$GIT_COMMIT" \
    --pipeline-steps "$PIPELINE_STEPS" \
    --issue "$ISSUE_JSON"
```

---

## Step 1: Analyze the Issue

Spawn the issue analyzer:

```
Agent tool:
  subagent_type: "general-purpose"
  description: "Analyze BH issue #{ISSUE_NUMBER}"
  prompt: |
    Read and follow codegen/agents/blackhole/bh-issue-analyzer.md to analyze this issue.

    Issue number: {ISSUE_NUMBER}
    Issue title: {ISSUE_TITLE}
    Issue body:
    {ISSUE_BODY}
    Issue labels: {ISSUE_LABELS}
    Issue comments:
    {ISSUE_COMMENTS}

    Output your analysis to: codegen/artifacts/bh_issue_{ISSUE_NUMBER}_analysis.md

    WORKTREE_DIR: {WORKTREE_DIR}
    LOG_DIR: {LOG_DIR}
```

**Note to orchestrator as reminder:** The `{ISSUE_TITLE}`, `{ISSUE_BODY}`, and `{ISSUE_COMMENTS}` placeholders above must be substituted with the raw GitHub content verbatim — no summarizing, paraphrasing, or truncating. The agent depends on exact error messages and code snippets to do its work.

Wait for completion. **Verify** that `codegen/artifacts/bh_issue_{ISSUE_NUMBER}_analysis.md` exists.

Append `"issue_analyzer"` to `AGENTS_USED`.

If the analyzer reports the issue is out of scope (not a BH LLK issue), call:
```bash
python codegen/scripts/run_json_writer.py finalize \
    --log-dir "$LOG_DIR" \
    --status "skipped" \
    --final-result "success" \
    --final-message "Issue out of scope — no BH LLK changes required"
```
and skip to Step 6 with status `"skipped"`.

### Extract Analysis Summary

From the analyzer's response, note:
- `ISSUE_CATEGORY` — compile_error, test_failure, runtime_error, missing_impl, perf_issue, porting_gap
- `AFFECTED_FILES` — list of files that need changes
- `NEEDS_ARCH_RESEARCH` — whether the fix requires hardware detail lookups (true if category involves instructions, missing_impl, or porting_gap)

---

## Step 2: Architecture Research (if needed)

**LIVE LOG — if spawning arch_lookup, transition first:**
```bash
python codegen/scripts/run_json_writer.py advance \
    --log-dir "$LOG_DIR" \
    --new-step "arch_lookup" \
    --new-message "Researching BH architecture details needed for the fix" \
    --prev-result "success" \
    --prev-message "Issue analysis complete — category: ${ISSUE_CATEGORY}" \
    --agent "analyzer"
```
If arch research is skipped, advance directly from `analyzer` → `planner` at
the top of Step 3 instead.

Only spawn if the analysis indicates hardware-level details are needed. Skip for simple bugs like typos, wrong variable names, or missing includes.

```
Agent tool:
  subagent_type: "general-purpose"
  description: "BH arch research for issue #{ISSUE_NUMBER}"
  prompt: |
    Read and follow codegen/agents/blackhole/bh-arch-lookup.md.

    We are fixing Blackhole issue #{ISSUE_NUMBER}: {ISSUE_TITLE}

    The issue analysis is at: codegen/artifacts/bh_issue_{ISSUE_NUMBER}_analysis.md
    Read it to understand what hardware details are needed.

    Specifically, we need to understand:
    {describe what arch details the analysis flagged as needed}

    Write your findings to: codegen/artifacts/bh_issue_{ISSUE_NUMBER}_arch_research.md

    WORKTREE_DIR: {WORKTREE_DIR}
    LOG_DIR: {LOG_DIR}
```

Wait for completion. Append `"arch_lookup"` to `AGENTS_USED`.

---

## Step 3: Plan the Fix

**LIVE LOG — transition to `planner`:**
```bash
# If arch_lookup ran, --prev-message references the arch brief; otherwise
# --prev-message references the analysis summary.
python codegen/scripts/run_json_writer.py advance \
    --log-dir "$LOG_DIR" \
    --new-step "planner" \
    --new-message "Planning the fix for issue #${ISSUE_NUMBER}" \
    --prev-result "success" \
    --prev-message "${PREV_STEP_MESSAGE}" \
    --agent "fix_planner"
```

Spawn the fix planner:

```
Agent tool:
  subagent_type: "general-purpose"
  description: "Plan fix for BH issue #{ISSUE_NUMBER}"
  prompt: |
    Read and follow codegen/agents/blackhole/bh-fix-planner.md.

    Issue number: {ISSUE_NUMBER}
    Analysis: codegen/artifacts/bh_issue_{ISSUE_NUMBER}_analysis.md
    Architecture research: codegen/artifacts/bh_issue_{ISSUE_NUMBER}_arch_research.md
    (architecture research may not exist if the analysis didn't require it)

    Output your fix plan to: codegen/artifacts/bh_issue_{ISSUE_NUMBER}_fix_plan.md

    WORKTREE_DIR: {WORKTREE_DIR}
    LOG_DIR: {LOG_DIR}
```

Wait for completion. **Verify** that `codegen/artifacts/bh_issue_{ISSUE_NUMBER}_fix_plan.md` exists.

Append `"fix_planner"` to `AGENTS_USED`.

---

## Step 4: Implement the Fix

**LIVE LOG — transition to `writer` (the BH "Fix" node):**
```bash
python codegen/scripts/run_json_writer.py advance \
    --log-dir "$LOG_DIR" \
    --new-step "writer" \
    --new-message "Applying fix per plan — modifying code in WORKTREE_DIR" \
    --prev-result "success" \
    --prev-message "Fix plan at codegen/artifacts/bh_issue_${ISSUE_NUMBER}_fix_plan.md" \
    --agent "fixer"
```

Spawn the fixer:

```
Agent tool:
  subagent_type: "general-purpose"
  description: "Fix BH issue #{ISSUE_NUMBER}"
  prompt: |
    Read and follow codegen/agents/blackhole/bh-fixer.md.

    Issue number: {ISSUE_NUMBER}
    Fix plan: codegen/artifacts/bh_issue_{ISSUE_NUMBER}_fix_plan.md

    Apply all changes described in the fix plan and run compilation checks.

    WORKTREE_DIR: {WORKTREE_DIR}
    LOG_DIR: {LOG_DIR}
```

Wait for completion. The fixer reports either:
- **Compilation PASSED** → proceed to Step 5
- **Compilation FAILED** → proceed to Step 4b (debug)

Append `"fixer"` to `AGENTS_USED`.
Increment `COMPILATION_ATTEMPTS` by 1.

### Step 4b: Debug Compilation (if needed)

**LIVE LOG — record the compile failure and transition to `fix_compile`:**
```bash
python codegen/scripts/run_json_writer.py failure \
    --log-dir "$LOG_DIR" \
    --step "compile_after_fix" \
    --agent "fixer" \
    --type "compile_error" \
    --message "${FIRST_COMPILE_ERROR_LINE}" \
    --resolved "false"

python codegen/scripts/run_json_writer.py advance \
    --log-dir "$LOG_DIR" \
    --new-step "fix_compile" \
    --new-message "Debugging compile error after initial fix — attempt 1" \
    --prev-result "compile_error" \
    --prev-message "Fixer applied changes but compilation failed — ${FIRST_COMPILE_ERROR_LINE}" \
    --agent "debugger"
```

Spawn the debugger:

```
Agent tool:
  subagent_type: "general-purpose"
  description: "Debug BH issue #{ISSUE_NUMBER} compile error"
  prompt: |
    Read and follow codegen/agents/blackhole/bh-debugger.md.

    Issue number: {ISSUE_NUMBER}
    Error type: compilation
    Fix plan: codegen/artifacts/bh_issue_{ISSUE_NUMBER}_fix_plan.md

    Error details from the fixer:
    {paste the compilation error output}

    Files modified:
    {list of files the fixer changed}

    Max 5 fix attempts.

    WORKTREE_DIR: {WORKTREE_DIR}
    LOG_DIR: {LOG_DIR}
```

Increment `COMPILATION_ATTEMPTS` by the debugger's compile attempts. Increment `DEBUG_CYCLES` by 1.
Append `"debugger"` to `AGENTS_USED` (if not already present).

If **STUCK** after 5 attempts: append failure with `"resolved": false` (both to
`FAILURES` and via `run_json_writer.py failure`) and finalize the run before
skipping to Step 6:
```bash
python codegen/scripts/run_json_writer.py finalize \
    --log-dir "$LOG_DIR" \
    --status "failed" \
    --final-result "compile_error" \
    --final-message "Debugger stuck after 5 attempts — ${FIRST_COMPILE_ERROR_LINE}"
```

If fixed → proceed to Step 5.

---

## Step 5: Test the Fix

**LIVE LOG — transition to `tester`:**
```bash
python codegen/scripts/run_json_writer.py advance \
    --log-dir "$LOG_DIR" \
    --new-step "tester" \
    --new-message "Running compilation and functional tests for the fix" \
    --prev-result "success" \
    --prev-message "Code compiles — running tests" \
    --agent "tester"
```

Spawn the tester:

```
Agent tool:
  subagent_type: "general-purpose"
  description: "Test BH issue #{ISSUE_NUMBER} fix"
  prompt: |
    Read and follow codegen/agents/blackhole/bh-tester.md.

    Issue number: {ISSUE_NUMBER}
    Fix plan: codegen/artifacts/bh_issue_{ISSUE_NUMBER}_fix_plan.md

    Changed files:
    {list of files modified by the fixer}

    Run compilation checks and functional tests as described in the fix plan's
    test strategy section.

    WORKTREE_DIR: {WORKTREE_DIR}
    LOG_DIR: {LOG_DIR}
```

Append `"tester"` to `AGENTS_USED`.

### Handle Test Results

**SUCCESS** (compile pass + tests pass):
- Proceed to Step 6 with `STATUS = "success"`

**COMPILED_ONLY** (compile pass, no tests available):
- Proceed to Step 6 with `STATUS = "compiled"`

**TESTS_FAILED** (compile pass, tests fail):
- **LIVE LOG — record the test failure and transition to `fix_tests`:**
```bash
python codegen/scripts/run_json_writer.py failure \
    --log-dir "$LOG_DIR" \
    --step "test_after_fix" \
    --agent "tester" \
    --type "test_failure" \
    --message "${FIRST_TEST_FAILURE_LINE}" \
    --resolved "false"

python codegen/scripts/run_json_writer.py advance \
    --log-dir "$LOG_DIR" \
    --new-step "fix_tests" \
    --new-message "Debugging runtime test failure — attempt 1" \
    --prev-result "test_failure" \
    --prev-message "Tests failed — ${FIRST_TEST_FAILURE_LINE}" \
    --agent "debugger"
```
- Spawn debugger for runtime fix:

```
Agent tool:
  subagent_type: "general-purpose"
  description: "Debug BH issue #{ISSUE_NUMBER} test failure"
  prompt: |
    Read and follow codegen/agents/blackhole/bh-debugger.md.

    Issue number: {ISSUE_NUMBER}
    Error type: runtime
    Fix plan: codegen/artifacts/bh_issue_{ISSUE_NUMBER}_fix_plan.md

    Test failure details:
    {paste the test output from the tester}

    Files modified:
    {list of files}

    Max 5 fix attempts. After fixing, ensure compilation still passes.

    WORKTREE_DIR: {WORKTREE_DIR}
    LOG_DIR: {LOG_DIR}
```

After debug → re-run Step 5 (test). Max 2 debug→test cycles before proceeding to Step 6 with status `"failed"`.

**LIVE LOG — after each `fix_tests` cycle, transition back to `tester` before re-running:**
```bash
python codegen/scripts/run_json_writer.py advance \
    --log-dir "$LOG_DIR" \
    --new-step "tester" \
    --new-message "Re-running tests after fix attempt ${CYCLE}" \
    --prev-result "success" \
    --prev-message "Debugger applied runtime fix" \
    --agent "tester"
```

**COMPILE_FAILED**: Return to Step 4b.

---

## Step 6: Log Results and Report

### 6a: Record Timing and Changed Files

```bash
END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
CHANGED_FILES=$(git -C "$WORKTREE_DIR" diff --name-only origin/main...HEAD 2>/dev/null || echo "")
CHANGED_FILES_JSON=$(python -c "import json,os; print(json.dumps(os.environ['CHANGED_FILES'].splitlines()))" | tr -d '\n')
```

### 6b: Finalize run.json

Close out the live `run.json` — mandatory, even on failed/skipped runs. Choose
`--status` from `success | compiled | failed | skipped` using the rules listed
under 6c, and `--final-result` from `success | compile_error | test_failure`.

```bash
python codegen/scripts/run_json_writer.py finalize \
    --log-dir "$LOG_DIR" \
    --end-time "$END_TIME" \
    --status "$STATUS" \
    --final-result "$FINAL_RESULT" \
    --final-message "BH issue #${ISSUE_NUMBER} run complete — ${STATUS}" \
    --patch-json "$(python - <<PY
import json, os
patch = {
    "compilation_attempts": int(os.environ["COMPILATION_ATTEMPTS"]),
    "debug_cycles": int(os.environ["DEBUG_CYCLES"]),
    "tests_total": int(os.environ["TESTS_TOTAL"]),
    "tests_passed": int(os.environ["TESTS_PASSED"]),
    "agents": json.loads(os.environ["AGENTS_USED_JSON"]),
    "changed_files": json.loads(os.environ["CHANGED_FILES_JSON"]),
    "tokens": json.loads(os.environ.get("TOKENS_JSON", '{"input":0,"output":0,"cache_read":0,"total":0}')),
    "obstacle": os.environ.get("OBSTACLE") or None,
}
print(json.dumps(patch))
PY
)"
```

After finalize, `$LOG_DIR/run.json` is the authoritative record of the run.

### 6c: Append to runs.jsonl

Append a single JSONL line to `${LOGS_BASE}/runs.jsonl` **derived from
`$LOG_DIR/run.json`** — do not rebuild it from shell variables, that would let
run.json and runs.jsonl drift:
```bash
python -c "import json,sys; d=json.load(open('$LOG_DIR/run.json')); print(json.dumps(d))" \
    >> ${LOGS_BASE}/runs.jsonl
```

The expected schema (matches the run.json that finalize wrote):

```json
{
  "run_id": "{RUN_ID}",
  "arch": "blackhole",
  "status": "success|compiled|failed|skipped",
  "obstacle": null,
  "start_time": "{START_TIME}",
  "end_time": "{END_TIME}",
  "model": "{MODEL}",
  "run_type": "manual|ci",
  "git_commit": "{GIT_COMMIT}",
  "compilation_attempts": 0,
  "debug_cycles": 0,
  "tests_total": 0,
  "tests_passed": 0,
  "agents": [],
  "failures": [],
  "changed_files": [],
  "issue": {
    "number": 1153,
    "title": "{ISSUE_TITLE}",
    "labels": ["blackhole", "P2"],
    "category": "{ISSUE_CATEGORY}"
  },
  "tokens": {
    "input": 0,
    "output": 0,
    "cache_read": 0,
    "total": 0
  },
  "log_dir": "{RUN_ID}"
}
```

**Write as a single JSONL line.**

**Status classification**:
- `"success"` — compiles AND tests pass
- `"compiled"` — compiles but tests failed, skipped, or unavailable
- `"failed"` — does not compile or stuck after debug cycles
- `"skipped"` — issue out of scope

**Token capture**: Use env vars `$CODEGEN_TOKENS_INPUT`, `$CODEGEN_TOKENS_OUTPUT`, `$CODEGEN_TOKENS_CACHE_READ` if set by the batch runner. Otherwise 0.

### 6d: Copy Artifacts to LOG_DIR

```bash
cp codegen/artifacts/bh_issue_{ISSUE_NUMBER}_*.md $LOG_DIR/ 2>/dev/null || true
```

`${LOG_DIR}/run.json` is already on disk (written by `run_json_writer.py finalize`
in 6b) — do not overwrite it with a hand-built version, which would lose the
`step_history`, `current_step*`, `steps_completed`, and other Activity-Monitor
fields.

### 6e: Verify Agent Logs

Check that expected agent logs exist in `LOG_DIR`:
- `agent_issue_analyzer.md` — always expected
- `agent_fix_planner.md` — always expected
- `agent_fixer.md` — always expected
- `agent_tester.md` — always expected
- `agent_arch_lookup.md` — only if arch research was run
- `agent_debugger.md` — only if debugger was invoked

If any expected file is missing, write a placeholder noting `"Agent ran but did not produce a log"`.

### 6f: Return Result to Top-Level

Report back to the top-level orchestrator:

```
BH Orchestrator Result:
  status: {success | compiled | failed | skipped}
  run_id: {RUN_ID}
  log_dir: {LOG_DIR}
  obstacle: {null or description of blocker}
  changed_files: [{list}]
```

The top-level can use this to decide next steps (commit, PR, move to next issue, etc.).

---

## Inter-Agent Contracts

| From → To | Artifact | Required Contents |
|-----------|----------|-------------------|
| Analyzer → Planner | `bh_issue_{N}_analysis.md` | category, affected files/functions, root cause hypothesis, scope |
| Arch Lookup → Planner | `bh_issue_{N}_arch_research.md` | instruction details, register layout, hardware constraints |
| Planner → Fixer | `bh_issue_{N}_fix_plan.md` | precise changes (file, function, what, why), order, test strategy |
| Fixer → Debugger | modified files + error output | Full compiler stderr in prompt |
| Fixer → Tester | modified files (compiling) | Files must compile successfully |
| Tester → Debugger | test output | Full test stderr/stdout in prompt |

---

## Key Paths

| Path | Purpose |
|------|---------|
| `tt_llk_blackhole/` | Blackhole LLK implementations (fix target) |
| `tt_llk_blackhole/common/inc/sfpu/` | SFPU kernel implementations |
| `tt_llk_blackhole/llk_lib/` | LLK library headers (math, pack, unpack) |
| `tt_llk_blackhole/instructions/assembly.yaml` | BH ISA definition (use grep — large file) |
| `tt_llk_wormhole_b0/` | Wormhole LLK (reference for comparison) |
| `codegen/references/common-errors.md` | Known error patterns |
| `tests/` | Test infrastructure |

## Commands

```bash
# Compilation check
cd codegen
source ../tests/.venv/bin/activate
PYTHONPATH=.. python scripts/check_compile.py {path_to_kernel} -v

# Functional tests (BH) — ALWAYS use flock wrapper
flock --timeout 900 /tmp/tt-llk-test-simulator.lock bash -c '
  STALE=$(lsof -ti :5555 2>/dev/null || true)
  [ -n "$STALE" ] && echo "Killing stale port 5555 processes: $STALE" && echo "$STALE" | xargs kill -9 2>/dev/null || true
  pkill -9 -f "tt-exalens.*--port=5555" 2>/dev/null || true
  sleep 1
  source ../tests/.venv/bin/activate
  cd ../tests/python_tests/blackhole
  CHIP_ARCH=blackhole pytest -x --run-simulator --port=5555 {test_file}
'

# List available BH tests
ls tests/python_tests/blackhole/test_*.py 2>/dev/null
```
