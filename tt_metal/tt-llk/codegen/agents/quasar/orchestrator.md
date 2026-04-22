# LLK CodeGen Orchestrator

## Git Policy: Read-Only

Read-only git commands are allowed (`git rev-parse`, `git log`, `git status`, `git diff`, `git show`) in the orchestrator and all subagents. **NEVER push, commit, checkout, restore, reset, or otherwise modify** the repo via git. This rule is absolute and applies to all agents spawned by this orchestrator.

---

## Pipeline Overview

When a user asks to **"generate {kernel} for {target_arch}"**, run this pipeline:

```
analyzer  →  [ writer → tester → refiner ] × up to 3 cycles  →  optimizer  →  format  →  report
```

- **analyzer** (`llk-analyzer.md`) — one-shot. Produces `codegen/artifacts/{op}_analysis.md` covering arch research, target-pattern survey, instruction mapping, solution approach, and format applicability.
- **writer** (`llk-kernel-writer.md`) — transcribes the analysis into a kernel file and compile-checks.
- **tester** (`llk-tester.md`) — writes/extends tests and runs them, with an **internal 10-attempt** compile+test fix loop. Returns `PASS` or `STUCK`.
- **refiner** (`llk-analysis-refiner.md`) — invoked only when the writer fails compile or the tester returns `STUCK`. Archives the failed attempt, rewrites the analysis in place, returns `REFINED` or `ESCALATE`.
- **Loop cap: 3 writer-tester cycles**. Cycles 1 and 2 can hand off to the refiner; cycle 3 cannot (the refiner itself caps at v2 = 2 refinements = 3 total cycles). If cycle 3 still fails, the run is reported `failed`.
- **optimizer** and **format** only run on success.

Agent playbooks live in `codegen/agents/quasar/`. The system discovers architectural patterns from authoritative sources — not hardcoded knowledge.

---

## Input

The top-level routing (`codegen/CLAUDE.md`) creates an isolated worktree and passes you:

- **KERNEL_NAME** — the kernel to generate (e.g., `gelu`)
- **TARGET_ARCH** — target architecture (default: `quasar`)
- **WORKTREE_DIR** — absolute path to the isolated git worktree (e.g., `/tmp/codegen_worktree_generate-gelu-quasar`)
- **WORKTREE_BRANCH** — the branch name (e.g., `ai-code-gen/generate-gelu-quasar-v1`)

**CRITICAL: All code writes and file modifications MUST happen inside `$WORKTREE_DIR/tt_metal/tt-llk`.** The worktree has `codegen/` populated with symlinks to the source branch (read-only: `agents/`, `scripts/`, `references/`, `config/`, `CLAUDE.md`, `skills/`) plus a real per-worktree `codegen/artifacts/` directory for this run's outputs. Anything you or a subagent writes outside the worktree leaks into the source branch.

Before any other work, enter the worktree:

```bash
cd "$WORKTREE_DIR/tt_metal/tt-llk"
```

Every `cd`, file path, and subagent prompt below assumes this as the starting CWD. Each agent you spawn MUST also operate inside the worktree — pass `WORKTREE_DIR` in every agent prompt and tell them to `cd` there before doing anything.

---

## Step -1: Validate Environment

Before starting, verify prerequisites:

```bash
cd codegen
PYTHONPATH=.. python -c "from codegen.config.settings import settings; issues = settings.validate(); [print(f'ISSUE: {i}') for i in issues]; exit(1) if issues else print('Environment OK')"
```

If any issues are reported, **stop and tell the user** what needs to be fixed before codegen can work.

---

## Step 0: Setup Metrics Logging

Record the start time and create a unique log directory for this run. Every
variable that's later referenced by a Python heredoc (`os.environ[...]`) is
`export`'d so the subprocess can read it.

```bash
export START_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
export RUN_ID=$(date +%Y-%m-%d)_{kernel}_{arch}_$(head -c 4 /dev/urandom | xxd -p)
export LOG_DIR=/proj_sw/user_dev/llk_code_gen/quasar/$RUN_ID
export GIT_COMMIT=$(git -C "$WORKTREE_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
mkdir -p $LOG_DIR/instructions
```

### Live run.json writing — MANDATORY

Every step transition in this orchestrator MUST update `$LOG_DIR/run.json` via
`codegen/scripts/run_json_writer.py`. The Activity Monitor tab on the dashboard
reads this file while the run is in progress (status: "running") and relies on
`current_step`, `current_step_started`, `current_step_message`, `steps_completed`,
`step_history`, and `per_phase` staying current. The fields and writing cadence
are specified by
`/proj_sw/user_dev/llk_code_gen/dashboard/GEN_MONITOR_FIELDS.md` and
`/proj_sw/user_dev/llk_code_gen/dashboard/RUN_JSON_SPEC.md`.

**Rules (do NOT skip any):**

1. Call `run_json_writer.py init` once, immediately after creating `$LOG_DIR`
   (Step 0a below). Pass the `pipeline_steps` JSON so the dashboard knows the
   canonical step set: `analyzer → writer → tester → refiner → optimizer → format`.
2. Call `run_json_writer.py advance` at every pipeline step boundary.
   **Reusing the same step ID across cycles is required and expected** — the
   dashboard renders retries correctly when the same step appears multiple times
   in `step_history`. The writer and tester will appear once per cycle
   (up to 3 times each); the refiner appears at most twice.
3. Call `run_json_writer.py phase-start / phase-test / phase-end` at the start,
   during tester execution, and at the end of every writer-tester cycle. Each
   cycle is represented as one phase in `per_phase[]` (phase 1 = cycle 1, …).
4. Call `run_json_writer.py failure` whenever a failure is recorded (same
   content as the `FAILURES` bash list — write both).
5. Call `run_json_writer.py finalize` at the very end of Step 5 to flip `status`
   to its terminal value (`success`, `compiled`, or `failed`) and close the last
   `step_history` entry. **Never leave a run in status="running"**; even on early
   exit, run `finalize --status failed --final-result ...` before returning.

All writes are atomic (write-to-temp + rename) and safe to interleave with the
dashboard's reads.

### Authoring rule for `--new-message` / `--prev-message` / `--final-message`

Every message passed to `run_json_writer.py advance / phase-end / finalize` MUST
be a self-contained sentence a stranger reading only `step_history` can follow.
These messages are the public record of what each step did — they land in the
dashboard and in `$LOG_DIR/run.json` and are the only summary most viewers will
ever see. Short placeholder strings force the viewer to open the agent log to
understand anything. That is a defect, not a style preference.

Rules:

1. **Name the concrete artifact produced or consumed.** Not "Analysis complete"
   but "Analysis produced `fill_analysis.md` with 3 enumerated helpers and an
   8-row Semantic→Instruction table".
2. **Quantify.** Helper count, test count, cycle number, compile attempts,
   refinement version — whatever applies. "156/156 variants passed" beats
   "tests passed".
3. **When a failure is being recorded, name the first meaningful error line**
   (file:line symbol), not just the category. Dashboard readers scan these for
   regressions.
4. **Mid-step progress**: long-running steps (analyzer Confluence sweeps,
   tester simulator runs) SHOULD call `run_json_writer.py message` at natural
   breakpoints so the dashboard's spinner has something to render instead of a
   stale step-start message. Each `message` call costs one atomic rename —
   call it freely; do not batch.

Example call:
```bash
python codegen/scripts/run_json_writer.py message \
    --log-dir "$LOG_DIR" \
    --message "Tester attempt 4/10 — DATA_MISMATCH on Int32 dest_acc=Yes; fixing LREG1 init ordering"
```

### Step 0a: Write the initial run.json

Right after `mkdir -p $LOG_DIR/instructions`, write the initial run.json so the
dashboard immediately picks up this run as "running":

```bash
export PIPELINE_STEPS_JSON='[
  {"id":"analyzer","name":"Analyze","desc":"Research arch + analyze reference, produce solution approach"},
  {"id":"writer","name":"Write","desc":"Scaffold + fill kernel, compile-check"},
  {"id":"tester","name":"Test","desc":"Write/extend tests, run, internal 10-attempt fix loop"},
  {"id":"refiner","name":"Refine","desc":"Rewrite analysis after writer/tester failure (max 2 refinements)"},
  {"id":"optimizer","name":"Optimize","desc":"Replay-buffer optimization (success only)"},
  {"id":"format","name":"Format","desc":"Run pre-commit formatters on generated files"}
]'

python codegen/scripts/run_json_writer.py init \
    --log-dir "$LOG_DIR" \
    --run-id "$RUN_ID" \
    --kernel "{op}" \
    --kernel-type "{kernel_type}" \
    --arch "{target_arch}" \
    --reference-arch "{ref_arch}" \
    --reference-file "tt_llk_{ref_arch}/{kernel_path}" \
    --generated-file "tt_llk_{target_arch}/{kernel_path}" \
    --start-time "$START_TIME" \
    --first-step "analyzer" \
    --first-message "Analyzing {ref_arch} reference and producing solution approach for {op}" \
    --prompt "$PROMPT" \
    --batch-id "${CODEGEN_BATCH_ID:-}" \
    --model "$MODEL" \
    --run-type "$RUN_TYPE" \
    --git-commit "$GIT_COMMIT" \
    --phases-total 1 \
    --pipeline-steps "$PIPELINE_STEPS_JSON"

# Persist key vars for refresh_cost.sh across Bash tool-call shells.
# Each Bash tool call runs in a fresh shell — exported vars from prior calls
# are gone. refresh_cost.sh sources this file as a fallback so cost tracking
# works regardless of which shell block calls it.
cat > /tmp/codegen_run_state.sh << _STATE_EOF
export START_TIME="${START_TIME}"
export LOG_DIR="${LOG_DIR}"
export MODEL="${MODEL}"
_STATE_EOF

# Discover the session ID NOW — at startup it is still the most recently
# started session, so the fallback in session_cost.py is reliable.  Saving it
# here means refresh_cost.sh and extract_run_transcripts.py can pass it
# explicitly on all later calls (hours later, other sessions may have started
# and the fallback picks the wrong one).
_SESSION_PAIR=$(python codegen/scripts/session_cost.py --print-session 2>/dev/null || echo "")
SESSION_ID=$(echo "$_SESSION_PAIR" | awk '{print $1}')
PROJECT_CWD=$(echo "$_SESSION_PAIR" | cut -d' ' -f2-)
if [ -n "$SESSION_ID" ]; then
    echo "export SESSION_ID=\"${SESSION_ID}\"" >> /tmp/codegen_run_state.sh
    echo "export PROJECT_CWD=\"${PROJECT_CWD}\"" >> /tmp/codegen_run_state.sh
fi
```

Note: `--phases-total` starts at `1` (we always run at least one writer-tester
cycle). It is bumped to 2 / 3 via the `metric` subcommand each time the refiner
enables another cycle.

Track these variables throughout the run. **All of them must be
`export`'d** because the `finalize` step reads them from a Python subprocess via
`os.environ[...]`:

```bash
export PROMPT="Generate {kernel} for {arch}"       # the original user prompt verbatim
export BATCH_ID="${CODEGEN_BATCH_ID:-}"             # empty string if not a batch run
export MODEL="${CODEGEN_MODEL:-sonnet}"               # opus | sonnet | haiku
export RUN_TYPE="$([ -n "$BATCH_ID" ] && echo ci || echo manual)"

export CYCLE=1                                      # current writer-tester cycle (1..3)
export MAX_CYCLES=3                                 # hard cap
export REFINEMENT_COUNT=0                           # how many times the refiner ran

export COMPILATION_ATTEMPTS=0                       # every compile across writer + tester's internal loop
export DEBUG_CYCLES=0                               # refinement iterations (== REFINEMENT_COUNT at end)
export PHASES_TOTAL=1                               # cycles attempted; bumped to 2/3 as refinements happen
export PHASES_COMPLETED=0                           # 1 if a cycle passed, else 0
export TESTS_TOTAL=0                                # total test variants run in the successful cycle
export TESTS_PASSED=0
export LINES_GENERATED=0
export TESTS_GENERATED=false                        # true if the tester created new test files
export OPTIMIZED=false                              # true if optimizer applied a change
export OPTIMIZATION_TYPE=none                       # replay | none
export FORMATS_TESTED_JSON='[]'
export FORMATS_EXCLUDED_JSON='{}'
export AGENTS_JSON='[]'
export TOKENS_JSON='{"input":0,"output":0,"cache_read":0,"cache_creation":0,"total":0,"cost_usd":0}'
export OBSTACLE=                                    # set if the run is blocked
```

Two list-valued items stay in shell (too awkward as exported strings):
- `PER_PHASE=[]` — build up per-cycle results as each cycle ends
- `FAILURES=[]` — append every failure encountered during the run (see below)

**Failure tracking**: Whenever an agent fails, a compilation fails, a test fails, or an infrastructure error occurs, append an entry to `FAILURES`:
```json
{
  "step": "writer_cycle_1 | tester_cycle_2 | refiner_v1 | analyzer | optimizer | format",
  "agent": "writer | tester | refiner | analyzer | optimizer",
  "type": "compile_error | test_failure | agent_error | infra_error",
  "message": "First meaningful line of the error (stderr, traceback, or test output)",
  "resolved": true
}
```
- `step`: Which pipeline step failed, suffixed with the cycle or refinement number when relevant.
- `agent`: Which agent was running when the failure occurred.
- `type`: Category — `"compile_error"` (compiler stderr), `"test_failure"` (pytest/simulator), `"agent_error"` (agent stuck/crashed), `"infra_error"` (simulator timeout, env issue).
- `message`: The actual error — first meaningful line of compiler stderr, pytest failure, or agent error. Keep it concise but specific enough to diagnose.
- `resolved`: `true` if the issue was fixed during the run (e.g., cycle 1 failed but cycle 2 passed), `false` if it blocked completion.

Copy the agent playbooks used (snapshot for reproducibility):
```bash
cp codegen/agents/quasar/llk-analyzer.md         $LOG_DIR/instructions/
cp codegen/agents/quasar/llk-kernel-writer.md    $LOG_DIR/instructions/
cp codegen/agents/quasar/llk-tester.md           $LOG_DIR/instructions/
cp codegen/agents/quasar/llk-analysis-refiner.md $LOG_DIR/instructions/
cp codegen/agents/quasar/llk-optimizer.md        $LOG_DIR/instructions/
```

Pass `LOG_DIR` to every agent prompt so they can self-log their reasoning.

### Step 0a.1: Live cost tracking — MANDATORY

Claude Code writes every assistant turn to `~/.claude/projects/<cwd-mapped>/<sessionId>.jsonl`
and every sub-agent (Agent tool) turn under `.../<sessionId>/subagents/*.jsonl`. Each
`type: assistant` entry carries `message.usage` (`input_tokens`, `output_tokens`,
`cache_read_input_tokens`, `cache_creation_input_tokens`) and `message.model`.
`codegen/scripts/session_cost.py` aggregates these across the main jsonl + all
subagent jsonls, filters to entries with `timestamp >= $START_TIME`, applies
per-model Anthropic pricing, and atomically patches `$LOG_DIR/run.json` — both
the `tokens` object and the top-level `cost_usd` field — so the dashboard
reflects cumulative spend in real time.

For batch runs (`claude -p ... --output-format json > cli_output.json`) the
batch runner later drops `cli_output.json` into `$LOG_DIR` and the dashboard
backfills the authoritative cumulative total from there — that supersedes
anything `session_cost.py` wrote.

**Accuracy note.** `cost_usd` in `run.json` is an estimate, same quality as
the `/cost` slash command — both multiply token counts by a local pricing
table. Anthropic itself says `/cost` "may differ from your actual bill; for
authoritative billing see the Usage page in the Claude Console." Expect
`session_cost.py` and `/cost` to agree within a few percent as long as
`PRICING` in `session_cost.py` tracks current list prices. For billing-grade
numbers use the Claude Console, or — for batch runs — `cli_output.json`'s
`total_cost_usd`, which is the CLI-summed per-request cost and the closest
non-Console equivalent. If Anthropic publishes a price change, update the
`PRICING` table at the top of `codegen/scripts/session_cost.py`.

Call `bash codegen/scripts/refresh_cost.sh` at every natural boundary below.
The script is a thin wrapper around `session_cost.py` that reads `START_TIME`,
`LOG_DIR`, and (optionally) `MODEL` from the environment. **Use the script, not
a bash function**: `Bash` tool calls do not share shell state, so an
in-shell `refresh_cost()` defined in one call is undefined in every subsequent
call and silently no-ops — this caused historical runs to land in `run.json`
with `cost_usd: null`. The script form survives across tool calls because it
lives on disk.

Call it after:
- the analyzer returns (Step 1),
- every `phase-end` in the writer-tester loop (Step 2),
- the optimizer returns (Step 3),
- the format step (Step 4),
- the final `run_json_writer.py finalize` call (Step 5b).

Each call takes <1s and reads the jsonl append-only, so it's safe to interleave
with Claude Code writing new turns.

---

## Step 0b: Identify Kernel Type and Architecture

Determine the kernel category and architecture from the request:

| Category | Keywords | Path Pattern |
|----------|----------|--------------|
| **SFPU** | sigmoid, relu, exp, gelu, tanh, sqrt, recip | `common/inc/sfpu/ckernel_sfpu_{op}.h` |
| **Math** | matmul, reduce, eltwise, binary, unary | `llk_lib/llk_math_{op}.h` |
| **Pack** | pack, untilize | `llk_lib/llk_pack_{op}.h` |
| **Unpack** | unpack, tilize | `llk_lib/llk_unpack_{op}.h` |

Determine:
- **Reference architecture** (default: blackhole) — the existing implementation to port from
- **Target architecture** (default: quasar) — where the kernel needs to run

---

## Step 1: Analyzer

Spawn the analyzer. This agent produces the single authoritative analysis document used by every downstream agent; it covers arch research, target-pattern survey, instruction discovery, semantic→instruction mapping, solution approach, format applicability, and complexity/phases.

```
Agent tool:
  subagent_type: "general-purpose"
  description: "Analyze {op} for {target_arch}"
  prompt: |
    Read and follow codegen/agents/quasar/llk-analyzer.md to analyze the "{op}" kernel.
    Flow: generation (new kernel).
    Kernel: {op}
    Kernel type: {kernel_type}
    Reference architecture: {ref_arch}
    Target architecture: {target_arch}
    Reference path: tt_llk_{ref_arch}/{kernel_path}
    Output your analysis to: codegen/artifacts/{op}_analysis.md

    The analyzer is the ONLY arch-research step in this pipeline — there is no
    separate arch-lookup agent. You own discovering target instructions,
    register layouts, and format constraints. Follow the
    codegen/skills/llk-arch-lookup/SKILL.md page index for Confluence fetches.

    WORKTREE_DIR: {WORKTREE_DIR} — cd here before any file I/O. All paths in this
    prompt resolve inside the worktree, not the source branch. Never write outside it.
    LOG_DIR: {LOG_DIR}
```

Wait for completion. **Verify** that `codegen/artifacts/{op}_analysis.md` exists and contains: Problem Statement, Target Pattern Survey, Available Instructions, Semantic→Instruction Mapping, Solution Approach (with §6a signatures and §6b pseudocode), Format Applicability, Complexity & Phases. If missing, record a failure and finalize as `failed`.

On analyzer failure:
```bash
python codegen/scripts/run_json_writer.py failure \
    --log-dir "$LOG_DIR" \
    --step "analyzer" \
    --agent "analyzer" \
    --type "agent_error" \
    --message "${ANALYZER_ERROR_LINE}" \
    --resolved "false"

python codegen/scripts/run_json_writer.py finalize \
    --log-dir "$LOG_DIR" \
    --status "failed" \
    --final-result "compile_error" \
    --final-message "Analyzer failed to produce {op}_analysis.md"
```

**LIVE LOG — analyzer passed, advance to writer (cycle 1), start phase 1:**
```bash
python codegen/scripts/run_json_writer.py advance \
    --log-dir "$LOG_DIR" \
    --new-step "writer" \
    --new-message "Cycle 1 — writing kernel from analysis" \
    --prev-result "success" \
    --prev-message "Analysis complete — codegen/artifacts/{op}_analysis.md" \
    --agent "writer"

python codegen/scripts/run_json_writer.py phase-start \
    --log-dir "$LOG_DIR" \
    --phase 1 \
    --name "cycle 1 (fresh analysis)"

bash codegen/scripts/refresh_cost.sh   # capture analyzer spend in run.json
```

---

## Step 2: Writer → Tester → Refiner Loop (max 3 cycles)

Run this block for `CYCLE ∈ {1, 2, 3}`. Break out the moment the tester returns `PASS`. If the tester returns `STUCK` in cycle 3, finalize the run as `failed` — cycle 3 does NOT hand off to the refiner.

```
while CYCLE <= MAX_CYCLES:
    # Step 2a: Writer
    result = run_writer()
    if result == PASSED:
        # Step 2b: Tester
        result = run_tester()
        if result == PASS:
            phase_end(CYCLE, "passed")
            break                                   # -> Step 3 (optimizer)
        # tester STUCK
        phase_end(CYCLE, "failed")
        if CYCLE == MAX_CYCLES:
            finalize("failed", "test_failure")
            return
        # Step 2c: Refiner (only cycles 1 and 2)
        run_refiner()                               # REFINED or ESCALATE
        if escalated:
            finalize("failed", "test_failure")
            return
    else:
        # writer compile FAILED
        phase_end(CYCLE, "failed")
        if CYCLE == MAX_CYCLES:
            finalize("failed", "compile_error")
            return
        run_refiner()
        if escalated:
            finalize("failed", "compile_error")
            return

    CYCLE += 1
    phase_start(CYCLE, "cycle CYCLE (after refinement v{CYCLE-1})")
```

The three sub-steps follow.

### Step 2a: Writer

Per-cycle counters:
```bash
export PHASE_COMPILES=0
export PHASE_TEST_DETAILS=""
export PHASE_COMPILE_ERRORS_JSON='[]'
```

Spawn the writer:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Write {op} (cycle ${CYCLE})"
  prompt: |
    Read and follow codegen/agents/quasar/llk-kernel-writer.md to generate the "{op}" kernel.
    Kernel type: {kernel_type}
    Target architecture: {target_arch}
    Analysis: codegen/artifacts/{op}_analysis.md
    Output to: tt_llk_{target_arch}/{kernel_path}

    You are running inside cycle ${CYCLE} of a max-3 writer-tester loop. If your
    compile check fails, report FAILED and return — do NOT iterate internally.
    The orchestrator will route the failure to the refiner, which will update
    the analysis in place before the next cycle retries.

    If a prior refinement occurred (cycle > 1), the analysis at
    codegen/artifacts/{op}_analysis.md has been rewritten in place and a
    "Refinement History" section at the top lists what changed. Follow the
    refined analysis; do not reintroduce approaches from the prior failed
    attempt (archived under codegen/artifacts/{op}_failed_attempt_v*/).

    WORKTREE_DIR: {WORKTREE_DIR} — cd here before any file I/O. All paths in this
    prompt resolve inside the worktree, not the source branch. Never write outside it.
    LOG_DIR: {LOG_DIR}
```

Wait for completion. The writer returns `PASSED` / `FAILED` on its final compile check.

**Metrics**: Increment `COMPILATION_ATTEMPTS` and `PHASE_COMPILES` by the number of compiles the writer ran (typically 2: scaffold + final). Append any compile-error messages to `PHASE_COMPILE_ERRORS_JSON`.

**If writer reports PASSED**: go to Step 2b.

**If writer reports FAILED** (compile broken):
```bash
python codegen/scripts/run_json_writer.py failure \
    --log-dir "$LOG_DIR" \
    --step "writer_cycle_${CYCLE}" \
    --agent "writer" \
    --type "compile_error" \
    --message "${FIRST_COMPILE_ERROR_LINE}" \
    --resolved "false"

python codegen/scripts/run_json_writer.py phase-end \
    --log-dir "$LOG_DIR" \
    --phase "${CYCLE}" \
    --test-result "failed" \
    --compilation-attempts "${PHASE_COMPILES}" \
    --debug-cycles 0 \
    --test-details "writer compile failed: ${FIRST_COMPILE_ERROR_LINE}" \
    --compile-errors-json "${PHASE_COMPILE_ERRORS_JSON}"

bash codegen/scripts/refresh_cost.sh
```

If `CYCLE == MAX_CYCLES` (3): jump to Step 5 with `status=failed`, `final-result=compile_error`.
Otherwise: go to Step 2c (refiner).

### Step 2b: Tester

**LIVE LOG — advance to tester:**
```bash
python codegen/scripts/run_json_writer.py advance \
    --log-dir "$LOG_DIR" \
    --new-step "tester" \
    --new-message "Cycle ${CYCLE} — writing/running tests (internal 10-attempt loop)" \
    --prev-result "success" \
    --prev-message "Cycle ${CYCLE} writer compiled" \
    --agent "tester"

python codegen/scripts/run_json_writer.py phase-test \
    --log-dir "$LOG_DIR" \
    --phase "${CYCLE}" \
    --state "running"
```

Spawn the tester:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Test {op} (cycle ${CYCLE})"
  prompt: |
    Read and follow codegen/agents/quasar/llk-tester.md to validate the "{op}" kernel.
    Kernel: {op}
    Kernel type: {kernel_type}
    Target architecture: {target_arch}
    Kernel path: tt_llk_{target_arch}/{kernel_path}
    Flow: new-kernel
    Spec: codegen/artifacts/{op}_analysis.md
    Cycle: ${CYCLE}

    You own the full test-and-fix loop with a hard cap of 10 test runs. Each
    run = compile-producer + simulator-consumer. Diagnose and fix between runs.
    On attempt 10's failure, return STUCK — the orchestrator will route to the
    refiner.

    WORKTREE_DIR: {WORKTREE_DIR} — cd here before any file I/O.
    LOG_DIR: {LOG_DIR}
```

Wait for the tester to return `PASS` or `STUCK`.

**Metrics from the tester's report** (the agent reports attempts used, variants, formats tested, test files created):
- `COMPILATION_ATTEMPTS += tester_compile_count`
- `PHASE_COMPILES += tester_compile_count`
- `PHASE_DEBUGS = tester_attempts_used` (the tester's 10-attempt loop is effectively the per-cycle debug count)
- `TESTS_TOTAL = variants_reported`
- `TESTS_PASSED = variants_passed_on_final_run`
- `TESTS_GENERATED = true` (set once, stays true)
- `FORMATS_TESTED_JSON` — set from the tester's report
- `FORMATS_EXCLUDED_JSON` — set from the tester's report

**If tester reports PASS**:
```bash
python codegen/scripts/run_json_writer.py phase-end \
    --log-dir "$LOG_DIR" \
    --phase "${CYCLE}" \
    --test-result "passed" \
    --compilation-attempts "${PHASE_COMPILES}" \
    --debug-cycles "${PHASE_DEBUGS}" \
    --test-details "${TESTS_PASSED}/${TESTS_TOTAL} variants passed" \
    --compile-errors-json "${PHASE_COMPILE_ERRORS_JSON}"

bash codegen/scripts/refresh_cost.sh
```
`PHASES_COMPLETED=1`. Break out of the cycle loop → Step 3 (optimizer).

**If tester reports STUCK**:
```bash
python codegen/scripts/run_json_writer.py failure \
    --log-dir "$LOG_DIR" \
    --step "tester_cycle_${CYCLE}" \
    --agent "tester" \
    --type "test_failure" \
    --message "${TESTER_LAST_FAILURE_LINE}" \
    --resolved "false"

python codegen/scripts/run_json_writer.py phase-end \
    --log-dir "$LOG_DIR" \
    --phase "${CYCLE}" \
    --test-result "failed" \
    --compilation-attempts "${PHASE_COMPILES}" \
    --debug-cycles "${PHASE_DEBUGS}" \
    --test-details "tester STUCK after 10 attempts: ${TESTER_LAST_FAILURE_LINE}" \
    --compile-errors-json "${PHASE_COMPILE_ERRORS_JSON}"

bash codegen/scripts/refresh_cost.sh
```

If `CYCLE == MAX_CYCLES` (3): jump to Step 5 with `status=failed`, `final-result=test_failure`.
Otherwise: go to Step 2c (refiner).

### Step 2c: Refiner (only for CYCLE ∈ {1, 2})

**LIVE LOG — advance to refiner:**
```bash
python codegen/scripts/run_json_writer.py advance \
    --log-dir "$LOG_DIR" \
    --new-step "refiner" \
    --new-message "Cycle ${CYCLE} failed — refining analysis (v${CYCLE})" \
    --prev-result "${PREV_RESULT}" \
    --prev-message "Cycle ${CYCLE} failed: ${FAILURE_SUMMARY}" \
    --agent "refiner"
```
(`PREV_RESULT` is `compile_error` if the writer failed, `test_failure` if the tester failed.)

Collect inputs for the refiner. The test files the tester used must be passed verbatim so the refiner can archive them:
```bash
TEST_FILES=""
for candidate in \
    "tests/sources/{target_arch}/sfpu_{op}_{target_arch}_test.cpp" \
    "tests/sources/{target_arch}/{op}_{target_arch}_test.cpp" \
    "tests/python_tests/{target_arch}/test_{op}_{target_arch}.py" \
    "tests/python_tests/{target_arch}/test_sfpu_{op}_{target_arch}.py"; do
    [ -f "$candidate" ] && TEST_FILES="$TEST_FILES $candidate"
done
```

Spawn the refiner:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Refine {op} analysis (v${CYCLE})"
  prompt: |
    Read and follow codegen/agents/quasar/llk-analysis-refiner.md to refine the "{op}" analysis.
    Kernel: {op}
    Kernel type: {kernel_type}
    Target architecture: {target_arch}
    Kernel path: tt_llk_{target_arch}/{kernel_path}
    Original analysis: codegen/artifacts/{op}_analysis.md
    Tester log: {LOG_DIR}/agent_tester_cycle${CYCLE}.md
    Writer log: {LOG_DIR}/agent_writer_cycle${CYCLE}.md
    Test files: ${TEST_FILES}
    Previous failure: ${FAILURE_SUMMARY}

    You are refinement iteration v${CYCLE}. Your own doc caps you at v2.
    Archive the failed attempt, identify the structural error in the analysis,
    rewrite the impeached sections in place, and return REFINED or ESCALATE.

    WORKTREE_DIR: {WORKTREE_DIR} — cd here before any file I/O.
    LOG_DIR: {LOG_DIR}
```

Wait for the refiner to return `REFINED` or `ESCALATE`.

**Metrics**:
- `REFINEMENT_COUNT += 1`
- `DEBUG_CYCLES = REFINEMENT_COUNT`
- `PHASES_TOTAL = CYCLE + 1` (we now know another cycle will run)

Patch `phases_total` so the dashboard shows the correct total:
```bash
python codegen/scripts/run_json_writer.py metric \
    --log-dir "$LOG_DIR" \
    --patch-json "{\"phases_total\": ${PHASES_TOTAL}, \"debug_cycles\": ${DEBUG_CYCLES}}"
```

**If refiner reports ESCALATE**:
```bash
python codegen/scripts/run_json_writer.py failure \
    --log-dir "$LOG_DIR" \
    --step "refiner_v${CYCLE}" \
    --agent "refiner" \
    --type "agent_error" \
    --message "refiner escalated: ${REFINER_REASON}" \
    --resolved "false"
```
Jump to Step 5 with `status=failed`, `final-result=${PREV_RESULT}`.

**If refiner reports REFINED**:
```bash
CYCLE=$((CYCLE + 1))

python codegen/scripts/run_json_writer.py advance \
    --log-dir "$LOG_DIR" \
    --new-step "writer" \
    --new-message "Cycle ${CYCLE} — writing kernel from refined analysis" \
    --prev-result "success" \
    --prev-message "Refinement v$((CYCLE - 1)) complete — analysis rewritten in place" \
    --agent "writer"

python codegen/scripts/run_json_writer.py phase-start \
    --log-dir "$LOG_DIR" \
    --phase "${CYCLE}" \
    --name "cycle ${CYCLE} (after refinement v$((CYCLE - 1)))"
```

Loop back to Step 2a.

---

## Step 3: Optimizer (success path only)

Only run after a cycle returned `PASS`. Skip if the tester never passed.

### Step 3a: Check whether optimization is applicable

Only SFPU kernels whose Blackhole reference uses replay buffers are candidates:

```bash
REPLAY_USES=$(grep -c "replay\|load_replay_buf" "tt_llk_{ref_arch}/{kernel_path}" || echo 0)
```

If `REPLAY_USES == 0` or the kernel is not SFPU: set `OPTIMIZED=false`, `OPTIMIZATION_TYPE=none`, and skip to Step 4 (format). Do **not** advance to the optimizer step — the run stays on `tester` until format/finalize.

### Step 3b: Run the optimizer

Snapshot the pre-optimization kernel (for comparison / rollback reporting):
```bash
cp "tt_llk_{target_arch}/{kernel_path}" "$LOG_DIR/pre_opt_$(basename tt_llk_{target_arch}/{kernel_path})"
```

**LIVE LOG — advance to optimizer:**
```bash
python codegen/scripts/run_json_writer.py advance \
    --log-dir "$LOG_DIR" \
    --new-step "optimizer" \
    --new-message "Applying replay-buffer optimization to {op}" \
    --prev-result "success" \
    --prev-message "Cycle ${CYCLE} passed — entering optimization" \
    --agent "optimizer"
```

Spawn the optimizer:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Optimize {op}"
  prompt: |
    Read and follow codegen/agents/quasar/llk-optimizer.md to optimize the "{op}" kernel.
    Kernel path: tt_llk_{target_arch}/{kernel_path}
    Reference path: tt_llk_{ref_arch}/{kernel_path}
    Analysis: codegen/artifacts/{op}_analysis.md
    Kernel type: {kernel_type}
    Target architecture: {target_arch}

    The kernel already compiles and passes all tests. Optimize with replay
    buffers without breaking correctness. If optimization fails, revert to the
    pre-optimization version.

    WORKTREE_DIR: {WORKTREE_DIR} — cd here before any file I/O.
    LOG_DIR: {LOG_DIR}
```

Wait for completion. The optimizer either:
- Applied a change (compile + tests re-verified) → `OPTIMIZED=true`, `OPTIMIZATION_TYPE=replay`
- Reverted → `OPTIMIZED=false`, `OPTIMIZATION_TYPE=none`

```bash
bash codegen/scripts/refresh_cost.sh   # capture optimizer spend
```

---

## Step 4: Format Generated Code

Run the repo's pre-commit formatters on all generated files so they match CI.
**Not optional** — generated code must pass pre-commit checks before reporting completion.

**LIVE LOG — advance to format:**
```bash
python codegen/scripts/run_json_writer.py advance \
    --log-dir "$LOG_DIR" \
    --new-step "format" \
    --new-message "Running pre-commit formatters on generated files" \
    --prev-result "success" \
    --prev-message "Optimization complete (optimized=${OPTIMIZED})" \
    --agent "format"
```

```bash
source tests/.venv/bin/activate 2>/dev/null || true

FILES="tt_llk_{target_arch}/{kernel_path}"
[ -f "tests/sources/{target_arch}/sfpu_{op}_{target_arch}_test.cpp" ]  && FILES="$FILES tests/sources/{target_arch}/sfpu_{op}_{target_arch}_test.cpp"
[ -f "tests/sources/{target_arch}/{op}_{target_arch}_test.cpp" ]       && FILES="$FILES tests/sources/{target_arch}/{op}_{target_arch}_test.cpp"
[ -f "tests/python_tests/{target_arch}/test_{op}_{target_arch}.py" ]   && FILES="$FILES tests/python_tests/{target_arch}/test_{op}_{target_arch}.py"
[ -f "tests/python_tests/{target_arch}/test_sfpu_{op}_{target_arch}.py" ] && FILES="$FILES tests/python_tests/{target_arch}/test_sfpu_{op}_{target_arch}.py"

# Run pre-commit twice — some hooks need a second pass (e.g. trailing-whitespace after clang-format)
pre-commit run --files $FILES || true
pre-commit run --files $FILES || true

bash codegen/scripts/refresh_cost.sh   # capture format spend
```

Verify compilation still passes after formatting. If the formatter broke the
build (rare — usually a clang-format line-break edge case), inspect the diff
and manually fix. Do NOT revert the formatting. If a manual fix is required,
record a failure with `step=format`, `type=compile_error`.

Set `FORMATTED=true` in the finalize patch.

---

## Step 5: Finalize and Report

### 5a: Gather final metrics

```bash
export END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
export LINES_GENERATED=$(wc -l < tt_llk_{target_arch}/{kernel_path})
```

Determine the terminal status:
- `success` — cycle passed AND `TESTS_PASSED == TESTS_TOTAL` AND `TESTS_TOTAL > 0`
- `compiled` — writer compiled but tests failed, were skipped, or unavailable
- `failed` — never compiled, or all 3 cycles exhausted, or refiner escalated

### 5b: Finalize run.json

```bash
python codegen/scripts/run_json_writer.py finalize \
    --log-dir "$LOG_DIR" \
    --end-time "$END_TIME" \
    --status "$STATUS" \
    --final-result "$FINAL_RESULT" \
    --final-message "Run complete — {op} on {target_arch} (${CYCLE}/${MAX_CYCLES} cycles)" \
    --patch-json "$(python - <<'PY'
import json, os
patch = {
    "phases_total": int(os.environ["PHASES_TOTAL"]),
    "phases_completed": int(os.environ["PHASES_COMPLETED"]),
    "compilation_attempts": int(os.environ["COMPILATION_ATTEMPTS"]),
    "debug_cycles": int(os.environ["DEBUG_CYCLES"]),
    "tests_total": int(os.environ["TESTS_TOTAL"]),
    "tests_passed": int(os.environ["TESTS_PASSED"]),
    "lines_generated": int(os.environ["LINES_GENERATED"]),
    "tests_generated": os.environ["TESTS_GENERATED"].lower() == "true",
    "prettified": False,
    "formatted": True,
    "optimized": os.environ.get("OPTIMIZED", "false").lower() == "true",
    "optimization_type": os.environ.get("OPTIMIZATION_TYPE", "none"),
    "formats_tested": json.loads(os.environ.get("FORMATS_TESTED_JSON", "[]")),
    "formats_excluded": json.loads(os.environ.get("FORMATS_EXCLUDED_JSON", "{}")),
    "obstacle": os.environ.get("OBSTACLE") or None,
    "agents": json.loads(os.environ["AGENTS_JSON"]),
    "tokens": json.loads(os.environ.get("TOKENS_JSON", "{\"input\":0,\"output\":0,\"cache_read\":0,\"cache_creation\":0,\"total\":0}")),
    "refinement_count": int(os.environ.get("REFINEMENT_COUNT", "0")),
    "cycles_attempted": int(os.environ.get("CYCLE", "1")),
    "cycles_cap": int(os.environ.get("MAX_CYCLES", "3")),
}
print(json.dumps(patch))
PY
)"
```

The finalize call closes the last `step_history` entry, flips `status` from
`"running"` to its terminal value, sets `end_time`, and merges the summary
fields. The run.json on disk is now the authoritative record.

```bash
bash codegen/scripts/refresh_cost.sh   # final authoritative refresh — overwrites the TOKENS_JSON
                                       # in the --patch-json above with live-measured spend
```

For batch runs this will later be superseded when `cli_output.json` lands in
`$LOG_DIR` and the dashboard backfills from Anthropic's cumulative
`total_cost_usd`; for interactive runs this is the final number.

### 5c: Append a runs.jsonl entry

`$LOG_DIR/run.json` is the authoritative per-run record — derive the runs.jsonl
entry from it so the two artifacts stay in sync:

```bash
python -c "import json; d=json.load(open('$LOG_DIR/run.json')); print(json.dumps(d))" \
    >> /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl
```

Do not rebuild the entry from shell variables — that would let run.json and runs.jsonl drift.

### 5d: Copy artifacts into LOG_DIR

Each run must be self-contained under `$LOG_DIR`:

```bash
cp codegen/artifacts/{op}_analysis.md       "$LOG_DIR/" 2>/dev/null || true
cp codegen/artifacts/{op}_arch_research.md  "$LOG_DIR/" 2>/dev/null || true   # present only if analyzer produced a separate file
cp codegen/artifacts/{op}_refinement_v*.md  "$LOG_DIR/" 2>/dev/null || true
cp -r codegen/artifacts/{op}_failed_attempt_v*  "$LOG_DIR/" 2>/dev/null || true
cp tt_llk_{target_arch}/{kernel_path}       "$LOG_DIR/"
cp tt_llk_{ref_arch}/{kernel_path}          "$LOG_DIR/ref_$(basename tt_llk_{ref_arch}/{kernel_path})"

# Test files (both naming patterns, silent-fail if absent)
cp tests/sources/{target_arch}/sfpu_{op}_{target_arch}_test.cpp       "$LOG_DIR/" 2>/dev/null || true
cp tests/sources/{target_arch}/{op}_{target_arch}_test.cpp            "$LOG_DIR/" 2>/dev/null || true
cp tests/python_tests/{target_arch}/test_{op}_{target_arch}.py        "$LOG_DIR/" 2>/dev/null || true
cp tests/python_tests/{target_arch}/test_sfpu_{op}_{target_arch}.py   "$LOG_DIR/" 2>/dev/null || true

# Simulator logs — copy BEFORE cleanup_worktree or they are gone permanently
cp tests/python_tests/{target_arch}/emu_*_.log      "$LOG_DIR/" 2>/dev/null || true
cp tests/python_tests/{target_arch}/tt-exalens.log  "$LOG_DIR/" 2>/dev/null || true
```

### 5e: Verify agent logs exist

The following files MUST be present in `$LOG_DIR` (write a placeholder noting
`"Agent ran but did not produce a log"` if any expected file is missing, so
missing logs are visible rather than silently absent):

- `agent_analyzer.md` (always)
- `agent_writer_cycle1.md` (always — cycle 1 writer)
- `agent_writer_cycle2.md`, `agent_writer_cycle3.md` (only when those cycles ran)
- `agent_tester_cycle1.md` (always — cycle 1 tester)
- `agent_tester_cycle2.md`, `agent_tester_cycle3.md` (only when those cycles ran)
- `agent_analysis_refiner_v1.md`, `agent_analysis_refiner_v2.md` (only when the refiner ran)
- `agent_optimizer.md` (only if the optimizer was invoked)

Each agent log MUST follow the structured template defined in the agent
playbooks — **Assumptions**, **Reasoning summary**, **Decisions & trade-offs**,
**Commands run**, **Artifacts read / written**, **Open questions**. If a log is
present but the structured sections are missing, record an `agent_error`
failure (`type: "agent_error"`, `message: "agent_<role>.md missing required
structured sections"`, `resolved: false`) — the dashboard flags these so we can
chase the agent that skipped its contract. Do NOT try to rewrite the log
yourself; the agent owns its own reasoning capture.

### Step 5e.1: Extract subagent transcripts

Dump each subagent's raw Claude Code jsonl transcript into
`$LOG_DIR/transcripts/` as per-agent reasoning / tools / commands markdown,
plus an `INDEX.md` that summarizes tool counts and timing. This makes the
model's chronological reasoning (assistant text, thinking blocks if emitted,
tool calls, trimmed results) auditable after the run without opening raw jsonl
files.

The extractor uses the same session-discovery logic as `session_cost.py` —
it reads `$CLAUDE_SESSION_PID` / `$PPID` to find the active session under
`~/.claude/sessions/<pid>.json`, then walks the subagents directory.
Non-fatal: if no session is found (e.g., running outside the claude CLI),
the extractor logs and exits 1 without blocking the run.

```bash
source /tmp/codegen_run_state.sh 2>/dev/null || true
python codegen/scripts/extract_run_transcripts.py --log-dir "$LOG_DIR" \
    ${SESSION_ID:+--session-id "$SESSION_ID" --project-cwd "$PROJECT_CWD"} \
    || echo "extract_run_transcripts: skipped (non-fatal)"
```

The extractor produces:

- `$LOG_DIR/transcripts/INDEX.md` — one row per agent (type, description,
  start/end timestamps, tool count) + per-agent links.
- `$LOG_DIR/transcripts/NN_{slug}_reasoning.md` — full chronology:
  assistant narration, thinking blocks (when present), every tool call with
  its rendered input, trimmed tool result.
- `$LOG_DIR/transcripts/NN_{slug}_tools.md` — histogram of tool calls and a
  sequence table with sequence #, tool, target, result status, byte count.
- `$LOG_DIR/transcripts/NN_{slug}_commands.md` — flat listings of Bash
  commands (verbatim + description), Confluence page IDs / CQL searches,
  DeepWiki questions, files read / written / edited, glob + grep patterns.

These artifacts make the run reproducible and auditable: the commands file is
a recipe the next engineer can replay; the reasoning file explains *why* each
step was taken without forcing them to rerun the whole pipeline.

### 5f: Write and print the report

Write `codegen/artifacts/{op}_report.md` AND print it to the terminal. The
report is a superset of what the dashboard shows — it MUST aggregate the
agents' structured self-logs (Assumptions / Reasoning / Commands) so the
generated file is self-contained enough that a reader never has to chase
sibling files to understand what happened.

Aggregation rules:

- **Assumptions**: concatenate the `## Assumptions` section from
  `agent_analyzer.md`, `agent_writer_cycle*.md`, and `agent_tester_cycle*.md` (plus
  refiner / optimizer when they ran). If an agent wrote "none", skip that
  agent's section entry. Preserve the agent-origin prefix (e.g., `[analyzer] ...`).
- **Reasoning highlights**: first paragraph (up to 5 sentences) of each
  agent's `## Reasoning summary` section. This is the executive summary.
- **Commands & tools summary**: top-5 Bash commands by length (verbatim, with
  description) from each agent's `transcripts/NN_{slug}_commands.md`, plus the
  per-agent tool histogram from `transcripts/NN_{slug}_tools.md`. If
  extraction failed in Step 5e.1, write "(transcript extraction skipped)"
  under each heading instead of leaving them blank.
- **Artifacts inventory**: every file written or modified inside the worktree
  during the run, with its final path under `$LOG_DIR/` if copied in Step 5d.

Template:

```
========================================
  LLK CodeGen — Generation Complete
========================================
Prompt:           {PROMPT}
Kernel:           {op}
Kernel Type:      {kernel_type}
Target Arch:      {target_arch}
Reference:        tt_llk_{ref_arch}/{kernel_path}
Generated File:   tt_llk_{target_arch}/{kernel_path}
Lines Generated:  {N}
----------------------------------------
Timing:
  Start:          {START_TIME}
  End:            {END_TIME}
  Duration:       {H}h{M}m{S}s
----------------------------------------
Tokens:
  Input:          {N}
  Output:         {N}
  Cache Read:     {N}
  Cache Creation: {N}
  Total:          {N}
  Cost:           ${X.XX} USD  (estimate; see Claude Console for billing)
----------------------------------------
Flow:
  Cycles Used:       {CYCLE}/{MAX_CYCLES}
  Refinements:       {REFINEMENT_COUNT}
  Status:            {STATUS}
----------------------------------------
Quality:
  Compile Attempts:  {COMPILATION_ATTEMPTS} (across writer + tester internal loop)
  Compilation:       PASSED/FAILED
  Functional Tests:  PASSED/FAILED/NOT_AVAILABLE ({TESTS_PASSED}/{TESTS_TOTAL})
  Tests Source:      GENERATED/PRE-EXISTING/NONE
  Formatted:         YES/NO
  Optimized:         YES/NO ({OPTIMIZATION_TYPE})
----------------------------------------
Per Cycle:
  Cycle 1 ({name}): compiles={N}, tester_attempts={N}, result={passed/failed}
  Cycle 2 ({name}): compiles={N}, tester_attempts={N}, result={passed/failed}
  Cycle 3 ({name}): compiles={N}, tester_attempts={N}, result={passed/failed}
----------------------------------------
Failures: {total_failures} ({resolved} resolved, {unresolved} unresolved)
  [compile_error] writer_cycle_1 (writer): unknown type 'vFloat' — RESOLVED
  [test_failure] tester_cycle_2 (tester): mismatch at idx 42 — UNRESOLVED
  ...
  (omit this section if FAILURES is empty)
----------------------------------------
Assumptions made during the run:
  [analyzer] Used ADDR_MOD_7 because every existing Quasar SFPU kernel does;
             would break if the parent wrapper switches to ADDR_MOD_3.
  [writer]   LREG1 reserved for the fill constant (lrelu convention);
             the scaffold's LREG0 remains unused.
  [tester]   UInt16 excluded from the test matrix; VALID_QUASAR_DEST_REG_FORMATS
             rejects it before the kernel runs — kernel code is still correct
             (SFPSTORE mode 6 is ISA-valid).
  ...
  (one line per assumption; prefix with agent origin; omit section if empty)
----------------------------------------
Reasoning highlights:
  [analyzer] {first paragraph of agent_analyzer.md § Reasoning summary}
  [writer]   {first paragraph of agent_writer_cycle{N}.md § Reasoning summary (per cycle)}
  [tester]   {first paragraph of agent_tester_cycle{N}.md § Reasoning summary (per cycle)}
  [refiner]  {...}  (if it ran)
  [optimizer]{...}  (if it ran)
----------------------------------------
Commands & tools summary:
  analyzer:
    Tool histogram: Read×18, Glob×9, Bash×4, Grep×4, Write×2,
                    mcp__atlassian__getConfluencePage×2, ...
    Key bash:
      - grep -n "^SFPLOADI:\|^SFPSTORE:" tt_llk_quasar/instructions/assembly.yaml
      - (verify target instructions exist on Quasar)
      - ...
  writer:
    Tool histogram: Read×12, Bash×8, Write×3, Edit×2
    Key bash:
      - python -m scripts.agent_tools.kernel_template fill --type sfpu --arch quasar
      - CHIP_ARCH=quasar python scripts/compiler.py ...
  tester:
    Tool histogram: Bash×28, Read×14, Edit×6, Write×2
    Key bash:
      - CHIP_ARCH=quasar pytest --compile-producer -n 15 test_fill_quasar.py
      - flock --timeout 900 /tmp/tt-llk-test-simulator.lock bash "$SIM_SCRIPT"
  (full per-agent detail in {LOG_DIR}/transcripts/NN_{slug}_{tools,commands}.md)
----------------------------------------
Formats Tested:
  Float16, Float16_b, Float32 (156 variants total)
  Excluded: UInt16 (not in VALID_QUASAR_DEST_REG_FORMATS — infrastructure limit, not kernel defect)
----------------------------------------
Key Changes:
  1. tt_llk_{target_arch}/{kernel_path} — NEW ({N} helpers)
  2. {additional files touched}
----------------------------------------
Artifacts:
  Analysis:
    - codegen/artifacts/{op}_analysis.md
    - codegen/artifacts/{op}_refinement_v*.md (if refinement occurred)
    - codegen/artifacts/{op}_failed_attempt_v*/ (if refinement occurred)
  Agent self-logs:
    - {LOG_DIR}/agent_analyzer.md
    - {LOG_DIR}/agent_writer_cycle*.md (one per cycle that ran: cycle1, cycle2, cycle3)
    - {LOG_DIR}/agent_tester_cycle*.md (one per cycle that ran)
    - {LOG_DIR}/agent_analysis_refiner_v*.md (if refinement occurred)
    - {LOG_DIR}/agent_optimizer.md (if optimizer ran)
  Subagent transcripts (raw chronology — assistant text + tool calls + results):
    - {LOG_DIR}/transcripts/INDEX.md
    - {LOG_DIR}/transcripts/NN_{slug}_reasoning.md  (one per agent)
    - {LOG_DIR}/transcripts/NN_{slug}_tools.md       (one per agent)
    - {LOG_DIR}/transcripts/NN_{slug}_commands.md    (one per agent)
  Generated File:
    - tt_llk_{target_arch}/{kernel_path}
Metrics:
  - /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl
  - {LOG_DIR}/
Branch:
  - {WORKTREE_BRANCH}
========================================
```

Copy the written report into `$LOG_DIR`:
```bash
cp codegen/artifacts/{op}_report.md "$LOG_DIR/"
```

---

## Inter-Agent Contracts

| From → To | Artifact | Required Contents |
|-----------|----------|-------------------|
| Analyzer → Writer, Tester, Refiner | `artifacts/{op}_analysis.md` | Problem Statement, Target Pattern Survey, Available Instructions, Semantic→Instruction Mapping, Instruction Encoding Constraints, Solution Approach (§6a–§6e), Format Applicability, Complexity & Phases |
| Writer → Tester | kernel file at `tt_llk_{target_arch}/{kernel_path}` | File must exist and compile successfully |
| Writer → Refiner (on compile failure) | kernel file + `$LOG_DIR/agent_writer_cycle{N}.md` + compile stderr | Refiner reads the writer log to distinguish faithful-writer-bad-plan from unfaithful-writer-OK-plan |
| Tester → Optimizer (on PASS) | tested kernel + test files | Kernel passes every variant |
| Tester → Refiner (on STUCK) | `$LOG_DIR/agent_tester_cycle{N}.md` (10-attempt fix log) + test files | Refiner forensically reconstructs what structural assumption in the analysis misled the writer |
| Refiner → Writer | `artifacts/{op}_analysis.md` overwritten in place, with `Refinement History` section at the top | Writer follows the refined plan; prior attempt archived under `artifacts/{op}_failed_attempt_v*/` |
| Optimizer → Report | optimized (or reverted) kernel file | Kernel compiles and all tests still pass |

---

## Key Paths

| Path | Purpose |
|------|---------|
| `tt_llk_blackhole/` | Blackhole LLK (reference architecture) |
| `tt_llk_quasar/` | Quasar LLK (target architecture) |
| `tt_llk_{arch}/instructions/assembly.yaml` | ISA definition (cross-check, use grep — large file) |
| `codegen/references/common-errors.md` | Known error patterns |
| `codegen/agents/quasar/` | All agent playbooks (co-located with this orchestrator) |
| `codegen/skills/llk-arch-lookup/SKILL.md` | Confluence page index and arch-research protocol |

## Commands

```bash
# Compilation check (syntax/type errors)
# -t flags generate constexpr defines (compile-time); -r flags generate RuntimeParams struct fields
# Find correct flags from the Python test's TestConfig(templates=[...], runtimes=[...])
# or map C++ symbols to param classes in tests/python_tests/helpers/test_variant_parameters.py
# See codegen/agents/quasar/llk-kernel-writer.md Step 4 for the full symbol→flag mapping.
cd codegen
source ../tests/.venv/bin/activate
CHIP_ARCH={target_arch} python scripts/compiler.py {path_to_test_source} \
    -t "PARAM(...)" -r "PARAM(...)" -v

# Functional tests — use run_llk_tests.sh (handles flock, venv, simulator path).
# Count variants (determines --maxfail from the 2.1 table in llk-tester.md).
VARIANT_COUNT=$(bash {WORKTREE_DIR}/codegen/scripts/run_llk_tests.sh count \
    --worktree {WORKTREE_DIR} --arch quasar --test test_{kernel_name}_quasar.py)

# Compile-producer step (parallel, no flock).
bash {WORKTREE_DIR}/codegen/scripts/run_llk_tests.sh compile \
    --worktree {WORKTREE_DIR} --arch quasar --test test_{kernel_name}_quasar.py

# Simulator-consumer step (flock-serialised, blocks until done).
# Invoke via Bash tool with timeout: 1800000 — never run_in_background.
bash {WORKTREE_DIR}/codegen/scripts/run_llk_tests.sh simulate \
    --worktree {WORKTREE_DIR} --arch quasar --test test_{kernel_name}_quasar.py \
    --maxfail 5   # omit for verification runs
TEST_EXIT=$?

# Or compile + simulate in one call (exit 2=compile fail, exit 1=test fail, exit 0=pass):
bash {WORKTREE_DIR}/codegen/scripts/run_llk_tests.sh run \
    --worktree {WORKTREE_DIR} --arch quasar --test test_{kernel_name}_quasar.py

# List available tests
ls ../tests/python_tests/quasar/test_*_quasar.py
```
