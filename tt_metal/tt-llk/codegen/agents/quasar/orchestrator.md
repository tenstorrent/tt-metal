# LLK CodeGen Orchestrator

## Git Policy: Read-Only

Read-only git commands are allowed (`git rev-parse`, `git log`, `git status`, `git diff`, `git show`) in the orchestrator and all subagents. **NEVER push, commit, checkout, restore, reset, or otherwise modify** the repo via git. This rule is absolute and applies to all agents spawned by this orchestrator.

---

## Input

The caller (the `codegen/CLAUDE.md` router) tells you only `WORKTREE_DIR` in
the prompt.

All executable steps live as `execute_step_*` functions in
`codegen/scripts/quasar/orchestrator_steps.sh`. Pass the values that vary
per run as arguments; never hand-edit the function bodies. And never read `orchestrator_steps.sh`

```bash
source "{worktree_dir}/tt_metal/tt-llk/codegen/scripts/quasar/orchestrator_steps.sh"
execute_step_validate_input "{worktree_dir}"
```

`execute_step_validate_input` prints `OK: ...`, or `REJECT: <what was wrong>`
and returning non-zero. **If it rejects, report the reason and stop.**
- `WORKTREE_DIR` — absolute path, must exist
- `KERNEL_NAME` — non-empty string
- `TARGET_ARCH` — must be `quasar` (this orchestrator is quasar-only)
- `SFPI_MODE` — must be exactly `true` or `false`
- `WORKTREE_BRANCH` — non-empty string
- `LOG_DIR_BASE` — must be exactly `/proj_sw/user_dev/llk_code_gen`

**CRITICAL: All code writes and file modifications MUST happen inside `$WORKTREE_DIR/tt_metal/tt-llk`, `$WORKTREE_DIR/tt_metal/hw/ckernels`, or `$WORKTREE_DIR/tt_metal/hw/inc/api`.** The worktree has `codegen/` populated with symlinks to the source branch (read-only: `agents/`, `scripts/`, `references/`, `config/`, `CLAUDE.md`, `skills/`) plus a real per-worktree `codegen/artifacts/` directory for this run's outputs. Anything you or a subagent writes outside the worktree leaks into the source branch.

**EVERY BASH COMMAND GIVEN MUST BE EXECUTED AND IN THE ORDER GIVEN BELOW, EXCEPT BLOCKS THAT ONLY ILLUSTRATE USAGE.**

---

## Out of Space — Abort Protocol

Any step that prints the `NO SPACE LEFT ON DEVICE` banner means the device is full. STOP: spawn no agents and run no further steps. Run this one command — it retries the run.json failed-finalize every 30s for up to 10 minutes until the write lands:

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_report_no_space "{step where space ran out}"
```

Then report the run failed with reason "no space on device" and end. Do not run Step 8.

---

## Pipeline Overview

```
analyzer  →  [ writer → tester → refiner ] × up to 3 cycles  →  optimizer  →  format  →  report
```

- **analyzer** (`llk-analyzer.md`) — returns `codegen/artifacts/{op}_analysis.md`.
- **writer** (`llk-kernel-writer.md`) — returns `PASSED` or `FAILED`.
- **tester** (`llk-tester.md`) — returns `PASS`, `STUCK` (→ refiner, cycles 1-2), or `ENV_ERROR` (never routed to the refiner).
- **refiner** (`llk-analysis-refiner.md`) — returns `REFINED` or `ESCALATE`.
- **Loop cap: 3 writer-tester cycles**. Cycles 1 and 2 can hand off to the refiner; cycle 3 CANNOT (the refiner itself caps at v2 = 2 refinements = 3 total cycles). If cycle 3 still fails, the run is reported `failed`.
- **optimizer** and **format** only run on success of tester.

Agent playbooks live in `codegen/agents/quasar/`.

### Agent I/O conventions

- **An agent's status and report are its final message** — read them from the tool result, not from files.
- **Spawn every agent synchronously in the foreground.** Never set `run_in_background` on an Agent or Bash call, and never end your turn while an agent or a `run_test.sh` sim is running. The tool's synchronous return is your wait.
- **Prompt variables (`${CYCLE}`, `${KERNEL_NAME}`, …) are placeholders you fill in yourself** before spawning; the Agent tool does not expand them.
- **The "first meaningful line" is a named field in the agent's report** (writer "Error summary", tester "Last failure signature") — copy it verbatim into the `execute_step_*` argument.

---

### Using `run_json_writer.py`

Wherever this playbook tells you to update run status, that means calling
`run_json_writer.py` — each call emits a progress/status entry for the run. Use
`message` for a mid-step update that doesn't change the current step:

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_message "Tester attempt 4/5 — DATA_MISMATCH on Int32 dest_acc=Yes; fixing LREG1 init ordering"
```

Other subcommands (`init`, `advance`, `phase-start/-test/-end`, `failure`,
`metric`, `finalize`) are used at their respective pipeline points below.

---

## Step 0: Validate Environment and Setup

Before starting, verify prerequisites:

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_validate_env
```

If any issues are reported, **stop and tell the caller** what needs to be fixed before codegen can work.

EXECUTE the following (computes run identity + timing, `mkdir`s `$LOG_DIR`, and
seeds both state files):

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_setup_run
```

### Run JSON rules

Part of your job is to keep the dashboard live by calling `run_json_writer.py`
at every pipeline transition. Each status string you pass MUST:

- **Name the artifact** produced or consumed — "produced `fill_analysis.md` (3
  helpers, 8-row Semantic→Instruction table)", not "Analysis complete".
- **Quantify** — counts, cycle number, compile attempts, refinement version.
  "156/156 variants passed", not "tests passed".
- **On failure, name the first meaningful error** (`file:line symbol`), not the
  category — dashboard readers scan these for regressions.

## Step 1: Identify Kernel Type and Architecture

Do this **before** writing run.json — the init call below records the kernel
type, reference file, and generated-file path, so they must already be in
`state.py`.

Don't guess the kernel category and reference path from keywords — run the
script below:

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_discover_kernels
```
Grep that output for `{op}` (and its synonyms) to find the matching line:
- Blackhole/Wormhole use letter-based names (`llk_unpack_A.h`); Quasar uses
  semantic names (`llk_unpack_unary_operand.h`) — expect to eyeball nearby
  matches rather than get an exact string hit.
- `{kernel_path}` is the matching path **exactly as `execute_step_discover_kernels` printed it** — tt-llk-relative and directly openable from `tt_metal/tt-llk`: e.g. `tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_gelu.h`, `tt_llk_{arch}/llk_lib/llk_math_matmul.h`, or (metal-layer) `../hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_{op}.h`. Do **not** strip or rewrite the prefix — every consumer opens this value directly.
- **SFPU only — `ckernel_sfpu_{op}.h` exists for the same `{ref_arch}` in both
  the tt-llk-layer and metal-layer loop output** → use the metal-layer file as
  the reference, UNLESS it is only a thin wrapper: it `#include`s
  `sfpu/ckernel_sfpu_{op}.h` and its `calculate_*` / `*_init` bodies just forward
  to the tt-llk lib's `_calculate_*_` / `_init_*_` (no raw `TTI_`/`sfpi::` of its
  own). A wrapper carries no algorithm — then use the tt-llk-layer file
  `tt_llk_{ref_arch}/common/inc/sfpu/ckernel_sfpu_{op}.h` as the reference instead.

Determine:
- **Reference architecture** (default: blackhole; wormhole is also a valid
  reference — check both when deciding, since one may have `{op}` and the
  other may not) — the existing implementation to port from

**Target architecture** IS **QUASAR** — where the kernel needs to run

#### Where the generated kernel is written — `GENERATED_KERNEL`

Pass the kernel type, reference arch, and reference path you determined above.
For **SFPU** leave the 4th arg empty — `GENERATED_KERNEL` is derived as
`ckernel_sfpu_{op}.h`. For **non-SFPU** (math/pack/unpack) the reference's name is
letter-based but Quasar uses semantic names, so determine the correct Quasar
semantic dest path yourself (from the discover output / the target survey) and
pass it as the 4th arg:

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_set_kernel_identity "{kernel_type}" "{ref_arch}" "{kernel_path}" "{gen_path or empty for sfpu}"
```

## Step 2: Write the initial run.json

EXECUTE the following. It writes run.json (pipeline steps + kernel identity read
from state), captures the session, seeds every counter, snapshots the agent
playbooks into `$LOG_DIR/instructions/`, and refreshes cost:

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_write_initial_run_json
```

## Step 2b: Hide Existing Implementation (blind regeneration)

EXECUTE the following. When `HIDE_EXISTING_KERNEL=true` it git-removes and commits the target op's existing files (metal wrapper + tt-llk lib impl) on the worktree branch, so the analyzer and writer regenerate blind — following their normal git-read policy they find no prior implementation on the branch. No-op when the flag is unset:

```bash
execute_step_hide_existing_kernel
```

---

## Step 3: Failure Tracking

**Failure tracking**: whenever an agent fails, a compilation fails, a test fails, or an infrastructure error occurs, call `run_json_writer.py failure` with:
```json
{
  "step": "analyzer | writer_cycle_{N} | tester_cycle_{N} | refiner_v{N}",
  "agent": "analyzer | writer | tester | refiner",
  "type": "compile_error | test_failure | agent_error | infra_error",
  "message": "First meaningful line of the error (stderr, traceback, or test output)",
  "resolved": true
}
```
- `step`/`agent`: `analyzer` has no cycle suffix. Elsewhere `{N}` is the current
  `CYCLE` — `writer_cycle_{N}`/`tester_cycle_{N}` for `N` = 1, 2, or 3;
  `refiner_v{N}` for `N` = 1 or 2 only (the refiner never runs on cycle 3). `optimizer`/`format` don't call `failure` — These
  only run after tests pass, and record soft outcomes (`OPTIMIZED`,
  `OPTIMIZATION_TYPE`), not hard failures.
- `type`: Category — `"compile_error"` (compiler stderr), `"test_failure"` (pytest/simulator), `"agent_error"` (agent stuck/crashed), `"infra_error"` (simulator timeout, env issue).
- `message`: The actual error — first meaningful line of compiler stderr, pytest failure, or agent error. Keep it concise but specific enough to diagnose.
- `resolved`: `true` if the issue was fixed during the run (e.g., cycle 1 failed but cycle 2 passed), `false` if it blocked completion.

**You never call `run_json_writer.py failure` by hand.** The `execute_step_*` functions in Step 5 (`execute_step_writer_failed`, `execute_step_tester_stuck`, `execute_step_tester_env_error`, `execute_step_refiner_escalate`) emit this record for you from the arguments you pass. The JSON above just documents the shape they write, so you know what each argument feeds.

## Step 4: Analyzer

Spawn the analyzer. This agent produces the single authoritative analysis document used by every downstream agent

```
Agent tool:
  subagent_type: "general-purpose"
  description: "Analyze {op} for {target_arch}"
  prompt: |
    Read and follow codegen/agents/quasar/llk-analyzer.md to analyze this run's kernel.
    Flow: generation (new kernel).

    The analyzer is the ONLY arch-research step in this pipeline — there is no
    separate arch-lookup agent. You own discovering target instructions,
    register layouts, and format constraints. Follow the
    llk-arch-lookup skill (via Skill tool) for Confluence fetches.
```

- Now WAIT for AGENT completion!
- IF the analyzer's final message begins with `ANALYSIS_FAILED`, treat it as FAILED (skip verification) and use that first line as the failure line.
- THEN **Verify** the analysis doc exists and has all required sections:
  ```bash
  source codegen/scripts/quasar/orchestrator_steps.sh
  execute_step_verify_analysis
  ```
  Prints `OK: ...`, or `MISSING`/`INCOMPLETE` and returns non-zero. If it fails,
  treat the analyzer as FAILED (below).

IF analyzer FAILED: pass the first meaningful line of the analyzer's failure —
the step records it, writes the failure, and finalizes the run as `failed`:
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_analyzer_failed "{first meaningful line of the analyzer's failure}"
```
The run is now finalized as `failed` — skip the writer/tester loop and jump straight to Step 8 (reporting). No kernel was generated, so metrics read 0.

**IF analyzer passed, advance to next stage by EXECUTING the following** (advance
to writer cycle 1, phase-start, refresh cost):

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_analyzer_passed
```

---

## Step 5: Writer → Tester → Refiner Loop (max 3 cycles)

This pseudocode is the required control flow for the loop, not an illustration — it specifies exactly what must happen for every outcome each agent can return (writer: `PASSED` / `FAILED`; tester: `PASS` / `STUCK` / `ENV_ERROR`; refiner: `REFINED` / `ESCALATE`). Run it for `CYCLE ∈ {1, 2, 3}`; break out the moment the tester returns `PASS`. Cycle 3 never hands off to the refiner, so a writer `FAILED` or tester `STUCK` result there finalizes the run as `failed` instead.

```
while CYCLE <= MAX_CYCLES:
    # Writer
    result = run_writer()
    if result == PASSED:
        # Tester
        result = run_tester()
        if result == PASS:
            phase_end(CYCLE, "passed")
            break                                   # -> optimizer
        if result == ENV_ERROR:
            # Infrastructure, not the kernel. Refinement cannot fix a broken
            # simulator — do NOT consume a cycle or invoke the refiner.
            phase_end(CYCLE, "failed")
            OBSTACLE = tester diagnosis
            finalize("failed", "test_failure")    # obstacle carries the infra cause
            return
        # tester STUCK
        phase_end(CYCLE, "failed")
        if CYCLE == MAX_CYCLES:
            finalize("failed", "test_failure")
            return
        # Refiner (only cycles 1 and 2)
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

### Step 5a: Writer

FIRST setup (reset the per-cycle counters):
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_writer_setup
```
THEN Spawn the writer:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Write kernel (cycle ${CYCLE})"
  prompt: |
    Read and follow codegen/agents/quasar/llk-kernel-writer.md to generate this run's kernel.

    You are running inside cycle ${CYCLE} of a max-3 writer-tester loop.

    If a prior refinement occurred (cycle > 1), the analysis has been rewritten
    in place and a "Refinement History" section at the top lists what changed.
    Follow the refined analysis; do not reintroduce approaches from the prior
    failed attempt (archived under codegen/artifacts/${KERNEL_NAME}_failed_attempt_v$((CYCLE - 1))/).
```

- Now WAIT for AGENT completion!
- THEN writer returns `PASSED` / `FAILED` on its final compile check.

**If `SKIP_TESTER` is true** (read it: `python codegen/scripts/state.py --log-dir "$(python codegen/scripts/state.py --worktree-dir "$WORKTREE_DIR" get LOG_DIR)" get SKIP_TESTER`): the writer already validated against the existing tests. Do **not** spawn the tester (Step 5b) or the refiner (Step 5c).
- Writer `PASSED` → EXECUTE `execute_step_tester_passed` (it records success from the counts the writer wrote), then go to Step 6.
- Writer `FAILED` → EXECUTE `execute_step_mark_status failed test_failure`, then jump to Step 8.
- Writer `ENV_ERROR` (emulator/infra unavailable — kernel not implicated) → EXECUTE `execute_step_tester_env_error "{writer's Diagnosis line}"`, then jump to Step 8. Do **not** invoke the refiner or optimizer.

Otherwise (normal flow):

**If writer reports PASSED**: go to Step 5b.

**If writer reports FAILED** (compile broken): pass the first meaningful line
from the writer's "Error summary" and the compile count `1` (the writer runs a
single compile-check). The step records the error, updates the compile counters,
writes the failure + phase-end, and refreshes cost:
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_writer_failed "{first meaningful line from the writer's Error summary}" 1
```

If the step printed `AT_CAP=yes` (cycle 3): EXECUTE the following, then jump to Step 8:
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_mark_status failed compile_error
```
Otherwise (`AT_CAP=no`): go to Step 5c (refiner).

### Step 5b: Tester

EXECUTE the following (advance to the tester, mark the phase running):
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_tester_advance
```

Spawn the tester:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Test kernel (cycle ${CYCLE})"
  prompt: |
    Read and follow codegen/agents/quasar/llk-tester.md to validate this run's kernel.

    You own the full test-and-fix loop with a hard cap of 5 simulator test runs
    (compile-time failures are excluded from the cap). Each run = compile-producer
    + simulator-consumer. Diagnose and fix between runs. On attempt 5's failure,
    return STUCK — the orchestrator will route to the refiner.
```

- WAIT for the Tester finish and return `PASS`, `STUCK`, or `ENV_ERROR`.

**If tester reports PASS**: EXECUTE the following. The step folds the tester's
compiles into the counters, writes phase-end, sets `STATUS`/`FINAL_RESULT` (and
`PHASES_COMPLETED=1`), and refreshes cost:
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_tester_passed
```
Break out of the cycle loop → Step 6 (optimizer).

**If tester reports STUCK**: pass the tester's last failure signature. The step
records it, folds in the tester's compiles, and writes the failure + phase-end:
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_tester_stuck "{tester's Last failure signature}"
```

If the step printed `AT_CAP=yes` (cycle 3): EXECUTE the following, then jump to Step 8:
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_mark_status compiled test_failure
```
Otherwise (`AT_CAP=no`): go to Step 5c (refiner).

**If tester reports ENV_ERROR** (infrastructure broken — simulator down, flock
timeout, missing venv; the kernel is not implicated): pass the tester's
diagnosis line. The step records it as the `OBSTACLE`, writes the infra failure
+ phase-end, and sets the terminal `compiled`/`test_failure` state itself:
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_tester_env_error "{tester's Diagnosis line}"
```
Jump to Step 8. Do **NOT** invoke the refiner.

### Step 5c: Refiner (only for CYCLE ∈ {1, 2})

**Advance to refiner:** pass a short summary of why the cycle failed (the
compile error line, or the tester's last failure signature):
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_refiner_advance "{failure summary of the failed cycle}"
```

Spawn the refiner:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Refine analysis (v${CYCLE})"
  prompt: |
    Read and follow codegen/agents/quasar/llk-analysis-refiner.md to refine this run's analysis.

    You are refinement iteration v${CYCLE}. Your own doc caps you at v2.
    Previous failure: {failure summary of the failed cycle}

    Archive the failed attempt, identify the structural error in the analysis,
    rewrite the impeached sections in place, and return REFINED or ESCALATE.
```

Wait for the refiner to return `REFINED` or `ESCALATE`.

EXECUTE the following (bump refinement/debug/phase counters and patch run.json):
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_refiner_bump
```

**If refiner reports ESCALATE**: pass why it escalated. The step writes the
failure and sets the terminal `STATUS`/`FINAL_RESULT` (derived from the prior
failure kind):
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_refiner_escalate "{why the refiner escalated}"
```
Jump to Step 8.

**If refiner reports REFINED**: EXECUTE the following (increment the cycle,
advance back to the writer, phase-start the new cycle):
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_refiner_refined
```

Loop back to Step 5a.

---

## Step 6: Optimizer (success path only)

Only run after a cycle returned `PASS`. Skip if the tester never passed, or if `KERNEL_TYPE` is not `sfpu` (replay-buffer and SFPI optimizations are SFPU-only) — in that case go straight to Step 7.

### Step 6a: Run the optimizer

Snapshot the pre-optimization kernel (for comparison / rollback reporting):
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_optimizer_snapshot
```

**LIVE LOG — advance to optimizer** (message picks SFPI vs replay-buffer by `SFPI_MODE`):
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_optimizer_advance
```

Start the Optimizer agent:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Optimize kernel"
  prompt: |
    Read and follow codegen/agents/quasar/llk-optimizer.md to optimize this run's kernel.

    The kernel already compiles and passes all tests. Optimize with replay
    buffers or sfpi code without breaking correctness. If optimization fails, revert to the pre-optimization version.
```

Wait for AGENTS completion.

THEN EXECUTE (capture optimizer spend):
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_refresh_cost
```

---

## Step 7: Format Generated Code

**Advance to format:** EXECUTE the following:
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_format_advance
```

Spawn the prettifier:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Prettify kernel"
  prompt: |
    Read and follow codegen/agents/quasar/llk-prettifier.md to refactor this
    run's kernel for maintainability.
```

Wait for completion.

---

## Step 8: Finalize and Report

### 8a: Gather final metrics

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_gather_metrics
```

THEN EXECUTE (snapshot the final generated kernel into `$LOG_DIR` as the bare `ckernel_sfpu_{op}.h` — the optimized/final version the dashboard code section renders beside `pre_opt_*` — and record it in run.json):
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_snapshot_generated_kernel
```

### 8b: Finalize run.json

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_finalize_run
```

The step re-reads every tracked value from state, calls `run_json_writer.py
finalize`, does the final authoritative cost refresh, and appends the run's
`run.json` to `runs.jsonl`.

### 8c: Copy artifacts into LOG_DIR

Each run must be self-contained under `$LOG_DIR`. Simulator logs must be
copied before `cleanup_worktree` runs or they are gone permanently — copy
them as-is:

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_copy_sim_logs
```

Everything else this run created or modified — the generated kernel, the
reference kernel, analysis/refinement artifacts, and test files — is captured
by the `generated.patch` git diff below. Do **not** also snapshot them
individually: enumerating expected filenames misses anything the run touched
that you did not predict, and copying a shared file whole (instead of
diffing it) would clobber unrelated upstream edits on re-apply. The worktree
is dedicated to this single run and was branched from `origin/main`, so
*every* uncommitted change under `tt_llk_${TARGET_ARCH}/`, `tests/`, and
`codegen/artifacts/` is, by definition, this run's output. Let git enumerate
it:

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_write_generated_patch
```

### 8d: Extract subagent transcripts

EXECUTE the following:
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_extract_transcripts
```

### 8e: Write and print the report

The report is built from `run.json` + the agent self-logs + the transcripts (it
aggregates each agent's Assumptions / Reasoning / tool+command summary and all
the run metrics). EXECUTE the following — it writes
`codegen/artifacts/{KERNEL_NAME}_report.md` and prints it:

```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_write_report
```

THEN copy it into `$LOG_DIR`:
```bash
source codegen/scripts/quasar/orchestrator_steps.sh
execute_step_copy_report
```

---
