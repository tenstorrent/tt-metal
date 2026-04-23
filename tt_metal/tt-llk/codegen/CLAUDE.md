# LLK CodeGen Orchestrator

## Git Policy: Read-Only

Read-only git commands are allowed (`git rev-parse`, `git log`, `git status`, `git diff`, `git show`) in the orchestrator and all subagents. **NEVER push, commit, checkout, restore, reset, or otherwise modify** the repo via git. This rule is absolute and applies to all agents spawned by this orchestrator.

---

When a user asks to **"generate {kernel} for {target_arch}"**, follow this workflow using the **Agent tool** to spawn subagents. Each agent runs in its own context, keeping the main conversation clean.

The system is designed to work across architectures. Agents must **discover** architectural patterns from authoritative sources — not rely on hardcoded knowledge.

---

## CRITICAL: Incremental Phase-Based Generation

**Kernels MUST be generated incrementally, one sub-kernel at a time.**

Most kernel files contain multiple sub-kernels (e.g., a basic variant, a dual-input variant, an optimized variant). Each sub-kernel is a group of related functions (init, main, uninit, mop_config) that form a logical unit.

**The rule**: Write one sub-kernel → compile → test → only proceed to the next sub-kernel when the current one passes. Never write the entire file at once.

**Why**: A single wrong architectural assumption in a monolithic write poisons 400+ lines with no working baseline. Incremental phases give test feedback early, keep blast radius small, and give agents a working foundation to build on.

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

Record the start time and create a unique log directory for this run:

```bash
START_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
RUN_ID=$(date +%Y-%m-%d)_{kernel}_{arch}_$(head -c 4 /dev/urandom | xxd -p)
LOG_DIR=/proj_sw/user_dev/llk_code_gen/quasar/$RUN_ID
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
mkdir -p $LOG_DIR/instructions
```

Track these variables throughout the run for metrics:
- `START_TIME` — captured above
- `PROMPT` — the original user prompt verbatim (e.g., "Generate gelu for Quasar")
- `BATCH_ID` — if provided via environment variable `CODEGEN_BATCH_ID`, use it; otherwise `null`
- `MODEL` — if provided via environment variable `CODEGEN_MODEL`, use it; otherwise detect from the current Claude CLI model (e.g., "opus", "sonnet", "haiku")
- `RUN_TYPE` — if `CODEGEN_BATCH_ID` is set, use `"ci"`; otherwise `"manual"`
- `GIT_COMMIT` — the repo commit hash at run start, captured above
- `COMPILATION_ATTEMPTS=0` — increment each time `check_compile.py` is run
- `TESTS_GENERATED=false` — set to `true` if the test-writer agent is spawned (Step 3d)
- `PER_PHASE=[]` — build up per-phase results as phases complete
- `FAILURES=[]` — append every failure encountered during the run (see below)

**Failure tracking**: Whenever an agent fails, a compilation fails, a test fails, or an infrastructure error occurs, append an entry to `FAILURES`:
```json
{
  "step": "compile_phase_1|test_phase_2|analyzer|tester|infra",
  "agent": "writer|debugger|tester|analyzer|planner|arch_lookup",
  "type": "compile_error|test_failure|agent_error|infra_error",
  "message": "First meaningful line of the error (stderr, traceback, or test output)",
  "resolved": true
}
```
- `step`: Which pipeline step failed (e.g., `"compile_phase_1"`, `"test_phase_2"`, `"analyzer"`, `"final_regression"`)
- `agent`: Which agent was running when the failure occurred
- `type`: Category — `"compile_error"` (compiler stderr), `"test_failure"` (pytest/simulator), `"agent_error"` (agent stuck/crashed), `"infra_error"` (simulator timeout, env issue)
- `message`: The actual error — first meaningful line of compiler stderr, pytest failure, or agent error. Keep it concise but specific enough to diagnose.
- `resolved`: `true` if the issue was fixed during the run, `false` if it blocked completion

Copy the agent playbooks used (snapshot for reproducibility):
```bash
cp codegen/agents/llk-analyzer.md $LOG_DIR/instructions/
cp codegen/agents/llk-planner.md $LOG_DIR/instructions/
cp codegen/agents/llk-kernel-writer.md $LOG_DIR/instructions/
cp codegen/agents/llk-debugger.md $LOG_DIR/instructions/
cp codegen/agents/llk-phase-tester.md $LOG_DIR/instructions/
cp codegen/agents/llk-regression-tester.md $LOG_DIR/instructions/
cp codegen/agents/llk-test-writer.md $LOG_DIR/instructions/
cp codegen/agents/llk-optimizer.md $LOG_DIR/instructions/
# prettifier disabled — skip copy
```

Pass `LOG_DIR` to every agent prompt so they can self-log their reasoning.

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

## Workflow: Spawn Agents Sequentially

### Step 1: Research Target Architecture

**This step is mandatory.** Before analyzing any code, gather architecture knowledge from authoritative sources.

Spawn an agent:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Research {target_arch} architecture for {op} kernel"
  prompt: |
    Read and follow codegen/agents/llk-arch-lookup.md for the full page index and process.

    Research the {target_arch} architecture to understand what's needed for implementing
    a {kernel_type} kernel ({op}).

    ## What to fetch from Confluence (use the page index in llk-arch-lookup.md):

    For SFPU kernels, you MUST fetch these pages (in order of priority):
    1. Quasar/Trinity SFPU Micro-Architecture Spec (page 1256423592) — THE key reference
    2. Tensix SFPU Instruction Set Architecture (page 1170505767) — per-instruction details
    3. srcS registers (page 141000706) — SFPU reads from here
    4. Dest register (page 195493892) — SFPU writes here
    5. Tensix Formats (page 237174853) — MANDATORY: comprehensive format reference
    6. Dest storage formats (page 80674824) — format layout in Dest register
    7. SrcA/B storage formats (page 83230723) — format layout in source registers
    8. Search ISA child pages (under page 1613201604) for specific instructions the kernel uses

    For math kernels, fetch: FPU MAS (881197063), data flow (57933869), srcA/srcB/Dest pages,
    PLUS format pages: Tensix Formats (237174853), Neo FPU Supported Formats (1124335662),
    Neo FPU Different-Input-Format Combos (1127908233), Dest/SrcAB storage formats.
    For pack/unpack, search Confluence for relevant pages + fetch register file pages +
    format pages: Tensix Formats (237174853), Implied Formats (547258441), storage format pages.

    Also query DeepWiki (repo: tenstorrent/tt-isa-documentation) for Blackhole comparison.
    Cross-check instructions exist in: tt_llk_{target_arch}/instructions/assembly.yaml

    ## Output
    Write a thorough architecture brief to: codegen/artifacts/{op}_arch_research.md
    Include:
    - SFPU execution model (lanes, slices, rows, how instructions execute)
    - Register file layouts (SrcS, Dest, GPRs, LREGs) with sizes and access patterns
    - Per-instruction details for every instruction the kernel needs
    - Data format support and conversion rules
    - **Format support matrix** — MANDATORY: start from the FULL set of Quasar-supported
      formats from QUASAR_DATA_FORMAT_ENUM_VALUES in tests/python_tests/helpers/format_config.py
      (Float32, Tf32, Float16, Float16_b, MxFp8R, MxFp8P, Int32, Int8, UInt8, UInt16, Int16).
      For each format, determine if the SFPU can load/store it and if the kernel's operation
      is semantically valid. Do NOT limit to what the Blackhole reference supports — Quasar
      has formats Blackhole lacks (Int16, MxFp8R, MxFp8P, Tf32). Include format-specific
      constraints (e.g., dest_acc requirements, MX unpacking behavior)
    - Pipeline constraints, instruction ordering, LOADMACRO rules
    - Blackhole differences (if relevant for porting)
    - Source reference for each fact (page ID and section)

    Be thorough — downstream agents depend on this research being complete.

    LOG_DIR: {LOG_DIR}
```

Wait for completion. **Verify** that `codegen/artifacts/{op}_arch_research.md` exists.

### Step 2: Analyze Reference Implementation

Spawn an agent:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Analyze {op} kernel"
  prompt: |
    Read and follow codegen/agents/llk-analyzer.md to analyze the "{op}" kernel.
    Kernel type: {kernel_type}
    Reference architecture: {ref_arch}
    Target architecture: {target_arch}
    Reference path: tt_llk_{ref_arch}/{kernel_path}
    Architecture research: codegen/artifacts/{op}_arch_research.md
    Output your analysis to: codegen/artifacts/{op}_analysis.md

    CRITICAL: Before reading the reference, you MUST read the target integration points:
    1. Test harness: Find and read tests/sources/*{op}*.cpp (look for #ifdef ARCH_{TARGET_UPPER})
    2. Parent file: Read the target file that #includes this kernel
    3. Closest existing target kernel: Read the most similar existing target kernel of this type line-by-line
    Document the target-expected API (function signatures, template params, target-only features, reference-only features to drop).

    CRITICAL: You MUST identify sub-kernel phases in your analysis. See the
    "Sub-Kernel Phases" section in llk-analyzer.md for the required output format.

    LOG_DIR: {LOG_DIR}
```

Wait for completion. Agent returns summary of analysis **including the phase plan**.

**Verify**: Check that `codegen/artifacts/{op}_analysis.md` exists and contains: kernel_type, functions list, dependencies, complexity classification, and sub-kernel phases. If missing, **stop and report the error to the user**.

### Step 2b: Extract Phase Plan

From the analyzer's output, extract the ordered list of phases. Each phase has:
- **Phase name** (a short label for the sub-kernel group)
- **Functions** (the functions belonging to this phase)
- **Test file(s)** (which test file validates this phase, if any)
- **Dependencies** (any prior phases that must pass first)

If the analysis identifies only a single sub-kernel (e.g., simple SFPU ops), there is one phase and the workflow is identical to a single-pass generation.

### Step 3: Loop Over Phases

For each phase **in order**, run Steps 3a–3e. Only proceed to the next phase when the current phase passes tests (or has no applicable test).

#### Step 3a: Plan Phase

Spawn an agent:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Plan {op} phase {N}"
  prompt: |
    Read and follow codegen/agents/llk-planner.md to plan the "{op}" kernel.
    Kernel type: {kernel_type}
    Target architecture: {target_arch}
    Analysis: codegen/artifacts/{op}_analysis.md
    Architecture research: codegen/artifacts/{op}_arch_research.md
    Output your spec to: codegen/artifacts/{op}_phase{N}_spec.md

    IMPORTANT - INCREMENTAL PHASE:
    You are planning ONLY phase {N}: "{phase_name}"
    Functions to plan: {phase_functions}
    Previously completed phases: {list of completed phase names, or "none"}

    Plan ONLY the functions listed above. Do not plan functions from other phases.
    If prior phases exist, their functions are already written and tested — your
    phase must be compatible with them but do not redesign them.

    CRITICAL: Design from target patterns, not reference patterns. The analysis contains
    target-expected API from the test harness and parent file — template params and
    function signatures MUST match those, not the reference.
    Verify init/uninit symmetry: uninit must restore what init changes.

    LOG_DIR: {LOG_DIR}
```

Wait for completion. **Verify** that `codegen/artifacts/{op}_phase{N}_spec.md` exists.

#### Step 3b: Generate Phase Code

Spawn an agent:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Generate {op} phase {N}"
  prompt: |
    Read and follow codegen/agents/llk-kernel-writer.md to generate the "{op}" kernel.
    Kernel type: {kernel_type}
    Target architecture: {target_arch}
    Spec: codegen/artifacts/{op}_phase{N}_spec.md
    Output to: tt_llk_{target_arch}/{kernel_path}
    Run compilation check after writing.

    IMPORTANT - INCREMENTAL PHASE:
    You are implementing ONLY phase {N}: "{phase_name}"
    Functions to write: {phase_functions}
    Previously completed phases: {list of completed phase names, or "none"}

    If prior phases exist, READ the current file first. Their functions are already
    written and tested — APPEND your new functions after them. Do NOT modify
    previously written functions.

    If this is phase 1, create the file with includes/headers and write your functions.

    CRITICAL: Before writing code, verify EVERY function signature against:
    1. The target test harness (tests/sources/*{op}*.cpp, #ifdef ARCH_{TARGET_UPPER} branch)
    2. The target parent file (tt_llk_{target_arch}/llk_lib/llk_{type}.h)
    3. The closest existing target kernel of this type
    If the spec conflicts with target sources, target sources WIN.
    Do NOT port reference features that the target test/parent don't reference.

    LOG_DIR: {LOG_DIR}
```

Wait for completion. Agent returns compilation result (PASSED or FAILED).

**Metrics**: Increment `COMPILATION_ATTEMPTS` by 1 (the writer always runs one compile check).

#### Step 3c: Debug Phase (if needed)

If Step 3b reports compilation failure, spawn debugger:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Debug {op} phase {N}"
  prompt: |
    Read and follow codegen/agents/llk-debugger.md to fix compilation errors.
    Kernel: {op}
    Kernel type: {kernel_type}
    Target architecture: {target_arch}
    Kernel path: tt_llk_{target_arch}/{kernel_path}
    Architecture research: codegen/artifacts/{op}_arch_research.md
    Max 5 fix attempts. Report when compilation passes or if stuck.

    IMPORTANT - INCREMENTAL PHASE:
    You are debugging ONLY phase {N}: "{phase_name}"
    Functions in this phase: {phase_functions}
    Do NOT modify functions from previously completed phases — they are tested and working.

    LOG_DIR: {LOG_DIR}
```

**Metrics**: Increment `COMPILATION_ATTEMPTS` by the number of compile attempts the debugger made (up to 5). Increment `debug_cycles` by 1.

**If debugger reports STUCK** after 5 attempts: **stop and report to the user** with the blocking error details. Do NOT proceed to the next phase.

#### Step 3d: Create Tests (if needed)

After compilation passes, check if a test exists for this kernel:
```bash
ls tests/python_tests/{target_arch}/test_{op}_{target_arch}.py 2>/dev/null
ls tests/python_tests/{target_arch}/test_sfpu_*{op}*_{target_arch}.py 2>/dev/null
```

If a test file exists, skip to Step 3e.

If NO test exists, spawn the test-writer agent:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Create tests for {op}"
  prompt: |
    Read and follow codegen/agents/llk-test-writer.md to create functional tests.
    Kernel: {op}
    Kernel type: {kernel_type}
    Target architecture: {target_arch}
    Kernel path: tt_llk_{target_arch}/{kernel_path}

    LOG_DIR: {LOG_DIR}
```

Wait for completion. The agent will create:
- `tests/sources/{target_arch}/sfpu_{op}_{target_arch}_test.cpp`
- `tests/python_tests/{target_arch}/test_{op}_{target_arch}.py`
- Any required infrastructure changes (SfpuType enum entries, etc.)

If the agent reports BLOCKED, record `test_result: "skipped"` for this phase and continue.

#### Step 3e: Test Phase

After compilation passes (and tests exist), spawn the tester:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Test {op} phase {N}"
  prompt: |
    Read and follow codegen/agents/llk-phase-tester.md to test the "{op}" kernel.
    Kernel: {op}
    Kernel type: {kernel_type}
    CHIP_ARCH: {target_arch}
    Architecture: {target_arch}

    IMPORTANT - INCREMENTAL PHASE:
    This is phase {N}: "{phase_name}"
    Functions in this phase: {phase_functions}
    Previously completed phases: {list of completed phase names, or "none"}

    You MUST CREATE a phase-specific test (C++ source + Python test) that exercises
    ONLY the functions from this phase. Do NOT use existing tests — they expect the
    complete kernel.

    After your phase test passes, also re-run phase tests from previous phases to
    confirm no regressions.

    LOG_DIR: {LOG_DIR}
```

Wait for test results. If PASSED, mark this phase complete and continue to the next phase.

If FAILED, spawn the debugger again but this time with test failure details instead of compilation errors:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Fix {op} test failure phase {N}"
  prompt: |
    Read and follow codegen/agents/llk-debugger.md to fix RUNTIME/TEST failures.
    Kernel: {op}
    Kernel type: {kernel_type}
    Target architecture: {target_arch}
    Kernel path: tt_llk_{target_arch}/{kernel_path}
    Architecture research: codegen/artifacts/{op}_arch_research.md

    THIS IS A TEST FAILURE, NOT A COMPILATION ERROR.
    Follow the "Runtime/Functional Debugging" section in llk-debugger.md.

    Test failure details:
    {paste the test output/error from the tester agent}

    IMPORTANT - INCREMENTAL PHASE:
    You are debugging ONLY phase {N}: "{phase_name}"
    Functions in this phase: {phase_functions}
    Do NOT modify functions from previously completed phases.

    After fixing, recompile to ensure compilation still passes:
      cd codegen && source ../tests/.venv/bin/activate
      PYTHONPATH=.. python scripts/check_compile.py ../{kernel_path} -v

    Max 5 fix attempts.

    LOG_DIR: {LOG_DIR}
```
After the debugger fixes the code, return to Step 3e (test) to verify.
Max 2 debug→test cycles per phase before escalating to the user.

**Metrics**: When a phase completes (pass or fail), record its per-phase result:
```
{
  "phase": {N},
  "name": "{phase_name}",
  "compilation_attempts": {N},  // compile attempts for THIS phase only
  "debug_cycles": {N},          // debug rounds for THIS phase only
  "test_result": "passed|failed|skipped",
  "compile_errors": [            // ALL compilation errors, in order
    {"attempt": 1, "error": "first error message (first line of stderr)"},
    {"attempt": 2, "error": "second error message after first fix"}
  ]
}
```
`compile_errors` must capture **every** failed compilation attempt's error message (first meaningful line of compiler stderr). Empty array `[]` if the phase compiled clean on the first try. This history is critical for understanding debug loops and recurring error patterns.

Append to the `PER_PHASE` list.

#### Step 3f: Cleanup Phase Tests

After **all phases pass**, delete the temporary phase test files:
```bash
rm -f tests/sources/{op}_phase*_test.cpp
rm -f tests/python_tests/test_{op}_phase*.py
```

### Step 4: Final Regression

After all phases complete and phase tests are cleaned up, run the existing repo tests that exercise this kernel to confirm the complete kernel works end-to-end:

```bash
flock --timeout 900 /tmp/tt-llk-test-simulator.lock bash -c '
  STALE=$(lsof -ti :5556 2>/dev/null || true)
  [ -n "$STALE" ] && echo "Killing stale port 5556 processes: $STALE" && echo "$STALE" | xargs kill -9 2>/dev/null || true
  pkill -9 -f "tt-exalens.*--port=5556" 2>/dev/null || true
  sleep 1
  source ../tests/.venv/bin/activate
  cd ../tests/python_tests/quasar
  TT_UMD_SIMULATOR_PATH=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-1x3 CHIP_ARCH=quasar pytest -x --run-simulator --port=5556 test_{op}_quasar.py
'
```

If no existing test covers this kernel, report NOT_AVAILABLE and move to Step 10.
If tests FAIL, return to the debug→test loop (Step 3c/3e) for the failing phase.

### Step 5: Optimize (SFPU kernels only)

After the final regression passes, check if the Blackhole reference uses replay buffers:

```bash
grep -c "replay\|load_replay_buf" tt_llk_blackhole/{kernel_path}
```

If the count is > 0, first snapshot the pre-optimization kernel for comparison:
```bash
cp tt_llk_{target_arch}/{kernel_path} {LOG_DIR}/pre_opt_$(basename tt_llk_{target_arch}/{kernel_path})
```

Then spawn the optimizer agent:
```
Agent tool:
  subagent_type: "general-purpose"
  description: "Optimize {op} kernel"
  prompt: |
    Read and follow codegen/agents/llk-optimizer.md to optimize the "{op}" kernel.
    Kernel path: tt_llk_{target_arch}/{kernel_path}
    Reference path: tt_llk_{ref_arch}/{kernel_path}
    Architecture research: codegen/artifacts/{op}_arch_research.md
    Kernel type: {kernel_type}
    Target architecture: {target_arch}

    The kernel already compiles and passes all tests. Your job is to optimize
    it with replay buffers without breaking correctness.

    If optimization fails, revert to the pre-optimization version.

    LOG_DIR: {LOG_DIR}
```

Wait for completion. The optimizer will either:
- Return an optimized kernel (compile + test verified) → set `"optimized": true`
- Revert to the original (optimization failed) → set `"optimized": false`

If the Blackhole reference does NOT use replay buffers, skip this step and set `"optimized": false`.

**Metrics**: Record in the run JSON:
- `"optimized": true/false` — whether optimization was applied
- `"optimization_type": "replay|none"` — what was optimized

### Steps 6–9: Prettifier (DISABLED)

The prettifier step is currently disabled. Set `"prettified": false` in the run metrics.

### Step 9b: Format Generated Code

Run the repo's pre-commit formatters on all generated files so they match the style enforced by CI. This step is **not optional** — generated code must pass pre-commit checks before reporting completion.

```bash
source tests/.venv/bin/activate

# Collect all generated files
FILES="tt_llk_{target_arch}/{kernel_path}"

# Add test files if they were generated
[ -f "tests/sources/{target_arch}/sfpu_{op}_{target_arch}_test.cpp" ] && FILES="$FILES tests/sources/{target_arch}/sfpu_{op}_{target_arch}_test.cpp"
[ -f "tests/sources/{target_arch}/{op}_{target_arch}_test.cpp" ] && FILES="$FILES tests/sources/{target_arch}/{op}_{target_arch}_test.cpp"
[ -f "tests/python_tests/{target_arch}/test_{op}_{target_arch}.py" ] && FILES="$FILES tests/python_tests/{target_arch}/test_{op}_{target_arch}.py"
[ -f "tests/python_tests/{target_arch}/test_sfpu_{op}_{target_arch}.py" ] && FILES="$FILES tests/python_tests/{target_arch}/test_sfpu_{op}_{target_arch}.py"

# Run pre-commit on the generated files (auto-fixes in place)
pre-commit run --files $FILES || true
# Run a second pass — some hooks (e.g., trailing-whitespace after clang-format) may need a re-run
pre-commit run --files $FILES || true
```

After formatting, **verify compilation still passes**:
```bash
cd codegen
source ../tests/.venv/bin/activate
PYTHONPATH=.. python scripts/check_compile.py ../{kernel_path} -v
```

If compilation fails after formatting (rare — usually a clang-format line-break issue), inspect the diff and manually fix. Do NOT revert the formatting.

**Metrics**: Set `"formatted": true` in the run JSON. If formatting broke compilation and required a manual fix, record a failure entry with `"step": "format"`, `"type": "compile_error"`.

### Step 10: Report Completion and Log Metrics

After all agents complete:

1. **Record end time**:
```bash
END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
```

2. **Count lines generated**:
```bash
LINES_GENERATED=$(wc -l < tt_llk_{target_arch}/{kernel_path})
```

3. **Append a run entry** to `/proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl`:
```json
{
  "kernel": "{op}",
  "kernel_type": "{sfpu|math|pack|unpack}",
  "arch": "{target_arch}",
  "reference_arch": "{ref_arch}",
  "reference_file": "tt_llk_{ref_arch}/{kernel_path}",
  "generated_file": "tt_llk_{target_arch}/{kernel_path}",
  "start_time": "{START_TIME}",
  "end_time": "{END_TIME}",
  "phases_total": 0,
  "phases_completed": 0,
  "compilation_attempts": 0,
  "debug_cycles": 0,
  "tests_total": 0,
  "tests_passed": 0,
  "lines_generated": 0,
  "tests_generated": false,
  "prettified": false,
  "formatted": true,
  "optimized": false,
  "optimization_type": "none",
  "formats_tested": ["Float16", "Float16_b", "Float32"],
  "formats_excluded": {"Int32": "requires instr_mod1=0, not implemented"},
  "status": "success",
  "obstacle": null,
  "failures": [
    {
      "step": "compile_phase_1",
      "agent": "writer",
      "type": "compile_error",
      "message": "unknown type 'vFloat' — SFPU vector type not available on target",
      "resolved": true
    }
  ],
  "per_phase": [
    {
      "phase": 1,
      "name": "{phase_name}",
      "compilation_attempts": 0,
      "debug_cycles": 0,
      "test_result": "passed",
      "compile_errors": [],
      "test_details": null
    }
  ],
  "prompt": "{original_prompt}",
  "batch_id": null,
  "tokens": {
    "input": 0,
    "output": 0,
    "cache_read": 0,
    "total": 0
  },
  "model": "{MODEL}",
  "run_type": "{RUN_TYPE}",
  "agents": ["analyzer", "planner", "writer", "tester", "debugger", "optimizer"],
  "run_id": "{RUN_ID}",
  "git_commit": "{GIT_COMMIT}",
  "log_dir": "logs/{RUN_ID}"
}
```
**Write as a single line** (the above is expanded for readability). Use actual values, not placeholders.

**Notes on special fields**:
- `status`: Use three-way classification:
  - `"success"` — compiles AND all tests pass (`tests_passed == tests_total` and `tests_total > 0`)
  - `"compiled"` — compiles but tests failed, were skipped, or unavailable (`tests_total == 0` or `tests_passed < tests_total`)
  - `"failed"` — does not compile
- `prompt`: Store the exact user prompt that initiated this run (e.g., "Generate gelu for Quasar")
- `batch_id`: Use `$CODEGEN_BATCH_ID` environment variable if set, otherwise `null`. The batch runner script sets this to group runs from a single session.
- `model`: Use `$CODEGEN_MODEL` environment variable if set (e.g., "opus", "sonnet", "haiku"). Otherwise, detect from the current Claude CLI model. The batch runner script sets this to track which model was used.
- `run_type`: `"ci"` if `$CODEGEN_BATCH_ID` is set (indicates a scheduled/automated batch run), `"manual"` otherwise (interactive session). This lets the dashboard distinguish Friday CI runs from ad-hoc manual runs.
- `tests_generated`: `true` if the test-writer agent was spawned to create new tests (Step 3d), `false` if pre-existing tests were found and used. This distinguishes runs that had existing test coverage from ones that had to generate their own.
- `formats_tested`: Array of DataFormat names included in the test format list (e.g., `["Float16", "Float16_b", "Float32", "Tf32", "MxFp8R", "MxFp8P"]`). Extract from the planner's spec "Recommended Test Formats" section. Use `[]` if no tests were generated.
- `formats_excluded`: Object mapping excluded format names to reasons (e.g., `{"UInt16": "not in VALID_QUASAR_DEST_REG_FORMATS"}`). Only include Quasar-supported formats that were excluded. Use `{}` if all formats are tested.
- `tokens`: If token usage is not available from the CLI output, set all values to `0`. When running via `claude -p "..." --output-format json`, the response includes `usage.input_tokens`, `usage.output_tokens`, and `usage.cache_read_input_tokens` — the batch runner script will pass these via `$CODEGEN_TOKENS_INPUT`, `$CODEGEN_TOKENS_OUTPUT`, `$CODEGEN_TOKENS_CACHE_READ` environment variables. `total = input + output`.

4. **Write the report** to `codegen/artifacts/{op}_report.md` AND print it directly to the terminal:

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
----------------------------------------
Tokens:
  Input:          {N}
  Output:         {N}
  Cache Read:     {N}
  Total:          {N}
----------------------------------------
Quality:
  Phases:           {completed}/{total}
  Compile Attempts: {N} (across all phases)
  Debug Cycles:     {N}
  Compilation:      PASSED/FAILED
  Functional Tests: PASSED/FAILED/NOT_AVAILABLE ({passed}/{total})
  Tests Source:     GENERATED/PRE-EXISTING/NONE
  Formatted:        YES/NO
  Prettified:       DISABLED
----------------------------------------
Per Phase:
  Phase 1 ({name}): compile_attempts={N}, debug={N}, test={passed/failed}
  Phase 2 ({name}): compile_attempts={N}, debug={N}, test={passed/failed}
  ...
----------------------------------------
Failures: {total_failures} ({resolved} resolved, {unresolved} unresolved)
  [compile_error] compile_phase_1 (writer): unknown type 'vFloat' — RESOLVED
  [test_failure] test_phase_2 (tester): mismatch at idx 42 — UNRESOLVED
  ...
  (omit this section if FAILURES is empty)
----------------------------------------
Artifacts:
  - codegen/artifacts/{op}_arch_research.md
  - codegen/artifacts/{op}_analysis.md
  - codegen/artifacts/{op}_phase{N}_spec.md (one per phase)
Metrics:
  - /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl
  - {LOG_DIR}/
========================================
```

This conclusion MUST be both:
1. **Written to file**: `codegen/artifacts/{op}_report.md`
2. **Output as text** in your response so the user sees it directly in the terminal

5. **Copy all artifacts to LOG_DIR** so each run is self-contained:
```bash
cp codegen/artifacts/{op}_report.md {LOG_DIR}/
cp codegen/artifacts/{op}_arch_research.md {LOG_DIR}/
cp codegen/artifacts/{op}_analysis.md {LOG_DIR}/
cp codegen/artifacts/{op}_phase*_spec.md {LOG_DIR}/
cp tt_llk_{target_arch}/{kernel_path} {LOG_DIR}/
# Copy reference kernel (ref_ prefix to avoid name collision with generated kernel)
cp tt_llk_{ref_arch}/{kernel_path} {LOG_DIR}/ref_$(basename tt_llk_{ref_arch}/{kernel_path})
# Copy test files — both naming patterns, silent fail if absent
cp tests/sources/{target_arch}/sfpu_{op}_{target_arch}_test.cpp {LOG_DIR}/ 2>/dev/null || true
cp tests/sources/{target_arch}/{op}_{target_arch}_test.cpp {LOG_DIR}/ 2>/dev/null || true
cp tests/python_tests/{target_arch}/test_{op}_{target_arch}.py {LOG_DIR}/ 2>/dev/null || true
cp tests/python_tests/{target_arch}/test_sfpu_{op}_{target_arch}.py {LOG_DIR}/ 2>/dev/null || true
# Copy emulator logs (emu_*_.log and tt-exalens.log) from the test directory
cp tests/python_tests/{target_arch}/emu_*_.log {LOG_DIR}/ 2>/dev/null || true
cp tests/python_tests/{target_arch}/tt-exalens.log {LOG_DIR}/ 2>/dev/null || true
# Copy the runs.jsonl entry as a standalone run.json for this run
```
Also write `{LOG_DIR}/run.json` containing **just this run's** JSONL entry (same content appended to runs.jsonl, but pretty-printed JSON). This makes each LOG_DIR a complete, self-contained record.

6. **Verify all agent logs exist** in LOG_DIR. The following files MUST be present:
   - `agent_analyzer.md`
   - `agent_planner.md`
   - `agent_writer.md`
   - `agent_phase_tester.md`
   - `agent_test_writer.md` (only if tests were created)
   - `agent_arch_lookup.md` (from the arch research agent)
   - `agent_debugger.md` (only if the debugger was invoked)
   - `agent_optimizer.md` (only if the optimizer was invoked)

If any expected file is missing, write a placeholder noting `"Agent ran but did not produce a log"` so missing logs are visible rather than silently absent.

---

## Inter-Agent Contracts

Each stage produces artifacts that the next stage consumes:

| From → To | Artifact | Required Contents |
|-----------|----------|-------------------|
| Researcher → Analyzer, Planner | `artifacts/{op}_arch_research.md` | Available instructions, register layout, arch constraints, **format support matrix** (all Quasar formats evaluated, not just reference formats) |
| Analyzer → Planner, Test Writer | `artifacts/{op}_analysis.md` | kernel_type, function signatures, dependencies, complexity_class, key constructs, sub-kernel phases, **format support** (starting from full Quasar format set, with per-format applicability and technical justification for exclusions) |
| Planner → Writer, Test Writer | `artifacts/{op}_phase{N}_spec.md` | target_file_path, instruction_sequence (pseudocode), resource_allocation, includes, **recommended test formats** (exact format list, filtering rules, MX/int handling) |
| Writer → Debugger | kernel file + error output | Full compiler stderr passed in prompt |
| Writer/Debugger → Tester | compiled kernel file | File must exist and compile successfully |
| Tester → next phase or Optimizer | tested kernel file | Phase tests pass before proceeding |
| Optimizer → Report | optimized kernel file | Kernel compiles and all tests still pass after optimization |

---

## Key Paths

| Path | Purpose |
|------|---------|
| `tt_llk_blackhole/` | Blackhole LLK (reference architecture) |
| `tt_llk_quasar/` | Quasar LLK (target architecture) |
| `tt_llk_{arch}/instructions/assembly.yaml` | ISA definition (cross-check, use grep — large file) |
| `codegen/references/common-errors.md` | Known error patterns for debugging |

## Commands

```bash
# Compilation check (syntax/type errors)
cd codegen
source ../tests/.venv/bin/activate
PYTHONPATH=.. python scripts/check_compile.py {path_to_kernel} -v

# Functional tests (correctness validation) — ALWAYS use flock wrapper for simulator exclusivity
flock --timeout 900 /tmp/tt-llk-test-simulator.lock bash -c '
  STALE=$(lsof -ti :5556 2>/dev/null || true)
  [ -n "$STALE" ] && echo "Killing stale port 5556 processes: $STALE" && echo "$STALE" | xargs kill -9 2>/dev/null || true
  pkill -9 -f "tt-exalens.*--port=5556" 2>/dev/null || true
  sleep 1
  source ../tests/.venv/bin/activate
  cd ../tests/python_tests/quasar
  TT_UMD_SIMULATOR_PATH=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-1x3 CHIP_ARCH=quasar pytest -x --run-simulator --port=5556 test_{kernel_name}_quasar.py
'

# List available tests
ls ../tests/python_tests/quasar/test_*_quasar.py
```
