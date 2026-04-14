# Metrics and Logging System

The codegen system includes a structured metrics and logging system for tracking runs, debugging failures, and benchmarking quality over time.

---

## LOG_DIR: Per-Run Log Directory

Each codegen run creates a unique log directory:

```
/proj_sw/user_dev/llk_code_gen/quasar/{RUN_ID}/
├── instructions/              # Snapshot of agent playbooks used
│   ├── llk-analyzer.md
│   ├── llk-planner.md
│   ├── llk-kernel-writer.md
│   ├── llk-debugger.md
│   ├── llk-phase-tester.md
│   ├── llk-regression-tester.md
│   └── llk-prettifier.md
├── run.json                   # This run's metrics (pretty-printed, same as runs.jsonl entry)
├── ckernel_sfpu_{op}.h        # Generated kernel snapshot (copy from tt_llk_{arch}/)
├── ref_ckernel_sfpu_{op}.h   # Reference kernel snapshot (copy from tt_llk_{ref_arch}/)
├── sfpu_{op}_{arch}_test.cpp # C++ test snapshot (if tests exist)
├── test_{op}_{arch}.py       # Python test snapshot (if tests exist)
├── {op}_report.md             # Final report (copy from codegen/artifacts/)
├── {op}_arch_research.md      # Architecture research (copy from codegen/artifacts/)
├── {op}_analysis.md           # Reference analysis (copy from codegen/artifacts/)
├── {op}_phase{N}_spec.md      # Phase specs (copy from codegen/artifacts/)
├── agent_arch_lookup.md       # Arch research agent's reasoning log
├── agent_analyzer.md          # Analyzer's reasoning log
├── agent_planner.md           # Planner's reasoning log
├── agent_writer.md            # Writer's reasoning log
├── agent_debugger.md          # Debugger's reasoning log (if invoked)
├── agent_test_writer.md       # Test writer's reasoning log (if tests created)
├── agent_tester.md            # Tester's reasoning log
└── agent_prettifier.md        # Prettifier's reasoning log
```

**Every LOG_DIR must be self-contained** — all artifacts, agent logs, and a `run.json` summary so you can understand a run from the LOG_DIR alone without needing to cross-reference `codegen/artifacts/`.

**RUN_ID format**: `{YYYY-MM-DD}_{kernel}_{arch}_{random_hex}`

The `LOG_DIR` is passed to every agent prompt. Agents write their reasoning to `{LOG_DIR}/agent_{name}.md`.

---

## runs.jsonl: Cumulative Run Metrics

After each run completes, the orchestrator appends a JSONL entry to:
```
/proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl
```

### Entry Format

```json
{
  "kernel": "gelu",
  "kernel_type": "sfpu",
  "arch": "quasar",
  "reference_arch": "blackhole",
  "reference_file": "tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_gelu.h",
  "generated_file": "tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_gelu.h",
  "start_time": "2026-03-24T14:30:00Z",
  "end_time": "2026-03-24T14:45:12Z",
  "phases_total": 2,
  "phases_completed": 2,
  "compilation_attempts": 4,
  "debug_cycles": 1,
  "tests_total": 12,
  "tests_passed": 12,
  "lines_generated": 187,
  "tests_generated": false,
  "prettified": true,
  "status": "success",
  "obstacle": null,
  "per_phase": [
    {"phase": 1, "name": "basic", "compilation_attempts": 1, "debug_cycles": 0, "test_result": "passed", "compile_errors": [], "test_details": null},
    {"phase": 2, "name": "derivative", "compilation_attempts": 3, "debug_cycles": 1, "test_result": "passed", "compile_errors": [{"attempt": 1, "error": "unknown type 'vFloat'"}, {"attempt": 2, "error": "use of undeclared identifier 'dst_reg'"}], "test_details": "12/12 passed"}
  ],
  "prompt": "Generate gelu for Quasar",
  "batch_id": "2026-03-28_weekly",
  "tokens": {
    "input": 125000,
    "output": 48000,
    "cache_read": 80000,
    "total": 173000
  },
  "model": "opus",
  "run_type": "ci",
  "agents": ["analyzer", "planner", "writer", "tester", "debugger"],
  "run_id": "2026-03-24_gelu_quasar_a1b2c3d4",
  "log_dir": "logs/2026-03-24_gelu_quasar_a1b2c3d4"
}
```

**Written as a single JSONL line** (expanded above for readability).

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `kernel` | string | Kernel name (e.g., "gelu", "reduce") |
| `kernel_type` | string | Category: "sfpu", "math", "pack", or "unpack" |
| `arch` | string | Target architecture |
| `reference_arch` | string | Reference architecture ported from |
| `reference_file` | string | Path to source reference file |
| `generated_file` | string | Path to generated output file |
| `start_time` | string | ISO 8601 UTC timestamp when run started |
| `end_time` | string | ISO 8601 UTC timestamp when run finished |
| `phases_total` | number | Number of sub-kernel phases identified |
| `phases_completed` | number | Number of phases that passed |
| `compilation_attempts` | number | Total compile invocations across all phases |
| `debug_cycles` | number | Total debug agent invocations across all phases |
| `tests_total` | number | Total test cases run |
| `tests_passed` | number | Total test cases passed |
| `lines_generated` | number | Line count of generated file |
| `tests_generated` | boolean | Whether tests were created by the test-writer agent (`true`) or pre-existing tests were used (`false`) |
| `prettified` | boolean | Whether prettification succeeded |
| `status` | string | `"success"` (compiles + tests pass), `"compiled"` (compiles but tests failed/skipped/unavailable), or `"failed"` (does not compile) |
| `obstacle` | string/null | Main blocker if failed/compiled, null if success |
| `failures` | array | All failures encountered during the run (see Failures Entry below). Empty array `[]` if no failures |
| `per_phase` | array | Per-phase breakdown (see below) |
| `prompt` | string | Original prompt used to launch the run (e.g., "Generate gelu for Quasar") |
| `batch_id` | string/null | Identifier grouping runs from a single batch session (null if manual run) |
| `tokens` | object | Token usage from the Claude CLI (see below) |
| `model` | string | Claude model used for the run (e.g., "opus", "sonnet", "haiku") |
| `run_type` | string | `"ci"` (scheduled batch run) or `"manual"` (interactive session) |
| `agents` | array | List of agents that were invoked |
| `run_id` | string | Unique run identifier |
| `git_commit` | string | Git commit hash (`git rev-parse HEAD`) at run start. `"unknown"` if not in a git repo |
| `log_dir` | string | Relative path to LOG_DIR |

### Per-Phase Entry

| Field | Type | Description |
|-------|------|-------------|
| `phase` | number | Phase number (1-indexed) |
| `name` | string | Short label for the sub-kernel group |
| `compilation_attempts` | number | Compile invocations for this phase only |
| `debug_cycles` | number | Debug agent invocations for this phase only |
| `test_result` | string | "passed", "failed", or "skipped" |
| `compile_errors` | array | All compilation errors for this phase. Each entry: `{"attempt": N, "error": "message"}`. Empty array if compiled clean on first try |
| `test_details` | string/null | Summary like "12/12 passed" or "failed: test_large_input — mismatch at idx 42". null if skipped |

### Failures Entry

Every failure encountered during a run is logged — both resolved and unresolved. This provides a complete picture of what problems the system hit.

| Field | Type | Description |
|-------|------|-------------|
| `step` | string | Pipeline step where failure occurred (e.g., `"compile_phase_1"`, `"test_phase_2"`, `"analyzer"`, `"final_regression"`) |
| `agent` | string | Agent that was running (`"writer"`, `"debugger"`, `"tester"`, `"analyzer"`, `"planner"`, `"arch_lookup"`) |
| `type` | string | Category: `"compile_error"`, `"test_failure"`, `"agent_error"`, `"infra_error"` |
| `message` | string | The actual error — first meaningful line of compiler stderr, pytest output, or agent error |
| `resolved` | boolean | `true` if fixed during the run, `false` if it blocked completion |

Example:
```json
{
  "failures": [
    {"step": "compile_phase_1", "agent": "writer", "type": "compile_error", "message": "unknown type 'vFloat'", "resolved": true},
    {"step": "compile_phase_1", "agent": "debugger", "type": "compile_error", "message": "use of undeclared identifier 'dst_reg'", "resolved": true},
    {"step": "test_phase_2", "agent": "tester", "type": "infra_error", "message": "simulator timeout after 600s", "resolved": false}
  ]
}
```

### Tokens Entry

| Field | Type | Description |
|-------|------|-------------|
| `input` | number | Total input tokens consumed |
| `output` | number | Total output tokens generated |
| `cache_read` | number | Tokens served from prompt cache |
| `total` | number | `input + output` (cache_read is a subset of input, not additive) |

**How to capture**: Run the orchestrator via `claude -p "..." --output-format json`. The JSON output includes a `usage` object with `input_tokens`, `output_tokens`, and `cache_read_input_tokens`. The batch runner script should parse these and pass them to the orchestrator, or the orchestrator should capture them from the CLI's final output.

---

## Key Metrics for Benchmarking

When building a dashboard from `runs.jsonl`, these are the most useful metrics:

### Quality Metrics
- **Full success rate**: `status == "success"` / total runs — compiles AND tests pass
- **Compilation rate**: `status != "failed"` / total runs — at least compiles (includes "compiled" + "success")
- **First-try compilation rate**: runs where `compilation_attempts == phases_total` (writer got it right every time)
- **Debug overhead**: `compilation_attempts - phases_total` — extra compiles needed beyond the minimum
- **Test pass rate**: `tests_passed / tests_total` (only for runs where `tests_total > 0`)

### Efficiency Metrics
- **Duration**: `end_time - start_time` — wall-clock time per run
- **Compile attempts per phase**: `compilation_attempts / phases_total` — lower is better
- **Debug cycles per phase**: `debug_cycles / phases_total`

### Cost Metrics
- **Tokens per run**: `tokens.total` — overall token consumption
- **Tokens per phase**: `tokens.total / phases_total` — normalized cost
- **Cache hit rate**: `tokens.cache_read / tokens.input` — higher means better prompt caching
- **Output ratio**: `tokens.output / tokens.total` — how much of the budget is generation vs context

### Trending Over Time
- Group by `start_time` (weekly) to see quality trends
- Compare same kernel across runs to measure improvement
- Track `lines_generated` for code bloat trends

### Example Queries
```bash
# Success rate
jq -s '[.[] | .status] | group_by(.) | map({(.[0]): length})' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl

# Average compile attempts by kernel type
jq -s 'group_by(.kernel_type) | map({type: .[0].kernel_type, avg_compiles: ([.[].compilation_attempts] | add / length)})' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl

# Failed runs with obstacles
jq 'select(.status == "failed") | {kernel, obstacle}' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl

# Worst phases (most debug cycles)
jq '.per_phase[] | select(.debug_cycles > 0) | {kernel: input_filename, phase: .name, debug_cycles}' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl

# Token usage per kernel
jq '{kernel, total_tokens: .tokens.total, cache_hit_pct: ((.tokens.cache_read / .tokens.input * 100) | floor)}' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl

# Average tokens by kernel type
jq -s 'group_by(.kernel_type) | map({type: .[0].kernel_type, avg_tokens: ([.[].tokens.total] | add / length | floor)})' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl

# Most expensive runs (top 5)
jq -s 'sort_by(-.tokens.total) | .[0:5] | .[] | {kernel, status, tokens: .tokens.total}' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl

# Runs in a specific batch
jq 'select(.batch_id == "2026-03-28_weekly") | {kernel, status, tokens: .tokens.total}' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl

# All unresolved failures
jq '.failures[]? | select(.resolved == false) | {kernel: input_filename, step, type, message}' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl

# Most common failure types
jq -s '[.[].failures[]?] | group_by(.type) | map({type: .[0].type, count: length}) | sort_by(-.count)' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl

# Failures for a specific kernel
jq 'select(.kernel == "gelu") | {kernel, failures}' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl
```

---

## Agent Self-Logging

Every agent writes a reasoning log to `{LOG_DIR}/agent_{name}.md` during execution. The log includes:

- **Files read** — which files were read and why
- **Key findings** — important patterns, issues, or decisions discovered
- **Decisions made** — what approach was chosen and why
- **Surprises** — anything unexpected or non-obvious

If no `LOG_DIR` is provided, agents skip logging (backward compatible).

---

## Reviewing Logs After a Run

### Quick status check
```bash
# See all runs
cat /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl

# See failed runs
jq 'select(.status == "failed")' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl

# See runs for a specific kernel
jq 'select(.kernel == "gelu")' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl

# See runs for a specific kernel type
jq 'select(.kernel_type == "sfpu")' /proj_sw/user_dev/llk_code_gen/quasar/runs.jsonl
```

### Deep dive into a run
```bash
# List log files for a run
ls /proj_sw/user_dev/llk_code_gen/quasar/{RUN_ID}/

# Read a specific agent's reasoning
cat /proj_sw/user_dev/llk_code_gen/quasar/{RUN_ID}/agent_analyzer.md

# Compare agent playbooks used vs current
diff /proj_sw/user_dev/llk_code_gen/quasar/{RUN_ID}/instructions/llk-planner.md codegen/agents/llk-planner.md
```
