---
name: tester
description: Validate an LLK issue fix using the selected backend: local or ttsim.
tools: Bash, Read, Write, Glob, Grep
---

# LLK Issue Tester

Run the fix plan's tt-llk Python tests on the selected backend and report the
result without editing code. This suite covers Layer-1 kernels; the
orchestrator handles changes that are not verifiable in this suite.

## Core Rules

- `TEST_BACKEND` is an operator choice, not a hint.
- Run all in-scope architectures sequentially in one multi-arch session and
  one self-log.
- For `TEST_BACKEND=local`, call `.claude/scripts/run_test.sh` directly. Do not
  invoke pytest.
- For `TEST_BACKEND=ttsim`, run selected pytest tests directly with the
  in-process simulator library. Do not use local, RTL-simulator, or
  compile-only flows.
- Set `TT_METAL_SIMULATOR`, `TT_METAL_DISABLE_SFPLOADMACRO`, and `CHIP_ARCH`
  inside every arch-specific ttsim command.
- Do not debug failures or edit files.
- Do not mark environment failures as compile-only success.
- Do not invoke the standalone `.claude` run-test skill or
  `llk-test-runner` agent; this pipeline tester owns execution.

## State

The spawn prompt provides `WORKTREE_DIR`. Resolve both state stores directly:

```bash
WT="$WORKTREE_DIR"
LOG_DIR="$(python codegen/scripts/state.py --worktree-dir "$WT" get LOG_DIR)"
sg() { python codegen/scripts/state.py --log-dir "$LOG_DIR" get "$1"; }
bg() { python codegen/scripts/state.py --worktree-dir "$WT" get "$1"; }
```

Read `ISSUE_NUMBER`, `RUN_MODE`, `TARGET_ARCH` or `TARGET_ARCHES_JSON`,
and `TEST_BACKEND` with `sg`. Derive the artifacts as
`codegen/artifacts/issue_<ISSUE_NUMBER>_analysis.md` and
`codegen/artifacts/issue_<ISSUE_NUMBER>_fix_plan.md`.

The router leaves simulator paths in bootstrap state. For ttsim, read
`TTSIM_SO_PATH` (single) or `TTSIM_SO_PATHS` (multi) with `bg`.

## Pre-Flight

```bash
cd "$WORKTREE_DIR/tt_metal/tt-llk"
mkdir -p "$LOG_DIR"
```

Read:

1. `.claude/CLAUDE.md`
2. the analysis artifact's `arch_scope`
3. the fix plan's `## Test Strategy`

Parse `TARGET_ARCHES_JSON` as JSON for multi-arch runs; otherwise use
`TARGET_ARCH`. Run only architectures marked `in_scope`. Preserve the
orchestrator's existing `SKIPPED` result for architectures marked
`out_of_scope`.

Normalize selectors relative to the pytest directory:

- `TEST_FILE` is the basename, such as `test_x.py`.
- `TEST_ID` retains `test_x.py::...` but drops a leading
  `tests/python_tests/` or `tests/python_tests/quasar/`.
- Keep repository-relative paths only in compile commands.

## Test Selection

Use the plan's test strategy:

| Plan item | Action |
|---|---|
| compile check only | local only: runner `compile` or the listed command |
| reproduction test | run first |
| regression test | run only after all reproduction tests pass |
| `-k` filter | pass the same filter |
| pytest id | pass as `TEST_ID` |
| no relevant functional test and `compile_only_ok: true` | report `COMPILED_ONLY` after compile check passes |

For ttsim, ignore `compile_checks` and run the listed reproduction/regression
pytest through the ttsim command below.

For each in-scope architecture, select tests whose `arch` is that architecture
or `all`. If none apply, return `ENV_ERROR` for an incomplete test strategy.
The only exception is a local compile-only plan that explicitly sets
`compile_only_ok: true`.

Use exact pytest IDs or narrow `-k` filters. Do not count unrelated
parametrizations as validation.

## Result Recording

Keep raw command output append-only in `run.log`/`compile.log`, readable
attempt history in `agent_tester.md`, and dashboard metrics in `run.json`.

For a single-arch run, patch the final counts into `run.json`:

```bash
python codegen/scripts/run_json_writer.py metric \
  --log-dir "$LOG_DIR" \
  --patch-json "{\"tests_total\":${tests_total},\"tests_passed\":${tests_passed}}"
```

For a multi-arch run, update the same `run.json` as each in-scope architecture
starts and ends. Use the architecture's one-based position in
`TARGET_ARCHES_JSON` as `phase_index`:

```bash
python codegen/scripts/run_json_writer.py message \
  --log-dir "$LOG_DIR" \
  --message "Testing ${arch} with ${TEST_BACKEND}"

python codegen/scripts/run_json_writer.py phase-start \
  --log-dir "$LOG_DIR" \
  --phase "$phase_index" \
  --name "Test ${arch}"
```

After each architecture completes, patch its result and the aggregate counts.
JSON-encode `obstacle_json`; never interpolate raw failure text into JSON. Do
not create per-architecture sibling `run.json` files.

```bash
python codegen/scripts/run_json_writer.py metric \
  --log-dir "$LOG_DIR" \
  --patch-json "{\"arch_results\":{\"${arch}\":{\"status\":\"done\",\"verdict\":\"${verdict}\",\"tests_total\":${tests_total},\"tests_passed\":${tests_passed},\"obstacle\":${obstacle_json}}},\"tests_total\":${aggregate_total},\"tests_passed\":${aggregate_passed}}"

python codegen/scripts/run_json_writer.py phase-end \
  --log-dir "$LOG_DIR" \
  --phase "$phase_index" \
  --test-result "$phase_result" \
  --test-details "$test_details"
```

Map `SUCCESS` and `COMPILED_ONLY` to `phase_result=passed`; map test,
compilation, simulator, and environment failures to `failed`.
`phase-end` increments `phases_completed`, so do not call it again for a phase
already recorded as passed. Retests must still replace `arch_results` and
append their raw and self-log evidence.

## Local Backend

Use `subcommand=run` for reproduction and regression tests. Use
`subcommand=compile` only for an allowed compile-only route:

```bash
bash .claude/scripts/run_test.sh "$subcommand" \
  --worktree "$WORKTREE_DIR/tt_metal/tt-llk" \
  --arch "$arch" \
  --test "$TEST_FILE" \
  --log-dir "$LOG_DIR" \
  --verbose
```

Add optional arguments from the plan:

`--k "$K_FILTER"`, `--test-id "$TEST_ID"`, or `--no-split`.

The wrapper appends raw output to the supplied log directory. Record the exact
invocation and final verdict marker in the self-log.

Local runner exit code mapping:

| Exit | Verdict |
|---|---|
| 0 | `SUCCESS` |
| 1 | `TESTS_FAILED` |
| 2 | `COMPILE_FAILED` |
| 3 | `ENV_ERROR` |
| 4 | `ENV_ERROR` |
| 5 | `TESTS_FAILED` with hang evidence |

## ttsim Backend

Resolve and validate the simulator once per architecture:

```bash
if [ "$(sg RUN_MODE)" = multi ]; then
  TTSIM_SO_PATH="$(python -c \
    'import json,sys; print(json.loads(sys.argv[1]).get(sys.argv[2], ""))' \
    "$(bg TTSIM_SO_PATHS)" "$arch")"
else
  TTSIM_SO_PATH="$(bg TTSIM_SO_PATH)"
fi
case "$TTSIM_SO_PATH" in
  "~/"*) SIM_SO="$HOME/${TTSIM_SO_PATH#\~/}" ;;
  *) SIM_SO="$TTSIM_SO_PATH" ;;
esac

if [ -z "$SIM_SO" ] || [ ! -f "$SIM_SO" ]; then
  echo "ENV_ERROR: missing ttsim library for $arch: ${SIM_SO:-<empty>}" |
    tee -a "$LOG_DIR/run.log"
  exit 3
fi

if [ ! -f "$(dirname "$SIM_SO")/soc_descriptor.yaml" ]; then
  echo "ENV_ERROR: no soc_descriptor.yaml beside $SIM_SO" |
    tee -a "$LOG_DIR/run.log"
  exit 3
fi
```

If validation fails, record `ENV_ERROR` for that architecture and continue
with any remaining architectures. Do not send environment failures to the
worker.

Run each selected test with the validated `SIM_SO`:

```bash
set -o pipefail
[ "$arch" = quasar ] && test_dir=tests/python_tests/quasar || test_dir=tests/python_tests
[ "$arch" = quasar ] && timeout="${TIMEOUT:-1200}" || timeout="${TIMEOUT:-600}"
cd "$WORKTREE_DIR/tt_metal/tt-llk/$test_dir"

pytest_args=(-x --run-simulator "--timeout=$timeout")
[ -n "${K_FILTER:-}" ] && [ -z "${TEST_ID:-}" ] &&
  pytest_args+=(-k "$K_FILTER")
pytest_args+=("${TEST_ID:-$TEST_FILE}")

printf '\n[tester] backend=ttsim arch=%s test=%s\n' \
  "$arch" "${TEST_ID:-$TEST_FILE}" | tee -a "$LOG_DIR/run.log"

env \
  TT_METAL_SIMULATOR="$SIM_SO" \
  TT_METAL_DISABLE_SFPLOADMACRO=1 \
  CHIP_ARCH="$arch" \
  pytest "${pytest_args[@]}" 2>&1 | tee -a "$LOG_DIR/run.log"
pytest_exit=${PIPESTATUS[0]}
echo "PYTEST_EXIT=$pytest_exit" | tee -a "$LOG_DIR/run.log"
exit "$pytest_exit"
```

`TEST_ID` takes precedence over `TEST_FILE` and `K_FILTER`. A missing
simulator path affects only the current architecture.

## Outcome Reading

Start with the final verdict marker for local runs:

```text
=== RUN_LLK_TESTS_VERDICT === ...
```

For ttsim runs, classify the most specific output evidence before applying the
generic pytest exit code:

| Evidence | Verdict |
|---|---|
| exit 0 and tests passed | `SUCCESS` |
| `UnimplementedFunctionality:` | `SIM_ISA_GAP` |
| `UnpredictableValueUsed`, `UndefinedBehavior`, or `NonContractualBehavior` | `TESTS_FAILED` with typed ttsim evidence |
| compiler/build error | `COMPILE_FAILED` |
| assertion/data mismatch/timeout/hang | `TESTS_FAILED` |
| pytest exit 5 / no tests selected | `ENV_ERROR` |
| missing/invalid `TTSIM_SO_PATH`, unusable ttsim install, bad runner invocation, missing environment | `ENV_ERROR` |
| local compile check passed, no functional test exists, and the plan allows compile-only | `COMPILED_ONLY` |

`SIM_ISA_GAP` is not an LLK bug. Record the opcode or function and affected
test; do not send it to the worker.

## Result

Return `TEST_RESULT` with the backend, each requested architecture's verdict,
counts and first evidence, plus the raw- and self-log paths. Include
analyzer-owned `SKIPPED` results; do not calculate a separate combined verdict.

## Self-Log

Create `${LOG_DIR}/agent_tester.md`, or append
`## Test Attempt — <UTC timestamp>` when it exists; never discard earlier
attempts. Record backend and scope, planned tests and normalized selectors,
exact commands, simulator path where applicable, exit codes, verdict markers,
counts, first failure per architecture, and deviations from the plan.

If `LOG_DIR` is empty, report that self-logging was skipped.
