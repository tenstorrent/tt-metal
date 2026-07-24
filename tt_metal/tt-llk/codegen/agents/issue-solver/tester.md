---
name: tester
description: Validate an LLK issue fix using the selected backend: local or ttsim.
tools: Bash, Read, Write, Glob, Grep
---

# LLK Issue Tester

Run the fix plan's tt-llk Python tests on the selected backend and report the
result without editing code. This suite covers Layer-1 kernels. Report
`UNVERIFIABLE_IN_LLK_SUITE` when no test reaches the change; reserve `SKIPPED`
for an architecture that the analysis marks out of scope.

## Core Rules

- Read `.claude/skills/run-test/SKILL.md` and `.claude/agents/llk-test-runner.md` only before running local tests.
- `TEST_BACKEND` is an operator choice, not a hint.
- Run all target architectures sequentially in one multi-arch session and one
  self-log.
- For `TEST_BACKEND=local`, use `.claude/scripts/run_test.sh`. Do not invoke pytest directly.
- For `TEST_BACKEND=ttsim`, run selected pytest tests directly with the
  in-process simulator library. Do not use local, RTL-simulator, or
  compile-only flows.
- Set `TT_METAL_SIMULATOR`, `TT_METAL_DISABLE_SFPLOADMACRO`, and `CHIP_ARCH`
  inside every arch-specific ttsim command.
- Do not debug failures or edit files.
- Do not mark environment failures as compile-only success.

## State

The spawn prompt provides `WORKTREE_DIR`. Resolve both state stores from
`<worktree>/tt_metal/tt-llk`:

```bash
WT="$(cd ../.. && pwd)"
LOG_DIR="$(python codegen/scripts/state.py --worktree-dir "$WT" get LOG_DIR)"
sg() { python codegen/scripts/state.py --log-dir "$LOG_DIR" get "$1"; }
bg() { python codegen/scripts/state.py --worktree-dir "$WT" get "$1"; }
```

Read `ISSUE_NUMBER`, `RUN_MODE`, `TARGET_ARCH` or `TARGET_ARCHES_JSON`,
`TEST_BACKEND`, `CHANGED_FILES`, `WORKTREE_DIR`, and `LOG_DIR` with `sg`.
Derive the fix plan as
`codegen/artifacts/issue_<ISSUE_NUMBER>_fix_plan.md`.

The router leaves simulator paths in bootstrap state. For ttsim, read
`TTSIM_SO_PATH` (single) or `TTSIM_SO_PATHS` (multi) with `bg`.

## Mandatory Pre-Flight

```bash
cd "$WORKTREE_DIR/tt_metal/tt-llk"
mkdir -p "$LOG_DIR"
```

Read:

1. `.claude/CLAUDE.md`
2. the fix plan's `## Test Strategy`
3. `.claude/skills/run-test/SKILL.md` only when `TEST_BACKEND=local`
4. `.claude/agents/llk-test-runner.md` only when `TEST_BACKEND=local`
5. `tests/TTSIM.md` only when `TEST_BACKEND=ttsim`

Parse `TARGET_ARCHES_JSON` as JSON for multi-arch runs. Otherwise run
`TARGET_ARCH`.

Normalize test names before running. If the plan gives `tests/python_tests/test_x.py` or `tests/python_tests/quasar/test_x.py`, set `TEST_FILE=test_x.py`. Keep the full path only for source/compile checks. Keep the full pytest id only in `TEST_ID`.

## Subcommand Selection

Use the plan's test strategy:

| Plan item | Action |
|---|---|
| compile check only | local backend only: local runner `compile`, or listed compiler command |
| reproduction test | run first |
| regression test | run after reproduction passes |
| `-k` filter | pass the same filter |
| pytest id | pass as `TEST_ID` |
| no relevant functional test and `compile_only_ok: true` | report `COMPILED_ONLY` after compile check passes |

For ttsim, ignore `compile_checks` and run the listed reproduction/regression
pytest through the ttsim command below.

For multi-arch plans, choose tests whose `arch` is the current arch or `all`. If a listed test is clearly specific to another arch, skip it for the current arch and explain that in the self-log. If no test is listed for an arch, mark that arch `SKIPPED` only when the fix plan explicitly explains why no validation applies to that arch; otherwise return `ENV_ERROR` with the missing test strategy as the obstacle.

Use exact pytest IDs or narrow `-k` filters. Do not count unrelated
parametrizations as validation.

## Multi-Arch Dashboard Updates

For multi-arch runs, update the single run as each architecture starts and
ends:

```bash
python codegen/scripts/run_json_writer.py message \
  --log-dir "$LOG_DIR" \
  --message "Testing ${arch} with ${TEST_BACKEND}"

python codegen/scripts/run_json_writer.py phase-start \
  --log-dir "$LOG_DIR" \
  --phase "$phase_index" \
  --name "Test ${arch}"
```

After each arch completes, patch `arch_results`, `tests_total`, and `tests_passed` with `run_json_writer.py metric`. Do not create per-arch sibling `run.json` files.

`metric` accepts `--patch-json` only. Patch nested fields as a nested JSON
object, not `--key`/`--value` pairs:

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

## Local Backend

For each selected target arch, set `arch` to the current arch and use the shared runner:

```bash
bash .claude/scripts/run_test.sh run \
  --worktree "$WORKTREE_DIR/tt_metal/tt-llk" \
  --arch "$arch" \
  --test "$TEST_FILE" \
  --log-dir "$LOG_DIR" \
  --verbose
```

Add optional arguments from the plan:

```bash
--k "$K_FILTER"
--test-id "$TEST_ID"
--maxfail "$MAXFAIL"
--no-split
```

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

Run one Bash command per test. For a multi-arch run, parse
`TTSIM_SO_PATHS` as JSON and set `TTSIM_SO_PATH` for the current architecture
before executing this template:

```bash
TTSIM_SO_PATH="$(
  python - "$(bg TTSIM_SO_PATHS)" "$CURRENT_ARCH" <<'PY'
import json
import sys

paths = json.loads(sys.argv[1])
print(paths.get(sys.argv[2], ""))
PY
)"
```

For a single-arch run, use `TTSIM_SO_PATH="$(bg TTSIM_SO_PATH)"`.

```bash
set -euo pipefail
cd "$WORKTREE_DIR/tt_metal/tt-llk"
mkdir -p "$LOG_DIR"

arch="${CURRENT_ARCH:-${TARGET_ARCH:-}}"

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

[ "$arch" = quasar ] && cd tests/python_tests/quasar || cd tests/python_tests
[ -n "${TIMEOUT:-}" ] || { [ "$arch" = quasar ] && TIMEOUT=1200 || TIMEOUT=600; }

pytest_args=(-x --run-simulator "--timeout=$TIMEOUT")
[ -n "${K_FILTER:-}" ] && [ -z "${TEST_ID:-}" ] &&
  pytest_args+=(-k "$K_FILTER")
pytest_args+=("${TEST_ID:-$TEST_FILE}")

set +e
env \
  TT_METAL_SIMULATOR="$SIM_SO" \
  TT_METAL_DISABLE_SFPLOADMACRO=1 \
  CHIP_ARCH="$arch" \
  pytest "${pytest_args[@]}" 2>&1 | tee -a "$LOG_DIR/run.log"
pytest_exit=${PIPESTATUS[0]}
set -e
echo "PYTEST_EXIT=$pytest_exit" | tee -a "$LOG_DIR/run.log"
exit "$pytest_exit"
```

`TEST_ID` takes precedence over `TEST_FILE` and `K_FILTER`. A missing
simulator path affects only the current architecture; record it as
`ENV_ERROR` and let the orchestrator request a corrected path.

## Outcome Reading

Start with the final verdict marker for local runs:

```text
=== RUN_LLK_TESTS_VERDICT === ...
```

For ttsim runs, classify from the pytest exit code and output.

| Evidence | Verdict |
|---|---|
| tests pass | `SUCCESS` |
| compiler/build error | `COMPILE_FAILED` |
| assertion/data mismatch/timeout/hang | `TESTS_FAILED` |
| `UnimplementedFunctionality:` from ttsim | `SIM_ISA_GAP` |
| `UnpredictableValueUsed`, `UndefinedBehavior`, or `NonContractualBehavior` from ttsim | `TESTS_FAILED` with typed ttsim evidence |
| pytest exit 5 / no tests selected | `ENV_ERROR` |
| missing/invalid `TTSIM_SO_PATH`, unusable ttsim install, bad runner invocation, missing environment | `ENV_ERROR` |
| local compile check passed, no functional test exists, and the plan allows compile-only | `COMPILED_ONLY` |

`SIM_ISA_GAP` is not an LLK bug. Report the opcode/function and test, then stop.

## Output Format

For multi-arch runs, include one block per arch and a final `arch_results` summary:

```text
MULTI_ARCH_TEST_RESULT - issue #<number> (<backend>)
arch_results:
  wormhole:
    verdict: SUCCESS|COMPILE_FAILED|TESTS_FAILED|SIM_ISA_GAP|ENV_ERROR|COMPILED_ONLY|SKIPPED
    tests_total: N
    tests_passed: N
    first_evidence: ...
  blackhole:
    verdict: ...
combined_verdict: SUCCESS|COMPILE_FAILED|TESTS_FAILED|SIM_ISA_GAP|ENV_ERROR|COMPILED_ONLY
```

`combined_verdict` is a human-readable roll-up only. The orchestrator does **not**
consume it — it reads per-arch `arch_results` and derives its own authoritative
`combined_status` (`success`/`partial`/`failed`/`skipped`) in Step 6.

```text
PASS - issue #<number> (<backend>, <arch>)
- Compilation: PASSED|NOT_RUN
- Tests total: N
- Tests passed: N
- Commands:
  - ...
```

```text
FAIL - issue #<number> (<backend>, <arch>)
- Verdict: COMPILE_FAILED|TESTS_FAILED|SIM_ISA_GAP|ENV_ERROR
- Tests total: N
- Tests passed: N
- First evidence: ...
- Commands:
  - ...
```

For a successful compile-only route, return:

```text
COMPILED_ONLY - issue #<number> (local, <arch>)
- Compilation: PASSED
- Reason: ...
```

## Limits

Run at most 10 test invocations in one tester session across all arches. If more are needed, return `TESTS_FAILED` with the reason.

## Self-Log

Before returning, write `${LOG_DIR}/agent_tester.md` with backend, exact
commands, selectors, exit codes, counts, verdicts, and the first meaningful
failure. If `LOG_DIR` is empty, report that self-logging was skipped.
