# Fallback execution routes — LLK tester

Reference for `tester.md` (suite `llk`). Read this **only** when the sealed dispatch path in that
playbook does not apply, that is when any of these hold:

- `HW_TEST_DISPATCH_CMD` is unset, or its `--help` does not advertise
  `--requirement-id`;
- `TEST_BACKEND=ttsim`;
- the architecture is `quasar`, which never uses the silicon queue;
- local silicon execution without a dispatcher.

For Blackhole or Wormhole with a sealed dispatcher, the playbook's sealed path
is authoritative and this file is not needed. Every coverage, scope, manifest,
identity and verdict rule in the playbook still applies to these routes.

---

## Local Compile and Execution

For a compile-only plan, use `subcommand=compile` and return
`COMPILED_ONLY` after it passes.

For a functional test:

- for Quasar, run `subcommand=compile` as the local gate and then follow
  **Local Quasar Aether**. Never submit Quasar to the silicon queue;
- for Blackhole/Wormhole with `HW_TEST_DISPATCH_CMD`, run
  `subcommand=compile` as the early gate and then follow **Queued Silicon**;
- without it, use `subcommand=run` so the wrapper compiles and runs on the
  local device.

Create one result path per sealed leaf before either local or queued execution:

```bash
mkdir -p "$LOG_DIR/verification-results/${CODEGEN_ATTEMPT_ID}"
RESULT_JSON_OUT="$LOG_DIR/verification-results/${CODEGEN_ATTEMPT_ID}/${CODEGEN_REQUIREMENT_ID}.json"
```

```bash
bash .claude/scripts/run_test.sh "$subcommand" \
  --worktree "$WORKTREE_DIR/tt_metal/tt-llk" \
  --arch "$arch" \
  --test "$TEST_FILE" \
  --log-dir "$LOG_DIR" \
  --result-json-out "$RESULT_JSON_OUT" \
  --verbose
```

Add optional arguments from the plan:

`--k "$K_FILTER"`, `--test-id "$TEST_ID"`, `--maxfail "$MAXFAIL"`, or `--no-split`.

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

Specific evidence overrides the generic exit mapping: a missing selector or
zero selected tests is `TESTS_FAILED` with `MISSING_TEST_COVERAGE`, even when
the wrapper reports an environment-style exit.

Do not submit a queue job after a local compile failure.

## Local Quasar Aether

Use this route for `arch=quasar` with `TEST_BACKEND=local`, whether or not
`HW_TEST_DISPATCH_CMD` is set. The compute runner owns the patched worktree and
the compile artifacts, so Quasar VCS/emulator execution stays on that same
machine. Blackhole/Wormhole remain queue-backed.

After the local `compile` gate passes, run:

```bash
set +e
bash .claude/scripts/run_test.sh simulate \
  --worktree "$WORKTREE_DIR/tt_metal/tt-llk" \
  --arch quasar \
  --test "$TEST_FILE" \
  --log-dir "$LOG_DIR" \
  --result-json-out "$RESULT_JSON_OUT" \
  --verbose
qsr_exit=$?
set -e
```

Add the plan's `--k "$K_FILTER"` or `--test-id "$TEST_ID"` selector. The
wrapper resolves `QSR_SIM_BACKEND=emu|vcs` to the corresponding configured UMD
path and uses `QSR_AETHER_LOCK`, which must be a shared-filesystem path so the
two compute hosts cannot start or reap each other's Aether jobs.

If the test requires `--no-split`, skip the separate compile/simulate pair and
use `run --no-split` once with the same arguments. It compiles and executes
while holding the shared Aether lock.

Classify the final `RUN_LLK_TESTS_VERDICT` with the local exit-code table above.
Record the selected `QSR_SIM_BACKEND` and no queue job ID.

## Queued Silicon

Use this route only for Blackhole/Wormhole with `TEST_BACKEND=local` and
`HW_TEST_DISPATCH_CMD`, after the corresponding local compile passes. The queue
repeats the producer in an isolated warm workspace, then owns card scheduling
and silicon execution; do not call the local wrapper's `run` or `simulate`
subcommands. Quasar is never valid on this route. Dispatch captures tracked,
modified, deleted, and untracked worktree files in the submitted binary diff.

The queue accepts the same exact pytest node or `-k` selector sealed in the
manifest. Pass `--k "$K_FILTER"` when present; never silently run a broader
test. The queue always uses split producer/consumer execution, so reject a test
that specifically requires `--no-split`.

Construct the selector relative to `tests/python_tests`, which differs from
the wrapper's arch-relative selector:

```bash
QUEUE_TEST="${TEST_ID:-$TEST_FILE}"
[ "$arch" = quasar ] && QUEUE_TEST="quasar/$QUEUE_TEST"
selector_args=()
[ -n "$K_FILTER" ] && selector_args+=(--k "$K_FILTER")
result_args=()
if [ "${CODEGEN_RUNNER_POOL:-prod}" = audit ]; then
  result_args+=(--result-json-out "$RESULT_JSON_OUT")
fi

set +e
$HW_TEST_DISPATCH_CMD --kind llk --arch "$arch" \
  --test "$QUEUE_TEST" \
  "${selector_args[@]}" \
  --worktree "$WORKTREE_DIR" \
  --base "$(sg GIT_COMMIT)" \
  --session "${HW_TEST_SESSION:-issue-${ISSUE_NUMBER}}" \
  "${result_args[@]}" \
  --timeout "${TIMEOUT:-1800}" 2>&1 | tee -a "$LOG_DIR/run.log"
dispatch_exit=${PIPESTATUS[0]}
set -e
```

Require one final `HW_TEST_RESULT arch=<arch>` marker and record its `job`
value. Treat `failure_stage=build` as `ENV_ERROR`: because the local compile
passed, the queue runner failed to reproduce it. No structured execution result
is expected because the producer correctly did not release its artifacts to a
card. Any other `ran=false` marker is also `ENV_ERROR`.

Record both compile durations in the self-log: the local wrapper duration and,
when the returned job has `build_started_at` and `built_at`, the queue producer
duration.

For an audit run that reached execution, also require `RESULT_JSON_OUT` to
contain the exact protocol-v2 result copied by dispatch. Validate its schema,
run, attempt, requirement, architecture, backend, and selector against the
sealed manifest; then derive the compatibility suite summary from its structured
evidence:

- `classification=success` -> `SUCCESS`;
- `candidate_failure|coverage_error` -> `TESTS_FAILED` for compatibility; retain
  the distinct classification, per-leaf reasons and receipt ID. The retry helper
  uses these fields; selected skips do not prove numerical failure or exemption;
- `infra_error|timed_out`, a missing/invalid result, or an identity mismatch ->
  `ENV_ERROR`.

Set `tests_total=collection.selected` and
`tests_passed=execution.passed`. Preserve skipped, xfailed, and xpassed counts
in the self-log; do not convert them into passes. The strict reducer rereads
the same result file and is authoritative for an audit verdict. The marker and
dispatch exit are correlation/supporting evidence only.

For production compatibility, no protocol-v2 result copy is requested and the
legacy marker remains authoritative:

| Marker | Verdict |
|---|---|
| `ok=true ran=true passed=true` | `SUCCESS` |
| `ok=false ran=true` | `TESTS_FAILED` |
| `failure_stage=build ran=false` | `ENV_ERROR` |
| missing, malformed, or `ran=false` | `ENV_ERROR` |

Legacy queue markers do not provide exact counts. Record zero counts with an
explicit obstacle instead of inventing them, and always retain the job ID so
the detailed queue result can be inspected.

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
