#!/bin/bash
# run_test.sh — centralised LLK test runner for codegen agents.
#
# Encapsulates the two-step compile-then-simulate flow, flock-based simulator
# serialisation, stale-process cleanup, and temp-file lifecycle so agents never
# have to manage any of that themselves.
#
# Usage:
#   run_test.sh <COMMAND> --worktree DIR --arch ARCH --test FILE [OPTIONS]
#
# Commands:
#   count     Count test variants (collection-only; outputs integer to stdout)
#   compile   Run compile-producer step (parallel, no flock needed)
#   simulate  Run the consumer step (flock-serialised, no xdist).
#             Quasar uses the UMD simulator; Blackhole/Wormhole run on silicon.
#   run       compile + simulate in sequence (most common agent use case)
#
# Required:
#   --worktree DIR    Absolute path to the LLK working directory
#                     (contains tests/ and tt_llk_<arch>/ as direct children)
#   --arch     ARCH   Target architecture (quasar, blackhole, ...)
#   --test     FILE   Test file name, e.g. test_sfpu_square_quasar.py
#
# Optional:
#   --maxfail  N      Stop after N failures (simulate/run; omit for verification) (default: 10)
#   --k        EXPR   pytest -k filter expression
#   --test-id  ID     Full parametrize ID for a single variant run
#                     (single-quotes, brackets, commas are safe — no escaping needed)
#   --port     PORT   Simulator port (default: 5556)
#   --timeout  SECS   pytest --timeout ceiling (default: 600)
#   --jobs     N      Compile parallelism (default: 15)
#   --lock     FILE   flock lock file (default: /tmp/tt-llk-test-<arch>.lock)
#                     Per-arch by default so QSR/BH/WH agents only block on
#                     same-arch peers (the simulator or that arch's silicon).
#   --lock-timeout N  Seconds to wait for the lock (default: 900)
#   --sim-path PATH   Override TT_UMD_SIMULATOR_PATH
#                     (default: /proj_sw/user_dev/$USER/tt-umd-simulators/build/emu-<arch>-1x3)
#   --speed-of-light  Pass pytest --speed-of-light (compile-time formats / SOL path).
#   --no-split        Skip compile-producer step; run pytest --run-simulator without
#                     --compile-consumer (combined compile+run in one pytest invocation).
#                     Use for issue-solver tests that don't pre-build ELFs.
#   --log-dir  DIR    DEBUG-ONLY. When set, append combined stdout+stderr from each
#                     phase to <DIR>/compile.log and <DIR>/run.log (created if missing).
#                     Output still streams to the terminal as usual; the file is only
#                     so you can scroll back through long verification runs and see
#                     in-timeline what happened. Default: no log file.
#   --progress        Manual-debug aid: emit a periodic `[progress]` status line to
#                     stderr during the compile and simulate phases (elapsed time;
#                     in simulate also the age of the last output line). Off by
#                     default. Use when a phase is quiet for tens of seconds (Quasar
#                     model load, BH/WH long compile) and you want to know it's
#                     still alive vs. silently stuck.
#   --progress-interval N
#                     Seconds between progress lines (default: 30). Ignored unless
#                     --progress is set.
#   --verbose         Print step headers to stderr
#
# Exit codes:
#   0  All tests passed (or count written to stdout successfully)
#   1  One or more tests failed
#   2  Compile step failed  (only from the 'run' command)
#   3  Environment error (flock timeout, simulator port stuck, venv missing)
#   4  Usage / validation error (missing required options)
#   5  Hang detected by watchdog. Both modes watch the consumer's stdout
#      cadence via a heartbeat file's mtime; if it goes quiet past the per-mode
#      threshold (QSR=120s, silicon=300s) the watchdog kills the consumer tree.
#      Silicon additionally runs tt-triage.py and tt-smi -r.
#      See _watchdog_emu / _watchdog_silicon.
#
# Agents invoke this via the Bash tool with timeout: 1800000 (synchronous,
# never run_in_background). The script blocks until completion and returns one
# of the exit codes above — the agent reads that code and decides the diagnosis
# (PASS / compile error / test failure / ENV_ERROR / HANG) without parsing
# internals. A HANG (exit 5) is the side-channel signal that the device or
# simulator wedged, distinct from a normal test failure (exit 1) — same trick
# as tt-metal's run_safe_pytest.sh uses with tt-triage.
#
# Examples:
#   # Count variants before deciding --maxfail
#   VARIANT_COUNT=$(bash "$WORKTREE_DIR/.claude/scripts/run_test.sh" count \
#       --worktree "$WORKTREE_DIR" --arch quasar --test test_sfpu_square_quasar.py)
#
#   # Compile + simulate, stop after 5 failures
#   bash "$WORKTREE_DIR/.claude/scripts/run_test.sh" run \
#       --worktree "$WORKTREE_DIR" --arch quasar --test test_sfpu_square_quasar.py \
#       --maxfail 5
#   RUN_EXIT=$?
#
#   # Verification run — full matrix, no maxfail
#   bash "$WORKTREE_DIR/.claude/scripts/run_test.sh" run \
#       --worktree "$WORKTREE_DIR" --arch quasar --test test_sfpu_square_quasar.py
#
#   # Single variant by parametrize ID (single-quotes and brackets are safe)
#   bash "$WORKTREE_DIR/.claude/scripts/run_test.sh" simulate \
#       --worktree "$WORKTREE_DIR" --arch quasar --test test_sfpu_square_quasar.py \
#       --test-id "test_sfpu_square_quasar.py::test_sfpu_square_quasar[formats:(..., 'SyncFull', ...)]"

# ── Parse ────────────────────────────────────────────────────────────────────

CMD="${1:-}"
shift 2>/dev/null || true

WORKTREE=""
ARCH=""
TEST_FILE=""
MAXFAIL="10"
K_FILTER=""
TEST_ID=""
PORT="5556"
TIMEOUT="600"
JOBS="15"
LOCKFILE=""  # set in _validate based on ARCH if not user-overridden
LOCK_TIMEOUT="900"
SIM_PATH=""
NO_SPLIT="false"
SPEED_OF_LIGHT="false"
LOG_DIR=""
PROGRESS_ENABLED="false"
PROGRESS_INTERVAL="30"
VERBOSE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worktree)      WORKTREE="$2";      shift 2 ;;
    --arch)          ARCH="$2";          shift 2 ;;
    --test)          TEST_FILE="$2";     shift 2 ;;
    --maxfail)       MAXFAIL="$2";       shift 2 ;;
    --k)             K_FILTER="$2";      shift 2 ;;
    --test-id)       TEST_ID="$2";       shift 2 ;;
    --port)          PORT="$2";          shift 2 ;;
    --timeout)       TIMEOUT="$2";       shift 2 ;;
    --jobs)          JOBS="$2";          shift 2 ;;
    --lock)          LOCKFILE="$2";      shift 2 ;;
    --lock-timeout)  LOCK_TIMEOUT="$2";  shift 2 ;;
    --sim-path)      SIM_PATH="$2";      shift 2 ;;
    --log-dir)       LOG_DIR="$2";       shift 2 ;;
    --progress)      PROGRESS_ENABLED="true"; shift ;;
    --progress-interval) PROGRESS_INTERVAL="$2"; shift 2 ;;
    --speed-of-light) SPEED_OF_LIGHT="true"; shift ;;
    --no-split)      NO_SPLIT="true";    shift   ;;
    --verbose|-v)    VERBOSE="true";     shift   ;;
    --help|-h)
      sed -n 's/^# \{0,1\}//p' "$0" | head -60
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      echo "Run with --help for usage." >&2
      exit 4
      ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

_vlog() {
  if [[ "$VERBOSE" == "true" ]]; then
    echo "[run_llk_tests] $*" >&2
  fi
}

# Validate required args and derive VENV, TEST_DIR, SIM_PATH.
# Sets these as script-global variables; exits on any problem.
_validate() {
  local errors=0
  [[ -z "$WORKTREE"  ]] && { echo "ERROR: --worktree is required" >&2; ((errors++)); }
  [[ -z "$ARCH"      ]] && { echo "ERROR: --arch is required"     >&2; ((errors++)); }
  [[ -z "$TEST_FILE" ]] && { echo "ERROR: --test is required"     >&2; ((errors++)); }
  [[ $errors -gt 0 ]] && exit 4

  VENV="${WORKTREE}/tests/.venv"
  # Test layout is arch-dependent:
  #   quasar              → tests/python_tests/quasar/  (arch-specific *_quasar.py)
  #   blackhole/wormhole  → tests/python_tests/         (cross-arch tests, CHIP_ARCH selects)
  case "$ARCH" in
    quasar)              TEST_DIR="${WORKTREE}/tests/python_tests/quasar" ;;
    blackhole|wormhole)  TEST_DIR="${WORKTREE}/tests/python_tests"        ;;
    *)                   TEST_DIR="${WORKTREE}/tests/python_tests/${ARCH}" ;;
  esac

  if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found: ${VENV}" >&2
    echo "       Run 'CHIP_ARCH=${ARCH} ./setup_testing_env.sh' in ${WORKTREE}/tests/ first." >&2
    exit 3
  fi
  if [[ ! -d "$TEST_DIR" ]]; then
    echo "ERROR: test directory not found: ${TEST_DIR}" >&2
    exit 3
  fi
  if [[ ! -f "${TEST_DIR}/${TEST_FILE}" ]]; then
    echo "ERROR: test file not found: ${TEST_DIR}/${TEST_FILE}" >&2
    echo "       Hint: blackhole/wormhole tests live at tests/python_tests/," >&2
    echo "             quasar tests live at tests/python_tests/quasar/." >&2
    exit 3
  fi

  if [[ -z "$SIM_PATH" ]]; then
    SIM_PATH="/proj_sw/user_dev/${USER}/tt-umd-simulators/build/emu-${ARCH}-1x3"
  fi

  # Per-arch lock file: QSR/BH/WH agents only serialise against their own arch.
  if [[ -z "$LOCKFILE" ]]; then
    LOCKFILE="/tmp/tt-llk-test-${ARCH}.lock"
  fi

  # Mode inferred from arch: quasar runs against the UMD simulator; blackhole
  # and wormhole run against physical silicon. Hardware mode skips the
  # --run-simulator flag, port cleanup, and TT_UMD_SIMULATOR_PATH.
  case "$ARCH" in
    quasar)              MODE="simulator" ;;
    blackhole|wormhole)  MODE="hardware"  ;;
    *)                   MODE="simulator" ;;
  esac
}

# ── watchdog ──────────────────────────────────────────────────────────────────
# Hang detection runs as a sidecar process during _do_simulate. It writes
# HANG_LOG as the side-channel signal that the wrapper checks after wait —
# non-empty = "this was a hang, not a normal pytest failure" (same pattern as
# tt-metal's run_safe_pytest.sh uses with tt-triage's log file).
#
# Both flavours poll a heartbeat file's mtime (fed by tee-ing consumer
# stdout/stderr). They differ only in stall threshold and on-trip cleanup:
#   * QSR (emulation): emu-quasar-1x3 is an SSH wrapper that blocks on a
#     remote Zebu emulator — local CPU ticks stay flat for tens of seconds
#     even when the remote model is healthy, so a CPU-tick signal produces
#     false positives. SSH output cadence (model-load INFO lines, xtor
#     init, pytest progress) is the only reliable local signal.
#   * BH/WH (silicon): same heartbeat signal. On stall, run tt-triage.py to
#     capture device callstacks/NoC state, dump them to HANG_LOG, and
#     tt-smi -r afterwards.

HANG_POLL_SECS=5
HANG_STALL_QUASAR=120       # absorbs Zebu model-load quiet phases (~30s max)
HANG_STALL_SILICON=300      # loose: absorbs compile-consumer's quiet phases
LLK_TRIAGE_SCRIPT_REL=".claude/scripts/llk_triage.py"

# Background ticker: prints a `[progress]` line to stderr every
# PROGRESS_INTERVAL seconds for as long as the supervised PID is alive. Lets
# the caller see "still in compile / still in simulate" during phases that go
# tens of seconds without output (Zebu model load, BH/WH long compile).
# Heartbeat is optional: when supplied, the line also includes "last_output=Ns
# ago" so a quiet but progressing phase looks different from a wedged one.
_progress_ticker() {
  local supervised_pid="$1" heartbeat="$2" phase="$3"
  local start_ts now mtime hb_age elapsed
  start_ts=$(date +%s)
  while kill -0 "$supervised_pid" 2>/dev/null; do
    sleep "$PROGRESS_INTERVAL"
    kill -0 "$supervised_pid" 2>/dev/null || break
    now=$(date +%s)
    elapsed=$((now - start_ts))
    if [[ -n "$heartbeat" && -f "$heartbeat" ]]; then
      mtime=$(stat -c %Y "$heartbeat" 2>/dev/null || echo "$now")
      hb_age=$((now - mtime))
      echo "[run_llk_tests][progress] ${phase}: elapsed=${elapsed}s, last_output=${hb_age}s ago" >&2
    else
      echo "[run_llk_tests][progress] ${phase}: elapsed=${elapsed}s" >&2
    fi
  done
}

# Recursively kill PID and all descendants. We need this instead of
# `pkill -9 -P pid` (which only walks one level) because the consumer tree
# is `subshell → bash sim_script → pytest → xdist workers`. A shallow kill
# leaves pytest alive and reparented to init, where it keeps holding the
# device — which then makes the next run's consumer phase hang on device
# acquisition, masquerading as a "tt-smi -r didn't recover" symptom.
_kill_tree() {
  local pid="$1" child
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    _kill_tree "$child"
  done
  kill -9 "$pid" 2>/dev/null || true
}

# Emit a single-line outcome marker at the very end of a phase so callers
# (especially the llk-test-runner agent) can identify what happened by
# tailing the output instead of scanning the full pytest stream. Goes to
# stderr to stay distinct from pytest stdout.
_emit_verdict() {
  local exit_code="$1" phase="$2"
  local verdict
  case "$exit_code" in
    0) verdict="PASS" ;;
    1) verdict="FAIL" ;;
    2) verdict="COMPILE_FAIL" ;;
    3) verdict="ENV_ERROR" ;;
    4) verdict="BAD_ARGS" ;;
    5) verdict="HANG" ;;
    *) verdict="EXIT_${exit_code}" ;;
  esac
  echo "=== RUN_LLK_TESTS_VERDICT === ${verdict} (exit ${exit_code}, phase=${phase}, test=${TEST_FILE}, arch=${ARCH})" >&2
}

# Record why the watchdog tripped. Just the reason line — actual device
# triage runs later in _do_simulate's hang-handling section, AFTER the
# consumer tree has been killed (so a fresh ttexalens session can open the
# device handle the dying pytest just released).
_dump_hang_diagnosis() {
  local hang_log="$1" reason="$2"
  printf '%s\n' "$reason" >"$hang_log"
}

# Run the LLK-specific triage script and stream output to stderr. Called
# after _kill_tree (so the device handle is free) and before tt-smi -r
# (so the wedged Tensix state is still observable). The script needs the
# tests venv for ttexalens — activate it before invoking python3.
#
# tt-metal/tools/tt-triage.py is intentionally NOT used here. It reads
# state through Metal's Inspector subsystem (RPC socket or
# /tmp/tt-metal/inspector log dir). LLK tests bypass Metal entirely, so
# Inspector data never exists and every downstream tt-triage check just
# prints "Cannot run script due to failed dependencies".
_run_llk_triage() {
  local triage_script="${WORKTREE}/${LLK_TRIAGE_SCRIPT_REL}"
  if [[ ! -f "$triage_script" ]]; then
    echo "[llk-triage] not found: ${triage_script}" >&2
    return
  fi
  echo "--- llk-triage ---" >&2
  (
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    timeout 60 python3 "$triage_script" --arch "$ARCH" 2>&1
  ) >&2 || true
  echo "--- end llk-triage ---" >&2
}

# QSR watchdog. Polls the heartbeat file's mtime — emu-quasar-1x3 is an SSH
# wrapper that's mostly idle while the remote Zebu emulator runs, so CPU
# ticks on the local wrapper are flat even during healthy emulation. The
# stream of model-load / xtor / pytest output piped through the heartbeat
# file is the reliable local signal that the remote side is alive.
_watchdog_emu() {
  local consumer_pid="$1" hang_log="$2" heartbeat="$3"
  local now mtime quiet_for

  while kill -0 "$consumer_pid" 2>/dev/null; do
    sleep "$HANG_POLL_SECS"
    now=$(date +%s)
    mtime=$(stat -c %Y "$heartbeat" 2>/dev/null || echo "$now")
    quiet_for=$((now - mtime))
    if [[ "$quiet_for" -ge "$HANG_STALL_QUASAR" ]]; then
      _dump_hang_diagnosis "$hang_log" \
        "emu-quasar hang: no output for ${quiet_for}s (threshold ${HANG_STALL_QUASAR}s)"
      pkill -9 -f "emu-${ARCH}" 2>/dev/null || true
      _kill_tree "$consumer_pid"
      return
    fi
  done
}

# Silicon (BH/WH) watchdog. Two trip conditions:
#   1. TENSIX TIMED OUT appears in the consumer's output. Inline detection
#      lets us catch a per-test timeout the moment pytest emits it. NOTE: with
#      stock pytest flags, the longrepr containing this string is buffered until
#      the end-of-run failures section, so this path mostly catches the tail.
#      The post-mortem grep in _do_simulate covers the buffered case.
#   2. Heartbeat mtime hasn't advanced past HANG_STALL_SILICON seconds — pytest
#      itself is wedged with no output at all (kernel hang where the Python
#      side never gets its own timeout).
# On trip: write HANG_LOG (with tt-triage output if available), kill the
# consumer tree, return. _do_simulate runs tt-smi -r after wait so it doesn't
# race with the still-running consumer.
_watchdog_silicon() {
  local consumer_pid="$1" hang_log="$2" heartbeat="$3"
  local now mtime quiet_for tensix_line

  while kill -0 "$consumer_pid" 2>/dev/null; do
    sleep "$HANG_POLL_SECS"

    # Live path: TENSIX TIMED OUT in the output stream.
    tensix_line=$(grep -m1 "TENSIX TIMED OUT" "$heartbeat" 2>/dev/null || true)
    if [[ -n "$tensix_line" ]]; then
      _dump_hang_diagnosis "$hang_log" \
        "tensix-timeout in output: ${tensix_line}"
      _kill_tree "$consumer_pid"
      return
    fi

    # Stall path: no output at all for HANG_STALL_SILICON seconds.
    now=$(date +%s)
    mtime=$(stat -c %Y "$heartbeat" 2>/dev/null || echo "$now")
    quiet_for=$((now - mtime))
    if [[ "$quiet_for" -ge "$HANG_STALL_SILICON" ]]; then
      _dump_hang_diagnosis "$hang_log" \
        "silicon hang: no pytest output for ${quiet_for}s (threshold ${HANG_STALL_SILICON}s)"
      _kill_tree "$consumer_pid"
      return
    fi
  done
}

# ── count ─────────────────────────────────────────────────────────────────────
# Outputs the variant count (integer) to stdout.
# Collection log (including any errors) goes to stderr so the caller can
# capture just the count: COUNT=$(run_llk_tests.sh count ...)

_do_count() {
  _validate
  _vlog "count: ${TEST_FILE} (arch=${ARCH})"

  local output exit_code
  exit_code=0
  output=$(
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    cd "${TEST_DIR}"
    CHIP_ARCH="${ARCH}" pytest --compile-producer --co -q "${TEST_FILE}" 2>&1
  ) || exit_code=$?

  # Show collection log to the agent (stderr stays visible in Bash tool output)
  printf '%s\n' "${output}" >&2

  if [[ $exit_code -ne 0 ]]; then
    echo "0"
    return "$exit_code"
  fi

  # Last non-blank line looks like "N tests collected" or "N test collected"
  local last count
  last=$(printf '%s\n' "${output}" | grep -v '^[[:space:]]*$' | tail -1)
  count=$(printf '%s' "${last}" | grep -oE '^[0-9]+' || echo "0")
  echo "${count}"
}

# ── compile ───────────────────────────────────────────────────────────────────
# Runs compile-producer with -n JOBS parallelism.
# No flock needed — compile does not touch the simulator.

_do_compile() {
  _validate
  _vlog "compile: ${TEST_FILE} (arch=${ARCH}, -n ${JOBS}${K_FILTER:+, -k '${K_FILTER}'}$([[ "$SPEED_OF_LIGHT" == true ]] && echo ', sol'))"

  # The two-phase flow requires compile and simulate to filter to the same set:
  # simulate's --compile-consumer reads per-variant artifacts that producer
  # wrote, so an unfiltered producer + filtered consumer would either rebuild
  # variants the consumer skips or miss variants the consumer needs.
  # Resolution order matches _do_simulate: --test-id > --k > whole file.
  local -a pytest_args=()
  if [[ -n "$TEST_ID" ]]; then
    pytest_args=("$TEST_ID")
  elif [[ -n "$K_FILTER" ]]; then
    pytest_args=(-k "$K_FILTER" "$TEST_FILE")
  else
    pytest_args=("$TEST_FILE")
  fi
  [[ "$SPEED_OF_LIGHT" == "true" ]] && pytest_args=(--speed-of-light "${pytest_args[@]}")

  # Background pytest so the progress ticker can run alongside. If progress is
  # disabled we still pay the (negligible) cost of background+wait — keeps the
  # control flow uniform.
  local compile_pid progress_pid="" compile_exit
  if [[ -n "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
    (
      # shellcheck disable=SC1091
      source "${VENV}/bin/activate"
      cd "${TEST_DIR}"
      CHIP_ARCH="${ARCH}" pytest --compile-producer -n "${JOBS}" -x "${pytest_args[@]}"
    ) > >(tee -a "${LOG_DIR}/compile.log") 2> >(tee -a "${LOG_DIR}/compile.log" >&2) &
    compile_pid=$!
  else
    (
      # shellcheck disable=SC1091
      source "${VENV}/bin/activate"
      cd "${TEST_DIR}"
      CHIP_ARCH="${ARCH}" pytest --compile-producer -n "${JOBS}" -x "${pytest_args[@]}"
    ) &
    compile_pid=$!
  fi

  if [[ "$PROGRESS_ENABLED" == "true" ]]; then
    _progress_ticker "$compile_pid" "" "compile" &
    progress_pid=$!
  fi

  wait "$compile_pid"
  compile_exit=$?

  if [[ -n "$progress_pid" ]]; then
    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true
  fi

  return "$compile_exit"
}

# ── simulate ──────────────────────────────────────────────────────────────────
# Runs the consumer step under a per-arch flock so only one agent at a time
# uses the resource for that arch — the UMD simulator for quasar, or the
# physical card for blackhole / wormhole. All internals (temp-script lifecycle,
# stale-process cleanup, lock acquisition) are handled here; callers just read
# the exit code.

_do_simulate() {
  _validate
  if [[ "$MODE" == "simulator" ]]; then
    _vlog "consume: ${TEST_FILE} (arch=${ARCH}, mode=simulator, port=${PORT})"
  else
    _vlog "consume: ${TEST_FILE} (arch=${ARCH}, mode=hardware)"
  fi

  # Remove temp scripts left by runs that died before their trap fired
  find /tmp -maxdepth 1 -name 'llk_run_sim_*.sh' -mmin +60 -delete 2>/dev/null || true

  # Write the pytest invocation to a temp file so that:
  #   1. flock can use "flock LOCK bash SCRIPT" without inline quoting nightmares
  #   2. TEST_ID values (single-quotes, brackets, commas) are literal, not shell-expanded
  local sim_script hang_log heartbeat
  sim_script=$(mktemp /tmp/llk_run_sim_XXXXXX.sh)
  hang_log=$(mktemp /tmp/llk_hang_XXXXXX.log)
  heartbeat=$(mktemp /tmp/llk_heartbeat_XXXXXX.log)
  : >"$hang_log"
  : >"$heartbeat"
  # Trap ensures temp files are removed even on SIGTERM / SIGINT
  trap 'rm -f "${sim_script}" "${hang_log}" "${heartbeat}"' EXIT INT TERM

  # Build the pytest flags string.
  # Simulator mode adds --run-simulator/--port; hardware mode targets silicon directly.
  # --no-split: run combined (no prior compile step needed, no --compile-consumer).
  # Default (split): requires a prior compile-producer run; passes --compile-consumer.
  local pytest_flags="--timeout=${TIMEOUT} -rN"
  if [[ "$MODE" == "simulator" ]]; then
    pytest_flags="${pytest_flags} --run-simulator --port=${PORT}"
  fi
  [[ "$NO_SPLIT" == "false" ]] && pytest_flags="${pytest_flags} --compile-consumer"
  [[ "$SPEED_OF_LIGHT" == "true" ]] && pytest_flags="${pytest_flags} --speed-of-light"
  [[ -n "$MAXFAIL" ]] && pytest_flags="${pytest_flags} --maxfail=${MAXFAIL}"

  # Determine the test target (file, -k filter, or single variant by ID)
  local pytest_target
  if [[ -n "$TEST_ID" ]]; then
    # printf '%q' produces bash-safe quoting even for IDs with single-quotes
    printf -v pytest_target '%q' "${TEST_ID}"
  elif [[ -n "$K_FILTER" ]]; then
    local quoted_k
    printf -v quoted_k '%q' "${K_FILTER}"
    pytest_target="-k ${quoted_k} '${TEST_FILE}'"
  else
    pytest_target="'${TEST_FILE}'"
  fi

  # Write the consumer script. Simulator mode prepends port-cleanup and the
  # TT_UMD_SIMULATOR_PATH env var; hardware mode skips both since BH/WH run
  # against silicon directly.
  # Single-quotes around variable expansions are intentional: the values are
  # fixed at script-write time and contain no shell-special chars that matter
  # to the inner bash invocation.
  {
    printf '#!/bin/bash\n'
    printf 'set -u\n\n'
    if [[ "$MODE" == "simulator" ]]; then
      printf '# Kill any process holding port %s\n' "${PORT}"
      printf 'STALE=$(lsof -ti :%s 2>/dev/null || true)\n' "${PORT}"
      printf 'if [ -n "$STALE" ]; then\n'
      printf '  echo "[run_llk_tests] Killing stale port %s processes: $STALE"\n' "${PORT}"
      printf '  echo "$STALE" | xargs kill -9 2>/dev/null || true\n'
      printf 'fi\n'
      printf 'pkill -9 -f "tt-exalens.*--port=%s" 2>/dev/null || true\n' "${PORT}"
      printf 'sleep 1\n\n'
    fi
    printf 'source %q\n' "${VENV}/bin/activate"
    printf 'cd %q\n\n' "${TEST_DIR}"
    if [[ "$MODE" == "simulator" ]]; then
      printf 'TT_UMD_SIMULATOR_PATH=%q \\\n' "${SIM_PATH}"
    fi
    printf '  CHIP_ARCH=%q \\\n' "${ARCH}"
    printf '  pytest %s %s\n' "${pytest_flags}" "${pytest_target}"
  } >"${sim_script}"
  chmod +x "${sim_script}"

  _vlog "Acquiring ${MODE} lock: ${LOCKFILE} (timeout=${LOCK_TIMEOUT}s)"

  # Use fd-based flock so we can distinguish lock-timeout (exit 3) from test
  # failure (exit 1). The subshell scopes the fd so it closes automatically.
  # Consumer is backgrounded so the watchdog runs concurrently; `wait` gives
  # us pytest's exit code. Output is tee'd through the heartbeat file so the
  # silicon watchdog has an mtime signal to poll. LOG_DIR adds an extra tee
  # target without disturbing the heartbeat path.
  local sim_exit consumer_pid watchdog_pid=""
  if [[ -n "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
    (
      exec 9>>"${LOCKFILE}"
      if ! flock --timeout "${LOCK_TIMEOUT}" 9; then
        echo "ERROR: Could not acquire ${MODE} lock ${LOCKFILE} after ${LOCK_TIMEOUT}s" >&2
        echo "       Another agent is likely running on arch=${ARCH}. Retry or check for stale locks." >&2
        exit 3
      fi
      _vlog "Lock acquired; running ${MODE}"
      bash "${sim_script}"
    ) > >(tee -a "${heartbeat}" "${LOG_DIR}/run.log") 2> >(tee -a "${heartbeat}" "${LOG_DIR}/run.log" >&2) &
    consumer_pid=$!
  else
    (
      exec 9>>"${LOCKFILE}"
      if ! flock --timeout "${LOCK_TIMEOUT}" 9; then
        echo "ERROR: Could not acquire ${MODE} lock ${LOCKFILE} after ${LOCK_TIMEOUT}s" >&2
        echo "       Another agent is likely running on arch=${ARCH}. Retry or check for stale locks." >&2
        exit 3
      fi
      _vlog "Lock acquired; running ${MODE}"
      bash "${sim_script}"
    ) > >(tee -a "${heartbeat}") 2> >(tee -a "${heartbeat}" >&2) &
    consumer_pid=$!
  fi

  # Spawn the appropriate watchdog. Picked by MODE rather than ARCH so future
  # simulator/silicon arches inherit the right behaviour automatically.
  if [[ "$MODE" == "simulator" ]]; then
    _watchdog_emu "$consumer_pid" "$hang_log" "$heartbeat" &
    watchdog_pid=$!
    _vlog "Watchdog: emu-${ARCH} heartbeat monitor (stall=${HANG_STALL_QUASAR}s, poll=${HANG_POLL_SECS}s)"
  else
    _watchdog_silicon "$consumer_pid" "$hang_log" "$heartbeat" &
    watchdog_pid=$!
    _vlog "Watchdog: silicon stdout-cadence (stall=${HANG_STALL_SILICON}s, poll=${HANG_POLL_SECS}s)"
  fi

  local progress_pid=""
  if [[ "$PROGRESS_ENABLED" == "true" ]]; then
    _progress_ticker "$consumer_pid" "$heartbeat" "simulate" &
    progress_pid=$!
  fi

  wait "$consumer_pid"
  sim_exit=$?

  # Stop watchdog. If it already tripped and returned, this is a no-op.
  kill "$watchdog_pid" 2>/dev/null || true
  wait "$watchdog_pid" 2>/dev/null || true

  if [[ -n "$progress_pid" ]]; then
    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true
  fi

  # Post-mortem: if the watchdog didn't trip live but pytest exited non-zero
  # AND the captured output contains TENSIX TIMED OUT, reclassify as a hang.
  # This is the common path on silicon: pytest's failures section emits the
  # longrepr only at end of run, so the watchdog's live grep usually fires too
  # late to kill pytest — but the device is still wedged from the last failing
  # variant, so tt-triage can still read meaningful state.
  if [[ ! -s "$hang_log" && $sim_exit -ne 0 && "$MODE" == "hardware" ]]; then
    if grep -q "TENSIX TIMED OUT" "$heartbeat" 2>/dev/null; then
      _dump_hang_diagnosis "$hang_log" \
        "post-mortem: TENSIX TIMED OUT found in pytest output (pytest exit ${sim_exit})"
    fi
  fi

  # Classification: non-empty hang_log = watchdog tripped or post-mortem
  # caught it = this was a hang, not a normal failure. Override sim_exit with
  # 5 and run the arch-specific cleanup (sim straggler kill, or device reset).
  if [[ -s "$hang_log" ]]; then
    echo "========================================" >&2
    echo "RUN_LLK_TESTS_HANG: watchdog tripped" >&2
    cat "$hang_log" >&2
    if [[ "$MODE" == "simulator" ]]; then
      pkill -9 -f "emu-${ARCH}" 2>/dev/null || true
    else
      # Reparented pytest descendants (PPID=1 after the parent subshell
      # died) aren't reachable from _kill_tree, so pattern-kill them by
      # cmdline before tt-smi -r. If a stale pytest survives the reset
      # holding the device, the NEXT run's consumer phase hangs on device
      # acquisition and looks like "tt-smi -r didn't work" — it did, but
      # there was nothing to reset to.
      pkill -9 -f "pytest.*--compile-consumer" 2>/dev/null || true
      # LLK-specific triage runs HERE: the consumer's ttexalens session
      # was released by _kill_tree + the safety-net pkill, but the Tensix
      # is still wedged. A fresh ttexalens session can now read mailbox /
      # RISC state. After triage, tt-smi -r fully resets the device.
      _run_llk_triage
      echo "Resetting device (tt-smi -r)..." >&2
      tt-smi -r >&2 || echo "WARNING: tt-smi -r failed" >&2
    fi
    echo "========================================" >&2
    sim_exit=5
  fi

  rm -f "${sim_script}" "${hang_log}" "${heartbeat}"
  trap - EXIT INT TERM

  return "$sim_exit"
}

# ── run ───────────────────────────────────────────────────────────────────────
# With --no-split: skips the compile step, calls simulate directly (combined mode).
# Default (split): compile first (exits 2 on failure), then simulate.

_do_run() {
  _validate
  _vlog "run ($([ "$NO_SPLIT" = true ] && echo combined || echo compile+simulate)): ${TEST_FILE} (arch=${ARCH})"

  if [[ "$NO_SPLIT" == "false" ]]; then
    _do_compile
    local compile_exit=$?
    if [[ $compile_exit -ne 0 ]]; then
      echo "ERROR: compile step failed (exit ${compile_exit})" >&2
      return 2
    fi
  fi

  _do_simulate
}

# ── Dispatch ─────────────────────────────────────────────────────────────────

_dispatch_exit=0
case "${CMD}" in
  count)    _do_count    ; _dispatch_exit=$? ;;
  compile)  _do_compile  ; _dispatch_exit=$? ;;
  simulate) _do_simulate ; _dispatch_exit=$? ;;
  run)      _do_run      ; _dispatch_exit=$? ;;
  help|--help|-h)
    sed -n 's/^# \{0,1\}//p' "$0" | head -60
    exit 0
    ;;
  "")
    echo "ERROR: No command specified. Use: count | compile | simulate | run" >&2
    exit 4
    ;;
  *)
    echo "ERROR: Unknown command '${CMD}'. Use: count | compile | simulate | run" >&2
    exit 4
    ;;
esac

# Emit a single verdict line for the workflow commands. `count` is excluded
# because its stdout contract is "just the integer" — a marker line would
# corrupt callers like `COUNT=$(run_test.sh count …)`.
case "${CMD}" in
  compile|simulate|run) _emit_verdict "$_dispatch_exit" "$CMD" ;;
esac

exit "$_dispatch_exit"
