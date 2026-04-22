#!/bin/bash
# run_llk_tests.sh — centralised LLK test runner for codegen agents.
#
# Encapsulates the two-step compile-then-simulate flow, flock-based simulator
# serialisation, stale-process cleanup, and temp-file lifecycle so agents never
# have to manage any of that themselves.
#
# Usage:
#   run_llk_tests.sh <COMMAND> --worktree DIR --arch ARCH --test FILE [OPTIONS]
#
# Commands:
#   count     Count test variants (collection-only; outputs integer to stdout)
#   compile   Run compile-producer step (parallel, no flock needed)
#   simulate  Run simulate-consumer step (flock-serialised, no xdist)
#   run       compile + simulate in sequence (most common agent use case)
#
# Required:
#   --worktree DIR    Absolute path to the LLK working directory
#                     (contains tests/ and tt_llk_<arch>/ as direct children)
#   --arch     ARCH   Target architecture (quasar, blackhole, ...)
#   --test     FILE   Test file name, e.g. test_sfpu_square_quasar.py
#
# Optional:
#   --maxfail  N      Stop after N failures (simulate/run; omit for verification)
#   --k        EXPR   pytest -k filter expression
#   --test-id  ID     Full parametrize ID for a single variant run
#                     (single-quotes, brackets, commas are safe — no escaping needed)
#   --port     PORT   Simulator port (default: 5556)
#   --timeout  SECS   pytest --timeout ceiling (default: 600)
#   --jobs     N      Compile parallelism (default: 15)
#   --lock     FILE   flock lock file (default: /tmp/tt-llk-test-simulator.lock)
#   --lock-timeout N  Seconds to wait for the lock (default: 900)
#   --sim-path PATH   Override TT_UMD_SIMULATOR_PATH
#                     (default: /proj_sw/user_dev/$USER/tt-umd-simulators/build/emu-<arch>-1x3)
#   --no-split        Skip compile-producer step; run pytest --run-simulator without
#                     --compile-consumer (combined compile+run in one pytest invocation).
#                     Use for issue-solver tests that don't pre-build ELFs.
#   --verbose         Print step headers to stderr
#
# Exit codes:
#   0  All tests passed (or count written to stdout successfully)
#   1  One or more tests failed
#   2  Compile step failed  (only from the 'run' command)
#   3  Environment error (flock timeout, simulator port stuck, venv missing)
#   4  Usage / validation error (missing required options)
#
# Agents invoke this via the Bash tool with timeout: 1800000 (synchronous,
# never run_in_background). The script blocks until completion and returns one
# of the exit codes above — the agent reads that code and decides the diagnosis
# (PASS / compile error / test failure / ENV_ERROR) without parsing internals.
#
# Examples:
#   # Count variants before deciding --maxfail (see tester Step 2.1 table)
#   VARIANT_COUNT=$(bash "$WORKTREE_DIR/codegen/scripts/run_llk_tests.sh" count \
#       --worktree "$WORKTREE_DIR" --arch quasar --test test_sfpu_square_quasar.py)
#
#   # Compile + simulate, stop after 5 failures
#   bash "$WORKTREE_DIR/codegen/scripts/run_llk_tests.sh" run \
#       --worktree "$WORKTREE_DIR" --arch quasar --test test_sfpu_square_quasar.py \
#       --maxfail 5
#   RUN_EXIT=$?
#
#   # Verification run — full matrix, no maxfail
#   bash "$WORKTREE_DIR/codegen/scripts/run_llk_tests.sh" run \
#       --worktree "$WORKTREE_DIR" --arch quasar --test test_sfpu_square_quasar.py
#
#   # Single variant by parametrize ID (single-quotes and brackets are safe)
#   bash "$WORKTREE_DIR/codegen/scripts/run_llk_tests.sh" simulate \
#       --worktree "$WORKTREE_DIR" --arch quasar --test test_sfpu_square_quasar.py \
#       --test-id "test_sfpu_square_quasar.py::test_sfpu_square_quasar[formats:(..., 'SyncFull', ...)]"

# ── Parse ────────────────────────────────────────────────────────────────────

CMD="${1:-}"
shift 2>/dev/null || true

WORKTREE=""
ARCH=""
TEST_FILE=""
MAXFAIL=""
K_FILTER=""
TEST_ID=""
PORT="5556"
TIMEOUT="600"
JOBS="15"
LOCKFILE="/tmp/tt-llk-test-simulator.lock"
LOCK_TIMEOUT="900"
SIM_PATH=""
NO_SPLIT="false"
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
  TEST_DIR="${WORKTREE}/tests/python_tests/${ARCH}"

  if [[ ! -d "$VENV" ]]; then
    echo "ERROR: venv not found: ${VENV}" >&2
    echo "       Run 'CHIP_ARCH=${ARCH} ./setup_testing_env.sh' in ${WORKTREE}/tests/ first." >&2
    exit 3
  fi
  if [[ ! -d "$TEST_DIR" ]]; then
    echo "ERROR: test directory not found: ${TEST_DIR}" >&2
    exit 3
  fi

  if [[ -z "$SIM_PATH" ]]; then
    SIM_PATH="/proj_sw/user_dev/${USER}/tt-umd-simulators/build/emu-${ARCH}-1x3"
  fi
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
  count=$(printf '%s' "${last}" | grep -oP '^\d+' || echo "0")
  echo "${count}"
}

# ── compile ───────────────────────────────────────────────────────────────────
# Runs compile-producer with -n JOBS parallelism.
# No flock needed — compile does not touch the simulator.

_do_compile() {
  _validate
  _vlog "compile: ${TEST_FILE} (arch=${ARCH}, -n ${JOBS})"

  (
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    cd "${TEST_DIR}"
    CHIP_ARCH="${ARCH}" pytest --compile-producer -n "${JOBS}" "${TEST_FILE}"
  )
}

# ── simulate ──────────────────────────────────────────────────────────────────
# Runs the simulate-consumer step under flock so only one agent uses the
# simulator at a time.  All internals (temp-script lifecycle, stale-process
# cleanup, lock acquisition) are handled here — callers just read the exit code.

_do_simulate() {
  _validate
  _vlog "simulate: ${TEST_FILE} (arch=${ARCH}, port=${PORT})"

  # Remove temp scripts left by runs that died before their trap fired
  find /tmp -maxdepth 1 -name 'llk_run_sim_*.sh' -mmin +60 -delete 2>/dev/null || true

  # Write the pytest invocation to a temp file so that:
  #   1. flock can use "flock LOCK bash SCRIPT" without inline quoting nightmares
  #   2. TEST_ID values (single-quotes, brackets, commas) are literal, not shell-expanded
  local sim_script
  sim_script=$(mktemp /tmp/llk_run_sim_XXXXXX.sh)
  # Trap ensures the temp file is removed even on SIGTERM / SIGINT
  trap 'rm -f "${sim_script}"' EXIT INT TERM

  # Build the pytest flags string.
  # --no-split: run combined (no prior compile step needed, no --compile-consumer).
  # Default (split): requires a prior compile-producer run; passes --compile-consumer.
  local pytest_flags="--run-simulator --port=${PORT} --timeout=${TIMEOUT} -rN"
  [[ "$NO_SPLIT" == "false" ]] && pytest_flags="${pytest_flags} --compile-consumer"
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

  # Write the simulator script.
  # Single-quotes around variable expansions are intentional: the values are
  # fixed at script-write time and contain no shell-special chars that matter
  # to the inner bash invocation.
  {
    printf '#!/bin/bash\n'
    printf 'set -u\n\n'
    printf '# Kill any process holding port %s\n' "${PORT}"
    printf 'STALE=$(lsof -ti :%s 2>/dev/null || true)\n' "${PORT}"
    printf 'if [ -n "$STALE" ]; then\n'
    printf '  echo "[run_llk_tests] Killing stale port %s processes: $STALE"\n' "${PORT}"
    printf '  echo "$STALE" | xargs kill -9 2>/dev/null || true\n'
    printf 'fi\n'
    printf 'pkill -9 -f "tt-exalens.*--port=%s" 2>/dev/null || true\n' "${PORT}"
    printf 'sleep 1\n\n'
    printf 'source %q\n' "${VENV}/bin/activate"
    printf 'cd %q\n\n' "${TEST_DIR}"
    printf 'TT_UMD_SIMULATOR_PATH=%q \\\n' "${SIM_PATH}"
    printf '  CHIP_ARCH=%q \\\n' "${ARCH}"
    printf '  pytest %s %s\n' "${pytest_flags}" "${pytest_target}"
  } >"${sim_script}"
  chmod +x "${sim_script}"

  _vlog "Acquiring simulator lock: ${LOCKFILE} (timeout=${LOCK_TIMEOUT}s)"

  # Use fd-based flock so we can distinguish lock-timeout (exit 3) from test
  # failure (exit 1).  The subshell scopes the fd so it closes automatically.
  local sim_exit
  (
    exec 9>>"${LOCKFILE}"
    if ! flock --timeout "${LOCK_TIMEOUT}" 9; then
      echo "ERROR: Could not acquire simulator lock ${LOCKFILE} after ${LOCK_TIMEOUT}s" >&2
      echo "       Another agent is likely running the simulator. Retry or check for stale locks." >&2
      exit 3
    fi
    _vlog "Lock acquired; running simulator"
    bash "${sim_script}"
  )
  sim_exit=$?

  rm -f "${sim_script}"
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

case "${CMD}" in
  count)    _do_count    ;;
  compile)  _do_compile  ;;
  simulate) _do_simulate ;;
  run)      _do_run      ;;
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
