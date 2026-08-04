#!/bin/bash
# run_llk_tests.sh — centralised LLK test runner for codegen agents.
#
# Encapsulates the two-step compile-then-simulate flow, flock-based simulator
# serialisation, stale-process cleanup, and temp-file lifecycle so agents never
# have to manage any of that themselves.
#
# Usage:
#   run_llk_tests.sh <COMMAND> --worktree DIR --arch ARCH --test FILE [FILE ...] [OPTIONS]
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
#   --test     FILE   Test file name, e.g. test_sfpu_square_quasar.py.
#                     Additional trailing test file names are accepted for
#                     count/compile/simulate/run.
#
# Optional:
#   --maxfail  N      Stop after N failures (simulate/run; omit for verification)
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
TEST_FILES=()
MAXFAIL=""
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
VERBOSE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worktree)      WORKTREE="$2";      shift 2 ;;
    --arch)          ARCH="$2";          shift 2 ;;
    --test)          TEST_FILES+=("$2"); shift 2 ;;
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
    --speed-of-light) SPEED_OF_LIGHT="true"; shift ;;
    --no-split)      NO_SPLIT="true";    shift   ;;
    --verbose|-v)    VERBOSE="true";     shift   ;;
    --help|-h)
      sed -n 's/^# \{0,1\}//p' "$0" | head -60
      exit 0
      ;;
    *)
      if [[ "$1" == -* ]]; then
        echo "ERROR: Unknown option: $1" >&2
        echo "Run with --help for usage." >&2
        exit 4
      fi
      TEST_FILES+=("$1")
      shift
      ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────

_vlog() {
  if [[ "$VERBOSE" == "true" ]]; then
    echo "[run_llk_tests] $*" >&2
  fi
}

_quote_args() {
  local out="" quoted arg
  for arg in "$@"; do
    printf -v quoted '%q' "$arg"
    out+="${quoted} "
  done
  printf '%s' "${out% }"
}

_test_label() {
  local joined
  printf -v joined '%s, ' "${TEST_FILES[@]}"
  printf '%s' "${joined%, }"
}

# Validate required args and derive VENV, TEST_DIR, SIM_PATH.
# Sets these as script-global variables; exits on any problem.
_validate() {
  local errors=0
  [[ -z "$WORKTREE"  ]] && { echo "ERROR: --worktree is required" >&2; ((errors++)); }
  [[ -z "$ARCH"      ]] && { echo "ERROR: --arch is required"     >&2; ((errors++)); }
  [[ ${#TEST_FILES[@]} -eq 0 ]] && { echo "ERROR: --test is required"     >&2; ((errors++)); }
  [[ -n "$TEST_ID" && ${#TEST_FILES[@]} -gt 1 ]] && {
    echo "ERROR: --test-id can only be used with one test file" >&2
    ((errors++))
  }
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
  local test_file
  for test_file in "${TEST_FILES[@]}"; do
    if [[ ! -f "${TEST_DIR}/${test_file}" ]]; then
      echo "ERROR: test file not found: ${TEST_DIR}/${test_file}" >&2
      echo "       Hint: blackhole/wormhole tests live at tests/python_tests/," >&2
      echo "             quasar tests live at tests/python_tests/quasar/." >&2
      exit 3
    fi
  done

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

# ── count ─────────────────────────────────────────────────────────────────────
# Outputs the variant count (integer) to stdout.
# Collection log (including any errors) goes to stderr so the caller can
# capture just the count: COUNT=$(run_llk_tests.sh count ...)

_do_count() {
  _validate
  _vlog "count: $(_test_label) (arch=${ARCH})"

  local output exit_code
  exit_code=0
  output=$(
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    cd "${TEST_DIR}"
    CHIP_ARCH="${ARCH}" pytest --compile-producer --co -q "${TEST_FILES[@]}" 2>&1
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

  # The two-phase flow requires compile and simulate to filter to the same set:
  # simulate's --compile-consumer reads per-variant artifacts that producer
  # wrote, so an unfiltered producer + filtered consumer would either rebuild
  # variants the consumer skips or miss variants the consumer needs.
  local -a kflag=()
  local -a solflag=()
  [[ "$SPEED_OF_LIGHT" == "true" ]] && solflag=(--speed-of-light)
  local -a pytest_targets=("${TEST_FILES[@]}")
  local target_label="$(_test_label)"
  if [[ -n "$TEST_ID" ]]; then
    pytest_targets=("$TEST_ID")
    target_label="$TEST_ID"
  elif [[ -n "$K_FILTER" ]]; then
    kflag=(-k "$K_FILTER")
  fi

  _vlog "compile: ${target_label} (arch=${ARCH}, -n ${JOBS}${kflag[*]:+, -k '${K_FILTER}'}$([[ "$SPEED_OF_LIGHT" == true ]] && echo ', sol'))"

  if [[ -n "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
    (
      # shellcheck disable=SC1091
      source "${VENV}/bin/activate"
      cd "${TEST_DIR}"
      CHIP_ARCH="${ARCH}" pytest --compile-producer -n "${JOBS}" "${solflag[@]}" "${kflag[@]}" "${pytest_targets[@]}"
    ) > >(tee -a "${LOG_DIR}/compile.log") 2> >(tee -a "${LOG_DIR}/compile.log" >&2)
  else
    (
      # shellcheck disable=SC1091
      source "${VENV}/bin/activate"
      cd "${TEST_DIR}"
      CHIP_ARCH="${ARCH}" pytest --compile-producer -n "${JOBS}" "${solflag[@]}" "${kflag[@]}" "${pytest_targets[@]}"
    )
  fi
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
    _vlog "consume: $(_test_label) (arch=${ARCH}, mode=simulator, port=${PORT})"
  else
    _vlog "consume: $(_test_label) (arch=${ARCH}, mode=hardware)"
  fi

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
    pytest_target="-k ${quoted_k} $(_quote_args "${TEST_FILES[@]}")"
  else
    pytest_target="$(_quote_args "${TEST_FILES[@]}")"
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
  # failure (exit 1).  The subshell scopes the fd so it closes automatically.
  local sim_exit
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
    ) > >(tee -a "${LOG_DIR}/run.log") 2> >(tee -a "${LOG_DIR}/run.log" >&2)
    sim_exit=$?
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
    )
    sim_exit=$?
  fi

  rm -f "${sim_script}"
  trap - EXIT INT TERM

  return "$sim_exit"
}

# ── run ───────────────────────────────────────────────────────────────────────
# With --no-split: skips the compile step, calls simulate directly (combined mode).
# Default (split): compile first (exits 2 on failure), then simulate.

_do_run() {
  _validate
  _vlog "run ($([ "$NO_SPLIT" = true ] && echo combined || echo compile+simulate)): $(_test_label) (arch=${ARCH})"

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
