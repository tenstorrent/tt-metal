#!/bin/bash
# run_test.sh — synchronous LLK test runner for codegen agents and humans.
#
# Invoke it like pytest: one blocking call, wait for the verdict. There is no
# timeout to set and no resume loop. The only wait is on the global lock (first
# come, first served, unbounded); once a run starts, a watcher bounds it — a hang
# is detected from a log stall and killed gracefully, so the call always returns
# on its own.
#
# Two device paths:
#   quasar              emulator — pytest --run-simulator --port; tt-exalens boots,
#                       runs, tears down. HANG = post-ready log stall.
#   blackhole/wormhole  real silicon (/dev/tenstorrent). HANG = TENSIX TIMED OUT
#                       or log stall; recovered with llk_triage.py + tt-smi -r.
#
# ONE global lock (/tmp/tt-llk-test.lock) serialises every invocation on the host,
# so while a run holds it no peer can wipe the shared build cache.
#
# BUILD STAMP (why simulate can rebuild): the shared build cache is wiped+rebuilt
# by any producer. `compile` is lock-free and stamps the cache with a fingerprint
# of THIS run's source. `simulate`/`run` take the lock and, if the stamp is not
# ours (a peer recompiled), rebuild under the lock before running — so the ELFs
# executed are always built from our own source.
#
# Usage:
#   run_test.sh <COMMAND> --worktree DIR --arch ARCH --test FILE [OPTIONS]
#
# Commands:
#   count     Count test variants (collection-only; prints an integer). Lock-free.
#   compile   Compile-producer step (parallel, -x). Lock-free. Stamps the build.
#   simulate  Run the pre-built variants. Takes the lock; rebuilds under it when
#             the build stamp is not ours; then runs on the device/emulator.
#   run       compile + run in one held-lock session (always rebuilds).
#
# Required:
#   --worktree DIR    LLK working dir (contains tests/ and tt_llk_<arch>/).
#   --arch     ARCH   quasar | blackhole | wormhole.
#   --test     FILE   Test file, e.g. test_sfpu_where_quasar.py
#
# Optional:
#   --maxfail  N      Stop after N failures (default 10). simulate/run only — lets a
#                     few variants fail so their tile dumps reveal the pattern, then
#                     pytest ends cleanly.
#   --k        EXPR   pytest -k filter (applied to compile AND run).
#   --test-id  ID     Full parametrize id (quotes/brackets safe). A leading
#                     "<arch>/" rootdir prefix is stripped automatically.
#   --no-split        Combined compile+run in one pytest invocation.
#   --jobs     N      compile parallelism (default 15).
#   --port     PORT   tt-exalens server port (default 5556).
#   --sim-path PATH   Override TT_UMD_SIMULATOR_PATH.
#   --lock     FILE   Global lock file (default /tmp/tt-llk-test.lock).
#   --log-dir  DIR    Append the run's output to <DIR>/run.log (compile output to
#                     <DIR>/compile.log).
#   --stall    SECS   Log-stall seconds that mark a hang (default 180 emulator,
#                     300 silicon). Also settable via HANG_STALL.
#   --verbose         Print step headers to stderr.
#
# Exit codes:
#   0  PASS   1  FAIL   2  COMPILE_FAIL   3  ENV_ERROR   4  BAD_ARGS   5  HANG
#
# Verdict line (always last, on stderr):
#   === RUN_LLK_TESTS_VERDICT === <V> (exit N, phase=<cmd>, test=<f>, arch=<a>)

# ── Args ─────────────────────────────────────────────────────────────────────

CMD="${1:-}"
shift 2>/dev/null || true

WORKTREE="" ARCH="" TEST_FILE=""
MAXFAIL="10" K_FILTER="" TEST_ID=""
PORT="5556" JOBS="15"
LOCKFILE="" SIM_PATH="" LOG_DIR=""
NO_SPLIT="false" VERBOSE="false"
STALL=""

# Tunables (rarely overridden).
WATCH_INTERVAL="${WATCH_INTERVAL:-5}"   # seconds between log-stall checks
GRACE_SECS="${GRACE_SECS:-30}"          # wait after SIGINT before SIGKILL
# tt-exalens readiness marker as it appears in the PYTEST log (the "[4B MODE]"
# string lives only in the separate tt-exalens.log, not here). The helper logs
# "tt-exalens ready (PID …)" to the pytest stream; match that (keep [4B MODE] as
# a fallback for 4B-mode configs that surface it).
READY_RE='tt-exalens ready|\[4B MODE\]'
EMU_HOST="${EMU_HOST:-${SSH_MACHINE_NAME:-soc-l-12}}"
NNG_LOCAL_BASE="5555"                   # local NNG bind (infra-forwarded; fixed)
DBD_BASE="54910"                        # NNG_SOCKET_ADDR debuda port (fixed)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worktree)  WORKTREE="$2";  shift 2 ;;
    --arch)      ARCH="$2";      shift 2 ;;
    --test)      TEST_FILE="$2"; shift 2 ;;
    --maxfail)   MAXFAIL="$2";   shift 2 ;;
    --k)         K_FILTER="$2";  shift 2 ;;
    --test-id)   TEST_ID="$2";   shift 2 ;;
    --port)      PORT="$2";      shift 2 ;;
    --jobs)      JOBS="$2";      shift 2 ;;
    --lock)      LOCKFILE="$2";  shift 2 ;;
    --sim-path)  SIM_PATH="$2";  shift 2 ;;
    --log-dir)   LOG_DIR="$2";   shift 2 ;;
    --stall)     STALL="$2";     shift 2 ;;
    --no-split)  NO_SPLIT="true"; shift ;;
    --verbose|-v) VERBOSE="true"; shift ;;
    # Deprecated no-ops: the watcher bounds the run, so there is no timeout or
    # poll budget. Accepted (and ignored) so pre-rewrite callers don't hard-fail.
    --timeout|--poll-budget) shift 2 ;;
    --help|-h)   sed -n 's/^# \{0,1\}//p' "$0" | head -70; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; echo "Run with --help for usage." >&2; exit 4 ;;
  esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────

_vlog() { [[ "$VERBOSE" == "true" ]] && echo "[run_test] $*" >&2; return 0; }

# Activate the venv only if it exists (external setup); else use the ambient
# python (tt-metal Docker image, deps installed system-wide).
# shellcheck disable=SC1091
_activate_venv() { [[ -f "${VENV}/bin/activate" ]] && source "${VENV}/bin/activate"; return 0; }

# Validate args and derive VENV, TEST_DIR, SIM_PATH, LOCKFILE, MODE, STALL, the
# NNG channel, and the build-stamp identity. Exits on error.
_validate() {
  local errors=0
  [[ -z "$WORKTREE"  ]] && { echo "ERROR: --worktree is required" >&2; ((errors++)); }
  [[ -z "$ARCH"      ]] && { echo "ERROR: --arch is required"     >&2; ((errors++)); }
  [[ -z "$TEST_FILE" ]] && { echo "ERROR: --test is required"     >&2; ((errors++)); }
  [[ $errors -gt 0 ]] && exit 4

  VENV="${WORKTREE}/tests/.venv"
  case "$ARCH" in
    blackhole|wormhole) TEST_DIR="${WORKTREE}/tests/python_tests" ;;
    *)                  TEST_DIR="${WORKTREE}/tests/python_tests/${ARCH}" ;;
  esac
  [[ -d "$TEST_DIR" ]] || { echo "ERROR: test directory not found: ${TEST_DIR}" >&2; exit 3; }
  [[ -f "${TEST_DIR}/${TEST_FILE}" ]] || { echo "ERROR: test file not found: ${TEST_DIR}/${TEST_FILE}" >&2; exit 3; }

  [[ -z "$SIM_PATH" ]] && SIM_PATH="/proj_sw/user_dev/${USER}/tt-umd-simulators/build/emu-${ARCH}-1x3"
  # One global lock for every invocation on this host, regardless of arch.
  [[ -z "$LOCKFILE" ]] && LOCKFILE="/tmp/tt-llk-test.lock"

  case "$ARCH" in
    blackhole|wormhole) MODE="hardware"  ;;
    *)                  MODE="simulator" ;;
  esac

  # Hang threshold: emulator gets the post-ready stall; silicon keeps the larger
  # default. Explicit --stall / HANG_STALL wins.
  if [[ -z "$STALL" ]]; then
    if [[ -n "${HANG_STALL:-}" ]]; then STALL="$HANG_STALL"
    elif [[ "$MODE" == "simulator" ]]; then STALL=180
    else STALL=300; fi
  fi

  # --collect-only / count emit node-ids relative to the pytest rootdir, so a
  # quasar id carries a leading "quasar/". We cd into the arch subdir before
  # running, so strip that prefix or pytest collects 0 items.
  [[ -n "$TEST_ID" ]] && TEST_ID="${TEST_ID#${ARCH}/}"

  # Build-stamp identity. SRC_ID fingerprints THIS run's source (worktree + LLK
  # arch tree + test file). ARTKEY is worktree-independent (peers building the
  # same target share it) so their differing SRC_IDs compare. STAMP_DIR lives
  # outside the wiped build cache so a stamp survives a peer's wipe.
  local _llk_dir
  case "$ARCH" in
    quasar)    _llk_dir="tt_llk_quasar" ;;
    blackhole) _llk_dir="tt_llk_blackhole" ;;
    wormhole)  _llk_dir="tt_llk_wormhole_b0" ;;
    *)         _llk_dir="tt_llk_${ARCH}" ;;
  esac
  SRC_ID=$( { printf '%s\n' "$WORKTREE"; find "${WORKTREE}/${_llk_dir}" "${TEST_DIR}/${TEST_FILE}" -type f -printf '%P %s %T@\n' 2>/dev/null | sort; } | sha256sum | cut -c1-16 )
  ARTKEY=$( printf '%s' "${ARCH}|${TEST_FILE}|${TEST_ID}|${K_FILTER}|${NO_SPLIT}|${MAXFAIL}" | sha256sum | cut -c1-16 )
  STAMP_DIR="/tmp/tt-llk-build-stamps-${ARCH}"
  STAMP="${STAMP_DIR}/${ARTKEY}"

  # Codegen infra (symlinked into each worktree).
  REAP="${WORKTREE}/codegen/scripts/reap_stale_emu.sh"
  TRIAGE="${WORKTREE}/.claude/scripts/llk_triage.py"
  RUN_TAG="ttllk_${ARCH}_$$"

  # NNG_SOCKET_ADDR (debuda) is the single infra-forwarded channel — keep the
  # shell value if set, else derive from the host in the shell's addr or hostname.
  local host
  if [[ "${NNG_SOCKET_ADDR:-}" =~ ^tcp://([^:]+): ]]; then host="${BASH_REMATCH[1]}"; else host="$(hostname)"; fi
  NNG_ADDR="${NNG_SOCKET_ADDR:-tcp://${host}:${DBD_BASE}}"
  NNG_LOCAL="${NNG_SOCKET_LOCAL_PORT:-$NNG_LOCAL_BASE}"
}

# SFPI (the RISC-V toolchain) is mandatory to compile. Fetch it if absent.
_ensure_sfpi() {
  [[ -d "${WORKTREE}/tests/sfpi/compiler/bin" ]] && return 0
  echo "[run_test] SFPI missing — fetching (CHIP_ARCH=${ARCH} ./setup_testing_env.sh)" >&2
  ( cd "${WORKTREE}/tests" && CHIP_ARCH="$ARCH" ./setup_testing_env.sh ) >&2 2>&1
  [[ -d "${WORKTREE}/tests/sfpi/compiler/bin" ]] || { echo "[run_test] SFPI still missing after setup" >&2; return 3; }
  return 0
}

# The pytest target selector (file, -k filter, or a single parametrize id).
_build_target() {
  TARGET=()
  if   [[ -n "$TEST_ID"  ]]; then TARGET=("$TEST_ID")
  elif [[ -n "$K_FILTER" ]]; then TARGET=(-k "$K_FILTER" "$TEST_FILE")
  else                            TARGET=("$TEST_FILE"); fi
}

_emit_verdict() {
  local code="$1" phase="$2" v
  case "$code" in
    0) v=PASS ;; 1) v=FAIL ;; 2) v=COMPILE_FAIL ;; 3) v=ENV_ERROR ;; 4) v=BAD_ARGS ;; 5) v=HANG ;; *) v="EXIT_${code}" ;;
  esac
  echo "=== RUN_LLK_TESTS_VERDICT === ${v} (exit ${code}, phase=${phase}, test=${TEST_FILE:-?}, arch=${ARCH:-?})" >&2
}

# ── Producer (compile) ─────────────────────────────────────────────────────────

# Parallel compile with the transient parallel-build-setup retry. The producer's
# xdist workers each rmtree+recreate the shared build dir at startup, which under
# load can race into a FileNotFoundError INTERNALERROR — not a real compile error.
# Retry on that signature; treat anything else as a genuine compile failure.
# Returns pytest's exit code.
_producer() {
  local plog; plog="$(mktemp "${TMPDIR:-/tmp}/tt-llk-prod.XXXXXX")"
  local prc=1 attempt
  for attempt in 1 2 3; do
    : > "$plog"
    ( CHIP_ARCH="$ARCH" pytest --compile-producer -n "$JOBS" -x "${TARGET[@]}" ) >"$plog" 2>&1
    prc=$?
    [[ -n "$LOG_DIR" ]] && { mkdir -p "$LOG_DIR"; cat "$plog" >> "${LOG_DIR}/compile.log"; }
    cat "$plog" >&2
    [[ $prc -eq 0 ]] && break
    if grep -q "create_build_directories" "$plog" && grep -q "FileNotFoundError" "$plog"; then
      echo "[run_test] transient parallel-build-setup race (attempt ${attempt}/3); retrying" >&2
      sleep 3; continue
    fi
    break   # genuine compile error — do not retry
  done
  rm -f "$plog"
  return "$prc"
}

# ── Silicon (BH/WH) hang cleanup ───────────────────────────────────────────────

# Free the device handle, dump LLK triage while the Tensix is still wedged, then
# reset. Triage is skipped if the script is absent.
_hw_hang_cleanup() {
  pkill -9 -f "pytest.*--compile-consumer" 2>/dev/null || true
  if [[ -f "$TRIAGE" ]]; then
    echo "--- llk-triage ---" >&2
    timeout 60 python3 "$TRIAGE" --arch "$ARCH" >&2 2>&1 || true
    echo "--- end llk-triage ---" >&2
  fi
  command -v tt-smi >/dev/null 2>&1 && { echo "[run_test] tt-smi -r" >&2; tt-smi -r >&2 2>&1 || true; }
}

# ── Consumer (run on device/emulator) with hang watchdog ───────────────────────

# Run pytest in the background, watch its log for a stall, and on a hang send
# SIGINT so conftest's handler tears tt-exalens down gracefully (releasing the
# remote emulator), escalating to SIGKILL if pytest ignores it. Sets/clears the
# global CONSUMER_PID (the EXIT trap uses it). Returns the classified exit code.
_run_consumer() {
  local log; log="$(mktemp "${TMPDIR:-/tmp}/tt-llk-run.XXXXXX")"
  local hangflag="${log}.hang"

  local -a flags=(-rN "--maxfail=${MAXFAIL}")
  [[ "$MODE" == "simulator" ]] && flags+=(--run-simulator "--port=${PORT}")
  [[ "$NO_SPLIT" == "false" ]] && flags+=(--compile-consumer)

  if [[ "$MODE" == "simulator" ]]; then
    export NNG_SOCKET_ADDR="$NNG_ADDR" NNG_SOCKET_LOCAL_PORT="$NNG_LOCAL" NNG_SOCKET_NAME="$RUN_TAG"
    export TT_UMD_SIMULATOR_PATH="$SIM_PATH"
    # Free this run's port before booting tt-exalens.
    local stale; stale="$(lsof -ti :"$PORT" 2>/dev/null || true)"
    [[ -n "$stale" ]] && echo "$stale" | xargs -r kill -9 2>/dev/null || true
    pkill -9 -f "tt-exalens.*--port=${PORT}" 2>/dev/null || true
    sleep 1
  fi

  ( CHIP_ARCH="$ARCH" pytest "${flags[@]}" "${TARGET[@]}" ) >>"$log" 2>&1 &
  CONSUMER_PID=$!

  # Watchdog: a healthy run keeps emitting progress lines in EVERY phase — during
  # boot the server prints "still waiting" every 10s, then per-variant results — so
  # a log that stops advancing for STALL seconds means it wedged, no matter the
  # phase. Armed from the start (boot wedges included); the readiness marker is used
  # only to CLASSIFY the outcome afterwards (pre-ready stall = ENV), not to arm.
  (
    while kill -0 "$CONSUMER_PID" 2>/dev/null; do
      sleep "$WATCH_INTERVAL"
      local now mtime; now="$(date +%s)"; mtime="$(stat -c %Y "$log" 2>/dev/null || echo "$now")"
      if [[ $((now - mtime)) -ge "$STALL" ]]; then
        : > "$hangflag"
        # Graceful: conftest turns SIGINT/SIGTERM into KeyboardInterrupt →
        # pytest_sessionfinish/atexit → ExalensServer.stop() sends `exit`.
        kill -INT "$CONSUMER_PID" 2>/dev/null
        local waited=0
        while kill -0 "$CONSUMER_PID" 2>/dev/null && [[ $waited -lt $GRACE_SECS ]]; do
          sleep 1; waited=$((waited + 1))
        done
        # Ignored the signal (wedged in a C call) → hard-kill the tree.
        if kill -0 "$CONSUMER_PID" 2>/dev/null; then
          for p in $(pgrep -P "$CONSUMER_PID" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
          kill -9 "$CONSUMER_PID" 2>/dev/null
        fi
        break
      fi
    done
  ) &
  local watch_pid=$!

  wait "$CONSUMER_PID"; local rc=$?
  kill "$watch_pid" 2>/dev/null; wait "$watch_pid" 2>/dev/null

  [[ -n "$LOG_DIR" ]] && { mkdir -p "$LOG_DIR"; cat "$log" >> "${LOG_DIR}/run.log" 2>/dev/null; }
  tail -80 "$log" >&2

  # Classify. Order matters: never-ready (infra) is checked before FAIL because a
  # kernel cannot run before the emulator is up.
  local code
  if [[ -f "$hangflag" ]] && [[ "$MODE" == "simulator" ]] && ! grep -qE "$READY_RE" "$log" 2>/dev/null; then
    # Stalled before tt-exalens ever reported ready → a boot wedge, not a kernel
    # hang. Transient (emulator congestion) → ENV so the caller may retry.
    echo "[run_test] ENV: stalled before tt-exalens became ready (boot wedge)" >&2
    [[ -x "$REAP" ]] && bash "$REAP" --arch "$ARCH" --emu-host "$EMU_HOST" --force >&2 2>&1 || true
    code=3
  elif [[ -f "$hangflag" ]]; then
    echo "[run_test] HANG: no output for ${STALL}s" >&2
    if [[ "$MODE" == "simulator" ]]; then
      [[ -x "$REAP" ]] && bash "$REAP" --arch "$ARCH" --emu-host "$EMU_HOST" --force >&2 2>&1 || true
    else
      _hw_hang_cleanup
    fi
    code=5
  elif [[ "$MODE" == "simulator" && $rc -ne 0 ]] && ! grep -qE "$READY_RE" "$log" 2>/dev/null; then
    echo "[run_test] ENV: tt-exalens never became ready" >&2
    [[ -x "$REAP" ]] && bash "$REAP" --arch "$ARCH" --emu-host "$EMU_HOST" --force >&2 2>&1 || true
    code=3
  elif [[ "$MODE" == "hardware" && $rc -ne 0 ]] && grep -qF "TENSIX TIMED OUT" "$log" 2>/dev/null; then
    echo "[run_test] HANG: TENSIX TIMED OUT" >&2
    _hw_hang_cleanup
    code=5
  elif [[ $rc -ne 0 ]] && grep -qiE "No Tenstorrent devices? (were|was)? ?detected|No Tenstorrent devices" "$log" 2>/dev/null; then
    echo "[run_test] ENV: no Tenstorrent device detected (CHIP_ARCH / device access)" >&2
    code=3
  elif [[ $rc -eq 0 ]]; then
    code=0
  else
    code=1
  fi

  CONSUMER_PID=""
  rm -f "$log" "$hangflag" 2>/dev/null
  return "$code"
}

# ── count / compile (lock-free) ────────────────────────────────────────────────

_do_count() {
  _validate; _activate_venv; cd "$TEST_DIR" || { echo "0"; return 3; }
  local out rc=0
  if [[ -n "$K_FILTER" ]]; then
    out="$(CHIP_ARCH="$ARCH" pytest --compile-producer --co -q -k "$K_FILTER" "$TEST_FILE" 2>&1)" || rc=$?
  else
    out="$(CHIP_ARCH="$ARCH" pytest --compile-producer --co -q "$TEST_FILE" 2>&1)" || rc=$?
  fi
  printf '%s\n' "$out" >&2
  [[ $rc -ne 0 ]] && { echo "0"; return "$rc"; }
  printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -1 | grep -oE '^[0-9]+' || echo "0"
}

_do_compile() {
  _validate; _activate_venv; _ensure_sfpi || return 3
  _build_target; cd "$TEST_DIR" || return 3
  _vlog "compile ${TEST_FILE} (arch=${ARCH}, -n ${JOBS})"
  _producer; local rc=$?
  [[ $rc -eq 0 ]] && { mkdir -p "$STAMP_DIR" 2>/dev/null; printf '%s' "$SRC_ID" > "$STAMP"; return 0; }
  return 2
}

# ── simulate / run (under the global lock) ─────────────────────────────────────

# Acquire the global lock, rebuild under it when the stamp is not ours (or forced),
# then run the consumer. The producer and consumer run back-to-back without
# releasing the lock, so the ELFs consumed are exactly the ones just produced.
_run_under_lock() {
  local force="$1"
  _validate; _activate_venv; _ensure_sfpi || return 3
  _build_target; cd "$TEST_DIR" || return 3

  exec 9>>"$LOCKFILE" || { echo "ERROR: cannot open lock ${LOCKFILE}" >&2; return 3; }
  _vlog "waiting for global lock ${LOCKFILE}"
  flock 9                       # unbounded — wait in line
  _vlog "acquired lock"

  # Pre-flight reap under the lock: any live emu job now is an orphan from a run
  # whose peer died non-gracefully. Clear it before booting ours.
  if [[ "$MODE" == "simulator" && -x "$REAP" ]]; then
    bash "$REAP" --arch "$ARCH" --emu-host "$EMU_HOST" --force >&2 2>&1 || true
  fi

  # Build under the lock if forced or the stamp is not ours (a peer recompiled).
  # --no-split compiles inside the consumer, so it is skipped here.
  if [[ "$NO_SPLIT" == "false" ]]; then
    local need="$force"
    [[ "$(cat "$STAMP" 2>/dev/null)" != "$SRC_ID" ]] && need=1
    if [[ "$need" == "1" ]]; then
      _vlog "building under lock (have=$(cat "$STAMP" 2>/dev/null) want=${SRC_ID} force=${force})"
      _producer || return 2
      mkdir -p "$STAMP_DIR" 2>/dev/null; printf '%s' "$SRC_ID" > "$STAMP"
    else
      _vlog "reusing build (stamp matches ${SRC_ID})"
    fi
  fi

  _run_consumer; return $?
  # The lock (fd 9) is released when the script exits.
}

# ── Cleanup trap ───────────────────────────────────────────────────────────────

CONSUMER_PID=""
# On any script exit — including a harness SIGTERM/SIGINT — if the consumer is
# still alive we are dying abnormally: tear it down gracefully so tt-exalens
# releases the remote emulator, escalate + reap if it ignores the signal. Normal
# completion clears CONSUMER_PID, so this no-ops.
_cleanup() {
  if [[ -n "${CONSUMER_PID:-}" ]] && kill -0 "$CONSUMER_PID" 2>/dev/null; then
    kill -INT "$CONSUMER_PID" 2>/dev/null
    local waited=0
    while kill -0 "$CONSUMER_PID" 2>/dev/null && [[ $waited -lt $GRACE_SECS ]]; do sleep 1; waited=$((waited + 1)); done
    kill -9 "$CONSUMER_PID" 2>/dev/null
    if [[ "${MODE:-}" == "simulator" && -x "${REAP:-}" ]]; then
      bash "$REAP" --arch "$ARCH" --emu-host "$EMU_HOST" --force >/dev/null 2>&1 || true
    fi
  fi
}
trap _cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# ── Dispatch ──────────────────────────────────────────────────────────────────

_rc=0
case "$CMD" in
  count)    _do_count       ; _rc=$? ;;
  compile)  _do_compile     ; _rc=$? ;;
  simulate) _run_under_lock 0 ; _rc=$? ;;
  run)      _run_under_lock 1 ; _rc=$? ;;
  help|--help|-h) sed -n 's/^# \{0,1\}//p' "$0" | head -70; exit 0 ;;
  "") echo "ERROR: no command. Use: count | compile | simulate | run" >&2; exit 4 ;;
  *)  echo "ERROR: unknown command '${CMD}'. Use: count | compile | simulate | run" >&2; exit 4 ;;
esac

# count's stdout contract is "just the integer" — no verdict line.
case "$CMD" in compile|simulate|run) _emit_verdict "$_rc" "$CMD" ;; esac
exit "$_rc"
