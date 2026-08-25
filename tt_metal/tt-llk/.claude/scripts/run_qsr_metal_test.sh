#!/usr/bin/env bash
# Run one unit_tests_llk filter or an explicit test command against Quasar
# Aether VCS/emulation.
#
# The caller owns the build and supplies a fresh TT_METAL_CACHE. This wrapper
# owns only the scarce remote Aether execution: backend selection, the
# cross-compute-host lock, preflight orphan cleanup, and failure cleanup.
set -u

BIN=""
FILTER=""
TT_METAL_HOME_ARG=""
CACHE=""
LOG_DIR=""
RUN_TIMEOUT="${TIMEOUT:-1200}"
COMMAND=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bin)           BIN="$2";               shift 2 ;;
    --gtest-filter)  FILTER="$2";            shift 2 ;;
    --tt-metal-home) TT_METAL_HOME_ARG="$2"; shift 2 ;;
    --cache)         CACHE="$2";             shift 2 ;;
    --log-dir)       LOG_DIR="$2";           shift 2 ;;
    --timeout)       RUN_TIMEOUT="$2";        shift 2 ;;
    --)              shift; COMMAND=("$@"); break ;;
    -h|--help)
      sed -n 's/^# \{0,1\}//p' "$0" | head -24
      exit 0
      ;;
    *) echo "ERROR: unknown option: $1" >&2; exit 4 ;;
  esac
done

[[ -n "$TT_METAL_HOME_ARG" ]] || { echo "ERROR: --tt-metal-home is required" >&2; exit 4; }
[[ -n "$CACHE" ]] || { echo "ERROR: --cache is required" >&2; exit 4; }
if [[ ${#COMMAND[@]} -gt 0 ]]; then
  [[ -z "$BIN" && -z "$FILTER" ]] || {
    echo "ERROR: explicit command cannot be combined with --bin/--gtest-filter" >&2
    exit 4
  }
  command -v "${COMMAND[0]}" >/dev/null 2>&1 || {
    echo "ERROR: command is not executable: ${COMMAND[0]}" >&2
    exit 3
  }
else
  [[ -n "$BIN" ]] || { echo "ERROR: --bin is required" >&2; exit 4; }
  [[ -n "$FILTER" ]] || { echo "ERROR: --gtest-filter is required" >&2; exit 4; }
  [[ -x "$BIN" ]] || { echo "ERROR: test binary is not executable: $BIN" >&2; exit 3; }
  COMMAND=("$BIN" "--gtest_filter=$FILTER")
fi

QSR_SIM_BACKEND="${QSR_SIM_BACKEND:-emu}"
case "$QSR_SIM_BACKEND" in
  emu|emulator)
    QSR_SIM_BACKEND="emu"
    SIM_PATH="${QSR_EMU_SIM_PATH:-/proj_sw/user_dev/${USER}/tt-umd-simulators/build/emu-quasar-1x3}"
    ;;
  vcs)
    SIM_PATH="${QSR_VCS_SIM_PATH:-/proj_sw/user_dev/${USER}/tt-umd-simulators/build/vcs-quasar-1x3}"
    ;;
  *)
    echo "ERROR: QSR_SIM_BACKEND must be emu or vcs, got '$QSR_SIM_BACKEND'" >&2
    exit 3
    ;;
esac
[[ -d "$SIM_PATH" ]] || { echo "ERROR: missing Quasar simulator path: $SIM_PATH" >&2; exit 3; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REAP="$LLK_ROOT/codegen/scripts/reap_stale_emu.sh"
LOCKFILE="${QSR_AETHER_LOCK:-/tmp/tt-llk-test.lock}"
EMU_HOST="${EMU_HOST:-${QSR_AETHER_HOST:-${SSH_MACHINE_NAME:-soc-l-12}}}"

mkdir -p "$(dirname "$LOCKFILE")" 2>/dev/null ||
  { echo "ERROR: cannot create lock directory for $LOCKFILE" >&2; exit 3; }
[[ -z "$LOG_DIR" ]] || mkdir -p "$LOG_DIR"

exec 9>>"$LOCKFILE" || { echo "ERROR: cannot open lock $LOCKFILE" >&2; exit 3; }
echo "[qsr-metal] waiting for Aether lock $LOCKFILE" >&2
flock 9
echo "[qsr-metal] acquired Aether lock (backend=$QSR_SIM_BACKEND)" >&2

cleanup_needed=true
cleanup() {
  if [[ "$cleanup_needed" == true && -x "$REAP" ]]; then
    bash "$REAP" --arch quasar --emu-host "$EMU_HOST" --lock "$LOCKFILE" --force >&2 2>&1 || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Anything still alive while the globally shared lock is ours is an orphan from
# a dead previous owner.
[[ -x "$REAP" ]] &&
  bash "$REAP" --arch quasar --emu-host "$EMU_HOST" --lock "$LOCKFILE" --force >&2 2>&1 || true

run_log="${LOG_DIR:+$LOG_DIR/metal_run_quasar.log}"
set +e
if [[ -n "$run_log" ]]; then
  env \
    TT_METAL_HOME="$TT_METAL_HOME_ARG" \
    TT_METAL_CACHE="$CACHE" \
    TT_METAL_SIMULATOR="$SIM_PATH" \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    SSH_MACHINE_NAME="$EMU_HOST" \
    timeout "$RUN_TIMEOUT" "${COMMAND[@]}" 2>&1 | tee -a "$run_log"
  rc=${PIPESTATUS[0]}
else
  env \
    TT_METAL_HOME="$TT_METAL_HOME_ARG" \
    TT_METAL_CACHE="$CACHE" \
    TT_METAL_SIMULATOR="$SIM_PATH" \
    TT_METAL_SLOW_DISPATCH_MODE=1 \
    SSH_MACHINE_NAME="$EMU_HOST" \
    timeout "$RUN_TIMEOUT" "${COMMAND[@]}"
  rc=$?
fi
set -e

# A normal completion asks UMD/Aether to exit. On any nonzero result, also run
# the idempotent reaper before releasing the shared lock so no following solve
# can race the cleanup.
if [[ $rc -eq 0 ]]; then
  cleanup_needed=false
else
  cleanup
  cleanup_needed=false
fi
trap - EXIT INT TERM
exit "$rc"
