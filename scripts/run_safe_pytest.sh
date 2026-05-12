#!/bin/bash
# run_safe_pytest.sh - Cooperative device-aware test runner
#
# Uses flock to serialize device access across multiple agents/terminals.
# Uses TT_METAL_OPERATION_TIMEOUT_SECONDS for precise hang detection at the
# dispatch layer (does not penalize setup/compilation time).
# Automatically resets the device after hangs, ensuring the next runner
# always gets a clean device.
#
# Simulator mode (TT_METAL_SIMULATOR set):
#   Exports TT_METAL_SLOW_DISPATCH_MODE=1 and TT_METAL_DISABLE_SFPLOADMACRO=1.
#   Skips flock, device resets, and triage (these require real hardware).
#   No hang protection — sim runs at kHz, so wall-clock timeouts are meaningless.
#
# Usage: scripts/run_safe_pytest.sh [--dev] [--run-all] <test_path> [extra_pytest_args...]
#
# Options:
#   --dev       Enables polling watcher (NoC sanitizer, waypoints, CB
#               sanitization), lightweight ebreak asserts, and auto-triage
#               on hang with full triage + watcher log dump.
#   --run-all   Run all tests instead of stopping on first failure (-x).
#               Useful for eval scoring where you need full pass/fail counts.
#
# Modes:
#   default  - Dispatch timeout only. Lean, no debug overhead.
#   --dev    - Debug mode with watcher, asserts, and triage (see above).
#
# Exit codes:
#   0 - All tests passed
#   1 - Test failure (normal pytest failure, no hang)
#   2 - Hang detected (dispatch timeout fired)
#   3 - Setup error (missing args, etc.)

set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISPATCH_TIMEOUT=5
TRIAGE_SCRIPT="${REPO_DIR}/tools/tt-triage.py"
WATCHER_LOG="${REPO_DIR}/generated/watcher/watcher.log"
TRIAGE_JSON_DIR="${REPO_DIR}/generated/tt-triage"
LOCK_FILE="/tmp/tt-device.lock"
DIRTY_FLAG="/tmp/tt-device.dirty"
TRIAGE_LOG="/tmp/safe-pytest-triage-$$.log"
TRIAGE_JSON="${TRIAGE_JSON_DIR}/triage.json"

# --- Device-lock contention profiling ---
# When $TT_DEVICE_TIMING_LOG is set, on EXIT we append one JSON line:
#   {source, pid, started_at_ms, wait_ms, run_ms, test_path, exit_code}
# wait_ms = script entry → flock acquired (contention)
# run_ms  = flock acquired → script exit  (device occupied)
# Skipped on sim mode (no flock contention) and when the script exits before
# acquiring the lock (TT_TIMING_LOCK_ACQUIRED_MS stays 0).
TT_TIMING_ENTRY_MS=$(date +%s%3N)
TT_TIMING_LOCK_ACQUIRED_MS=0
TT_TIMING_SOURCE="run_safe_pytest"
TT_TIMING_TEST_PATH=""

_emit_device_timing() {
    local ec=$?
    if [[ -n "${TT_DEVICE_TIMING_LOG:-}" && "$TT_TIMING_LOCK_ACQUIRED_MS" -ne 0 ]]; then
        local end_ms wait_ms run_ms log_dir esc_path
        end_ms=$(date +%s%3N)
        wait_ms=$(( TT_TIMING_LOCK_ACQUIRED_MS - TT_TIMING_ENTRY_MS ))
        run_ms=$(( end_ms - TT_TIMING_LOCK_ACQUIRED_MS ))
        log_dir="$(dirname "$TT_DEVICE_TIMING_LOG")"
        [[ -n "$log_dir" ]] && mkdir -p "$log_dir" 2>/dev/null
        # JSON-escape test_path: backslash first, then double-quote.
        esc_path="${TT_TIMING_TEST_PATH//\\/\\\\}"
        esc_path="${esc_path//\"/\\\"}"
        printf '{"source":"%s","pid":%d,"started_at_ms":%s,"wait_ms":%d,"run_ms":%d,"test_path":"%s","exit_code":%d}\n' \
            "$TT_TIMING_SOURCE" "$$" "$TT_TIMING_ENTRY_MS" "$wait_ms" "$run_ms" "$esc_path" "$ec" \
            >> "$TT_DEVICE_TIMING_LOG" 2>/dev/null || true
    fi
    return $ec
}
trap _emit_device_timing EXIT

# --- Detect simulator mode ---
SIM_MODE=false
if [[ -n "${TT_METAL_SIMULATOR:-}" ]]; then
    SIM_MODE=true
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export TT_METAL_DISABLE_SFPLOADMACRO=1
fi

# --- Parse flags ---
DEV_MODE=false
FAIL_FAST=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --run-all)
            FAIL_FAST=false
            shift
            ;;
        *)
            break
            ;;
    esac
done

# --- Argument validation ---
if [[ $# -eq 0 ]]; then
    echo "SAFE_PYTEST_ERROR: No test path provided" >&2
    echo "Usage: scripts/run_safe_pytest.sh [--dev] [--run-all] <test_path> [extra_pytest_args...]" >&2
    exit 3
fi

TEST_PATH="$1"
TT_TIMING_TEST_PATH="$TEST_PATH"
shift

# --- Acquire flock (hardware only) ---
if [[ "$SIM_MODE" == false ]]; then
    exec 9>"$LOCK_FILE"

    echo "SAFE_PYTEST: Waiting for device lock..." >&2
    flock 9
    TT_TIMING_LOCK_ACQUIRED_MS=$(date +%s%3N)
    echo "SAFE_PYTEST: Device lock acquired" >&2

    # --- Check if device needs reset from previous hang ---
    if [[ -f "$DIRTY_FLAG" ]]; then
        echo "SAFE_PYTEST: Device marked dirty from previous hang, resetting..." >&2
        if ! tt-smi -r; then
            echo "SAFE_PYTEST_ERROR: Device reset (tt-smi -r) failed" >&2
            exit 3
        fi
        rm -f "$DIRTY_FLAG"
        echo "SAFE_PYTEST: Device reset complete" >&2
    fi
fi

# --- Setup environment ---
cd "$REPO_DIR"
if [[ -f python_env/bin/activate ]]; then
    if ! source python_env/bin/activate; then
        echo "SAFE_PYTEST: WARNING: Failed to activate python_env virtual environment" >&2
    fi
else
    echo "SAFE_PYTEST: WARNING: python_env not found; using system Python" >&2
fi

# --- Hang detection setup (hardware only) ---
# On timeout, the dispatch layer runs tt-triage. Fires only on actual hang —
# zero overhead for passing tests. On sim there is no hang detection because
# wall-clock timeouts are meaningless at kHz clock speeds.
rm -f "$TRIAGE_LOG"
MISSING_TTEXALENS=false
if [[ "$SIM_MODE" == false ]]; then
    export TT_METAL_OPERATION_TIMEOUT_SECONDS="$DISPATCH_TIMEOUT"
    # Requires tt-exalens: uv pip install -r tools/triage/requirements.txt
    # Defer the missing-tool warning to EXIT via trap — otherwise it gets buried
    # in pytest / triage output and users never see it.
    if ! python3 -c "import ttexalens" 2>/dev/null; then
        MISSING_TTEXALENS=true
    fi
    mkdir -p "${TRIAGE_JSON_DIR}"
    export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 ${TRIAGE_SCRIPT} --disable-progress --skip-version-check --json-path=${TRIAGE_JSON} > ${TRIAGE_LOG} 2>&1"
fi

emit_missing_ttexalens_warning() {
    if [[ "$MISSING_TTEXALENS" == true ]]; then
        echo "" >&2
        echo "SAFE_PYTEST: WARNING: tt-exalens not installed — triage on hang is unavailable." >&2
        echo "SAFE_PYTEST: Install with: uv pip install -r tools/triage/requirements.txt" >&2
    fi
}
trap emit_missing_ttexalens_warning EXIT

if [[ "$DEV_MODE" == true ]]; then
    # Lightweight asserts: compiles ASSERT() as ebreak, halting the core at the
    # exact instruction. The dispatch timeout then fires and runs triage, which
    # captures callstacks from ALL cores — showing both the assert site and any
    # cores blocked waiting on the halted one.
    export TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1

    # LLK asserts: enables LLK_ASSERT() in the compute API / LLK layer.
    # Catches invalid hardware configurations, wrong unpack/pack parameters,
    # and API misuse deep in the compute pipeline. Also compiles as ebreak.
    export TT_METAL_LLK_ASSERTS=1

    # Polling watcher: enables all device-side instrumentation (NoC sanitizer,
    # waypoints, CB sanitization) with the host polling thread. Recent watcher
    # server improvements (t=0 sampling, 100ms quanta) minimize overhead for
    # short tests. Watcher log is dumped on hang for full diagnostic context.
    #
    # WATCHER_DISABLE_ASSERT disables the watcher's own assert mechanism (which
    # would conflict with the lightweight ebreak asserts above). We want ebreak
    # asserts to halt the core so triage can capture callstacks from all cores,
    # rather than having watcher handle asserts independently.
    export TT_METAL_WATCHER=1
    export TT_METAL_WATCHER_NOINLINE=1
    export TT_METAL_WATCHER_DISABLE_ASSERT=1
    export TT_METAL_WATCHER_DISABLE_DISPATCH=1

    if [[ "$SIM_MODE" == true ]]; then
        echo "SAFE_PYTEST: [sim+dev] asserts=ebreak llk_asserts=ON watcher=polling (no hang detection on sim)" >&2
    else
        echo "SAFE_PYTEST: [dev] asserts=ebreak llk_asserts=ON watcher=polling triage=ON timeout=${DISPATCH_TIMEOUT}s" >&2
    fi
elif [[ "$SIM_MODE" == true ]]; then
    echo "SAFE_PYTEST: [sim] no hang detection" >&2
else
    echo "SAFE_PYTEST: dispatch_timeout=${DISPATCH_TIMEOUT}s" >&2
fi
echo "SAFE_PYTEST: pytest ${TEST_PATH} $*" >&2
echo "========================================" >&2

# --- Mark device dirty before running tests (hardware only) ---
# Pessimistic: assume the device will get corrupted. If the script is killed at any
# point (SIGKILL, OOM, etc.), the flag persists and the next runner will reset.
# Cleared on clean exit or after a successful inline reset.
if [[ "$SIM_MODE" == false ]]; then
    touch "$DIRTY_FLAG"
fi

# --- Run pytest ---
# -x: stop on first failure (avoids running tests after a hang bricks the device)
# --run-all: skip -x to get full pass/fail counts (for eval scoring)
PYTEST_CMD=(pytest "${TEST_PATH}")
if [[ "$FAIL_FAST" == true ]]; then
    PYTEST_CMD+=(-x)
fi
PYTEST_CMD+=("$@")

# Signal handling: if this script is killed (e.g. parent process gets SIGTERM
# and we get reparented to init, or a watchdog kills us), forward SIGKILL to
# pytest and its descendants. Without this, pytest is orphaned with fd 9 ->
# /tmp/tt-device.lock and /dev/tenstorrent/* held, blocking all future runs.
CHILD_PID=
_signal_cleanup() {
    local sig=$1
    echo "" >&2
    echo "SAFE_PYTEST: Caught SIG${sig} — killing pytest, marking device dirty" >&2
    [[ "$SIM_MODE" == false ]] && touch "$DIRTY_FLAG" 2>/dev/null
    if [[ -n "$CHILD_PID" ]]; then
        pkill -KILL -P "$CHILD_PID" 2>/dev/null || true
        kill -KILL "$CHILD_PID" 2>/dev/null || true
    fi
    pkill -KILL -P $$ 2>/dev/null || true
    exit 143
}
trap '_signal_cleanup TERM' SIGTERM
trap '_signal_cleanup HUP'  SIGHUP
trap '_signal_cleanup INT'  SIGINT

# Run pytest in background so `wait` can be interrupted by a signal. Bash
# blocks signal delivery while a synchronous foreground command is running.
"${PYTEST_CMD[@]}" &
CHILD_PID=$!
wait "$CHILD_PID"
EXIT_CODE=$?
CHILD_PID=

echo "========================================" >&2

# --- Handle result ---
if [[ $EXIT_CODE -eq 0 ]]; then
    rm -f "$DIRTY_FLAG"
    rm -f "$TRIAGE_LOG"
    echo "SAFE_PYTEST_RESULT: PASS" >&2
    exit 0
fi

# Pytest setup errors (no device touched):
#   4 = usage error (bad args, nonexistent path)
#   5 = no tests collected (typo in path, bad marker filter, etc.)
if [[ $EXIT_CODE -eq 4 || $EXIT_CODE -eq 5 ]]; then
    rm -f "$DIRTY_FLAG"
    rm -f "$TRIAGE_LOG"
    if [[ $EXIT_CODE -eq 4 ]]; then
        echo "SAFE_PYTEST_ERROR: Pytest usage error (invalid path or arguments)" >&2
    else
        echo "SAFE_PYTEST_ERROR: No tests collected" >&2
    fi
    exit 3
fi

# Kill any remaining child processes (pytest may have left orphans)
pkill -9 -P $$ 2>/dev/null || true

# Determine if this was a hang:
#   Triage log non-empty = dispatch timeout handler ran tt-triage (definitive hang signal)
# On sim there is no hang detection, so IS_HANG always stays false.
IS_HANG=false
if [[ -s "$TRIAGE_LOG" ]]; then
    IS_HANG=true
fi

# Only reset device when the failure might have left it dirty.
# Hangs and crashes corrupt device state. Normal test failures (PCC mismatch,
# assertion errors) and collection errors don't touch the device.
if [[ "$IS_HANG" == true ]]; then
    echo "SAFE_PYTEST: Resetting device..." >&2
    if tt-smi -r; then
        sleep 2
        rm -f "$DIRTY_FLAG"
        echo "SAFE_PYTEST: Device reset complete" >&2
    else
        echo "SAFE_PYTEST: Device reset FAILED; leaving device marked dirty" >&2
    fi

    echo "SAFE_PYTEST_RESULT: HANG (exit code: $EXIT_CODE)" >&2
    echo "" >&2

    # Dump full triage log
    echo "=== TRIAGE LOG ===" >&2
    cat "$TRIAGE_LOG" >&2
    echo "=== END TRIAGE LOG ===" >&2
    echo "" >&2

    # In dev mode, also dump watcher log
    if [[ "$DEV_MODE" == true && -f "$WATCHER_LOG" ]]; then
        echo "=== WATCHER LOG (last 50 lines) ===" >&2
        tail -50 "$WATCHER_LOG" >&2
        echo "=== END WATCHER LOG ===" >&2
        echo "" >&2
    fi

    # Print the JSON triage path as the last line so machine-readers can find it.
    if [[ -f "$TRIAGE_JSON" ]]; then
        echo "SAFE_PYTEST: JSON triage: ${TRIAGE_JSON}" >&2
    fi

    rm -f "$TRIAGE_LOG"
    exit 2
fi

rm -f "$DIRTY_FLAG"
rm -f "$TRIAGE_LOG"
echo "SAFE_PYTEST_RESULT: FAIL (exit code: $EXIT_CODE)" >&2
exit 1
