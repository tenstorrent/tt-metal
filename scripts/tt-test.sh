#!/bin/bash
# tt-test.sh - Cooperative device-aware test runner
#
# Uses flock to serialize device access across multiple agents/terminals.
# Uses TT_METAL_OPERATION_TIMEOUT_SECONDS for precise hang detection at the
# dispatch layer (does not penalize setup/compilation time).
# Automatically resets the device after every run, ensuring the next runner
# always gets a clean device.
#
# Simulator mode (TT_METAL_SIMULATOR set):
#   Skips flock, device resets, dirty tracking, triage, and --dev features
#   (none of these work on the simulator). Keeps dispatch timeout for hang
#   detection (default 120s, override via TT_METAL_OPERATION_TIMEOUT_SECONDS).
#
# Usage: scripts/tt-test.sh [--dev] [--run-all] <test_path> [extra_pytest_args...]
#
# Options:
#   --dev       Enables polling watcher (NoC sanitizer, waypoints, CB
#               sanitization), lightweight ebreak asserts, and auto-triage
#               on hang with full triage + watcher log dump.
#               (ignored on simulator — these features require real hardware)
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
TRIAGE_SCRIPT="${REPO_DIR}/tools/tt-triage.py"
WATCHER_LOG="${REPO_DIR}/generated/watcher/watcher.log"
LOCK_FILE="/tmp/tt-device.lock"
DIRTY_FLAG="/tmp/tt-device.dirty"
TRIAGE_LOG="/tmp/tt-test-triage-$$.log"

# --- Detect simulator mode ---
SIM_MODE=false
if [[ -n "${TT_METAL_SIMULATOR:-}" ]]; then
    SIM_MODE=true
fi

# Sim is much slower than silicon — use a higher default timeout.
# User can override via TT_METAL_OPERATION_TIMEOUT_SECONDS.
if [[ "$SIM_MODE" == true ]]; then
    DISPATCH_TIMEOUT=${TT_METAL_OPERATION_TIMEOUT_SECONDS:-120}
else
    DISPATCH_TIMEOUT=5
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
    echo "TT_TEST_ERROR: No test path provided" >&2
    echo "Usage: scripts/tt-test.sh [--dev] <test_path> [extra_pytest_args...]" >&2
    exit 3
fi

TEST_PATH="$1"
shift

# --- Acquire flock (hardware only) ---
if [[ "$SIM_MODE" == false ]]; then
    exec 9>"$LOCK_FILE"

    LOCK_TIMEOUT=600  # 10min: accounts for queued runs holding device during dual-mode testing
    echo "TT_TEST: Waiting for device lock..." >&2
    if ! flock -w "$LOCK_TIMEOUT" 9; then
        echo "TT_TEST_ERROR: Could not acquire device lock after ${LOCK_TIMEOUT}s" >&2
        exit 3
    fi
    echo "TT_TEST: Device lock acquired" >&2

    # Signal to child processes (e.g. conftest device lock plugin) that the lock
    # is already held — they must not re-acquire it or they will deadlock.
    export TT_DEVICE_LOCK_HELD=1

    # --- Check if device needs reset from previous hang ---
    if [[ -f "$DIRTY_FLAG" ]]; then
        echo "TT_TEST: Device marked dirty from previous run, resetting..." >&2
        if ! tt-smi -r; then
            echo "TT_TEST_ERROR: Device reset (tt-smi -r) failed" >&2
            exit 3
        fi
        rm -f "$DIRTY_FLAG"
        echo "TT_TEST: Device reset complete" >&2
    fi
fi

# --- Setup environment ---
cd "$REPO_DIR"
if [[ -f python_env/bin/activate ]]; then
    if ! source python_env/bin/activate; then
        echo "TT_TEST: WARNING: Failed to activate python_env virtual environment" >&2
    fi
else
    echo "TT_TEST: WARNING: python_env not found; using system Python" >&2
fi

export TT_METAL_OPERATION_TIMEOUT_SECONDS="$DISPATCH_TIMEOUT"

# --- Hang detection setup ---
# On timeout, the dispatch layer runs this command. On hardware we get a full
# triage; on sim, triage/exalens are unsupported so we just write a marker file
# that the hang-detection logic below can check.
rm -f "$TRIAGE_LOG"
if [[ "$SIM_MODE" == true ]]; then
    export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="echo HANG > ${TRIAGE_LOG}"
else
    # Requires tt-exalens: scripts/install_debugger.sh
    if ! python3 -c "import ttexalens" 2>/dev/null; then
        echo "TT_TEST: WARNING: tt-exalens not installed — triage on hang will be unavailable." >&2
        echo "TT_TEST: Install with: scripts/install_debugger.sh" >&2
    fi
    export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 ${TRIAGE_SCRIPT} --disable-progress > ${TRIAGE_LOG} 2>&1"
fi

if [[ "$SIM_MODE" == true ]]; then
    if [[ "$DEV_MODE" == true ]]; then
        echo "TT_TEST: WARNING: --dev mode ignored on simulator (watcher/asserts/triage unsupported)" >&2
    fi
    echo "TT_TEST: [sim] dispatch_timeout=${DISPATCH_TIMEOUT}s" >&2
elif [[ "$DEV_MODE" == true ]]; then
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

    echo "TT_TEST: [dev] asserts=ebreak llk_asserts=ON watcher=polling triage=ON timeout=${DISPATCH_TIMEOUT}s" >&2
else
    echo "TT_TEST: dispatch_timeout=${DISPATCH_TIMEOUT}s" >&2
fi
echo "TT_TEST: pytest ${TEST_PATH} $*" >&2
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
if [[ "$FAIL_FAST" == true ]]; then
    pytest "${TEST_PATH}" -x "$@"
else
    pytest "${TEST_PATH}" "$@"
fi
EXIT_CODE=$?

echo "========================================" >&2

# --- Cleanup: kill orphans and reset device (hardware only) ---

# Kill any remaining child processes and their descendants.
if [[ $EXIT_CODE -ne 0 ]]; then
    for child_pid in $(pgrep -P $$ 2>/dev/null); do
        pkill -9 -P "$child_pid" 2>/dev/null || true
        kill -9 "$child_pid" 2>/dev/null || true
    done
fi

if [[ "$SIM_MODE" == false ]]; then
    # Always reset device after every run to guarantee a clean slate.
    echo "TT_TEST: Resetting device..." >&2
    if tt-smi -r; then
        sleep 2
        rm -f "$DIRTY_FLAG"
        echo "TT_TEST: Device reset complete" >&2
    else
        echo "TT_TEST: Device reset FAILED; leaving device marked dirty" >&2
    fi
fi

# --- Handle result ---

if [[ $EXIT_CODE -eq 0 ]]; then
    rm -f "$TRIAGE_LOG"
    echo "TT_TEST_RESULT: PASS" >&2
    exit 0
fi

# Pytest exit code 5 = no tests collected (typo in path, bad marker filter, etc.)
if [[ $EXIT_CODE -eq 5 ]]; then
    rm -f "$TRIAGE_LOG"
    echo "TT_TEST_ERROR: No tests collected" >&2
    exit 3
fi

# Determine if this was a hang:
#   Triage log non-empty = dispatch timeout handler fired (definitive hang signal).
#   On hardware this contains full triage output; on sim it's just a "HANG" marker.
if [[ -s "$TRIAGE_LOG" ]]; then
    echo "TT_TEST_RESULT: HANG (exit code: $EXIT_CODE)" >&2
    echo "" >&2

    if [[ "$SIM_MODE" == true ]]; then
        echo "TT_TEST: Dispatch timeout fired (simulator mode — no triage available)" >&2
    else
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
    fi

    rm -f "$TRIAGE_LOG"
    exit 2
fi

rm -f "$TRIAGE_LOG"
echo "TT_TEST_RESULT: FAIL (exit code: $EXIT_CODE)" >&2
exit 1
