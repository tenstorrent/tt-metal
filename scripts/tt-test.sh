#!/bin/bash
# tt-test.sh - Cooperative device-aware test runner
#
# Uses flock to serialize device access across multiple agents/terminals.
# Uses TT_METAL_OPERATION_TIMEOUT_SECONDS for precise hang detection at the
# dispatch layer (does not penalize setup/compilation time).
# Automatically resets the device after every run, ensuring the next runner
# always gets a clean device.
#
# Usage: scripts/tt-test.sh [--dev] [--run-all] <test_path> [extra_pytest_args...]
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
LOCK_FILE="/tmp/tt-device.lock"
DIRTY_FLAG="/tmp/tt-device.dirty"
TRIAGE_LOG="/tmp/tt-test-triage-$$.log"

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

# --- Acquire flock ---
exec 9>"$LOCK_FILE"

LOCK_TIMEOUT=300
echo "TT_TEST: Waiting for device lock..." >&2
if ! flock -w "$LOCK_TIMEOUT" 9; then
    echo "TT_TEST_ERROR: Could not acquire device lock after ${LOCK_TIMEOUT}s" >&2
    exit 3
fi
echo "TT_TEST: Device lock acquired" >&2

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

# Auto-triage: dispatch layer runs tt-triage when timeout fires (both modes).
# This only executes on actual hang, not on every test — zero overhead for passing tests.
# Requires tt-exalens: scripts/install_debugger.sh
if ! python3 -c "import ttexalens" 2>/dev/null; then
    echo "TT_TEST: WARNING: tt-exalens not installed — triage on hang will be unavailable." >&2
    echo "TT_TEST: Install with: scripts/install_debugger.sh" >&2
fi
rm -f "$TRIAGE_LOG"
export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 ${TRIAGE_SCRIPT} --disable-progress > ${TRIAGE_LOG} 2>&1"

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

    echo "TT_TEST: [dev] asserts=ebreak llk_asserts=ON watcher=polling triage=ON timeout=${DISPATCH_TIMEOUT}s" >&2
else
    echo "TT_TEST: dispatch_timeout=${DISPATCH_TIMEOUT}s" >&2
fi
echo "TT_TEST: pytest ${TEST_PATH} $*" >&2
echo "========================================" >&2

# --- Mark device dirty before running tests ---
# Pessimistic: assume the device will get corrupted. If the script is killed at any
# point (SIGKILL, OOM, etc.), the flag persists and the next runner will reset.
# Cleared on clean exit or after a successful inline reset.
touch "$DIRTY_FLAG"

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

# --- Cleanup: always kill orphans and reset device ---

# Kill any remaining child processes and their descendants.
if [[ $EXIT_CODE -ne 0 ]]; then
    for child_pid in $(pgrep -P $$ 2>/dev/null); do
        pkill -9 -P "$child_pid" 2>/dev/null || true
        kill -9 "$child_pid" 2>/dev/null || true
    done
fi

# Always reset device after every run to guarantee a clean slate.
echo "TT_TEST: Resetting device..." >&2
if tt-smi -r; then
    sleep 2
    rm -f "$DIRTY_FLAG"
    echo "TT_TEST: Device reset complete" >&2
else
    echo "TT_TEST: Device reset FAILED; leaving device marked dirty" >&2
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
#   Triage log non-empty = dispatch timeout handler ran tt-triage (definitive hang signal)
if [[ -s "$TRIAGE_LOG" ]]; then
    echo "TT_TEST_RESULT: HANG (exit code: $EXIT_CODE)" >&2
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

    rm -f "$TRIAGE_LOG"
    exit 2
fi

rm -f "$TRIAGE_LOG"
echo "TT_TEST_RESULT: FAIL (exit code: $EXIT_CODE)" >&2
exit 1
