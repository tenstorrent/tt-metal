#!/bin/bash
# tt-test.sh - Cooperative device-aware test runner
#
# Uses flock to serialize device access across multiple agents/terminals.
# Uses TT_METAL_OPERATION_TIMEOUT_SECONDS for precise hang detection at the
# dispatch layer (does not penalize setup/compilation time).
# Automatically resets the device after hangs, ensuring the next runner
# always gets a clean device.
#
# Usage: .claude/scripts/tt-test.sh [--dev] [--device N] <test_path> [extra_pytest_args...]
#
# Options:
#   --dev       Adds no-poll watcher (NoC sanitizer, waypoints without polling
#               overhead), lightweight ebreak asserts, and auto-triage on hang
#               with callstack summary and watcher log dump.
#   --device N  Target device ID (default: 0). Each device gets its own flock
#               and dirty flag, so tests on different cards run in parallel.
#
# Modes:
#   default  - Dispatch timeout only. Lean, no debug overhead.
#   --dev    - Debug mode with watcher, asserts, and triage (see above).
#
# Exit codes:
#   0 - All tests passed
#   1 - Test failure (normal pytest failure, no hang)
#   2 - Hang detected (dispatch timeout or safety-net timeout fired)
#   3 - Setup error (missing args, etc.)

set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DISPATCH_TIMEOUT=5
SAFETY_NET_TIMEOUT=300
TRIAGE_SCRIPT="${REPO_DIR}/tools/tt-triage.py"
TRIAGE_SUMMARIZER="${REPO_DIR}/.claude/scripts/summarize-triage.py"
WATCHER_LOG="${REPO_DIR}/generated/watcher/watcher.log"

# --- Parse flags ---
DEV_MODE=false
DEVICE_ID=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --device)
            if [[ -z "${2:-}" || "$2" == -* ]]; then
                echo "TT_TEST_ERROR: --device requires a device ID" >&2
                exit 3
            fi
            DEVICE_ID="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Derive per-device paths from DEVICE_ID
LOCK_FILE="/tmp/tt-device-${DEVICE_ID}.lock"
DIRTY_FLAG="/tmp/tt-device-${DEVICE_ID}.dirty"
TRIAGE_LOG="/tmp/tt-test-triage-dev${DEVICE_ID}.log"

# --- Argument validation ---
if [[ $# -eq 0 ]]; then
    echo "TT_TEST_ERROR: No test path provided" >&2
    echo "Usage: .claude/scripts/tt-test.sh [--dev] [--device N] <test_path> [extra_pytest_args...]" >&2
    exit 3
fi

TEST_PATH="$1"
shift

# --- Acquire flock ---
exec 9>"$LOCK_FILE"

echo "TT_TEST: Waiting for device ${DEVICE_ID} lock..." >&2
flock 9
echo "TT_TEST: Device ${DEVICE_ID} lock acquired" >&2

# --- Check if device needs reset from previous hang ---
if [[ -f "$DIRTY_FLAG" ]]; then
    echo "TT_TEST: Device ${DEVICE_ID} marked dirty from previous hang, resetting..." >&2
    tt-smi -r "$DEVICE_ID" 2>/dev/null
    sleep 2
    rm -f "$DIRTY_FLAG"
    echo "TT_TEST: Device ${DEVICE_ID} reset complete" >&2
fi

# --- Setup environment ---
cd "$REPO_DIR"
source python_env/bin/activate 2>/dev/null || true

export TT_METAL_OPERATION_TIMEOUT_SECONDS="$DISPATCH_TIMEOUT"

# Auto-triage: dispatch layer runs tt-triage when timeout fires (both modes)
# This only executes on actual hang, not on every test — zero overhead for passing tests.
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

    # No-poll watcher: enables all device-side instrumentation (NoC sanitizer,
    # waypoints, ring buffer) WITHOUT the host polling thread. This avoids the
    # DMA disable and polling overhead (~1.5x vs ~22x with polling watcher).
    # On error, kernel hangs with diagnostic info in L1; triage reads it via
    # the dispatch timeout command.
    export TT_METAL_WATCHER_NO_POLL=1
    export TT_METAL_WATCHER_NOINLINE=1
    export TT_METAL_WATCHER_DISABLE_ASSERT=1
    export TT_METAL_WATCHER_DISABLE_DISPATCH=1

    echo "TT_TEST: [dev] asserts=ebreak llk_asserts=ON watcher=no-poll triage=ON timeout=${DISPATCH_TIMEOUT}s" >&2
else
    echo "TT_TEST: dispatch_timeout=${DISPATCH_TIMEOUT}s safety_net=${SAFETY_NET_TIMEOUT}s" >&2
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
# NOTE: Do NOT use setsid here — it breaks signal delivery and prevents the dispatch
# timeout from running TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE (causes SIGABRT
# instead of a clean TT_THROW that pytest can catch).
timeout --foreground "$SAFETY_NET_TIMEOUT" pytest "${TEST_PATH}" -x --device-id="$DEVICE_ID" "$@" 2>&1
EXIT_CODE=$?

echo "========================================" >&2

# --- Handle result ---
if [[ $EXIT_CODE -eq 0 ]]; then
    rm -f "$DIRTY_FLAG"
    echo "TT_TEST_RESULT: PASS" >&2
    exit 0
fi

# Determine if this was a hang:
#   Exit code 124 = safety-net timeout fired (catastrophic hang)
#   Exit code 134 = SIGABRT, 137 = SIGKILL (process killed)
#   Triage log non-empty = dispatch timeout handler ran tt-triage (definitive hang signal)
#   Pytest output contains "TIMEOUT:" = dispatch timeout raised TT_THROW
IS_HANG=false
if [[ $EXIT_CODE -eq 124 || $EXIT_CODE -eq 134 || $EXIT_CODE -eq 137 ]]; then
    IS_HANG=true
fi
if [[ -s "$TRIAGE_LOG" ]]; then
    IS_HANG=true
fi

# Kill any remaining child processes (pytest may have left orphans)
pkill -9 -P $$ 2>/dev/null || true
sleep 1

# Only reset device when the failure might have left it dirty.
# Hangs and crashes corrupt device state. Normal test failures (PCC mismatch,
# assertion errors) and collection errors don't touch the device.
NEEDS_RESET=false
if [[ "$IS_HANG" == true ]]; then
    NEEDS_RESET=true
fi

if [[ "$NEEDS_RESET" == true ]]; then
    echo "TT_TEST: Resetting device ${DEVICE_ID}..." >&2
    tt-smi -r "$DEVICE_ID" 2>/dev/null
    sleep 2
    rm -f "$DIRTY_FLAG"
    echo "TT_TEST: Device ${DEVICE_ID} reset complete" >&2
fi

if [[ "$IS_HANG" == true ]]; then
    echo "TT_TEST_RESULT: HANG (exit code: $EXIT_CODE)" >&2
    echo "" >&2

    # Print triage summary (available in both modes)
    if [[ -s "$TRIAGE_LOG" ]]; then
        python3 "$TRIAGE_SUMMARIZER" "$TRIAGE_LOG" >&2 2>/dev/null || \
            echo "TT_TEST: Triage saved to: ${TRIAGE_LOG} (summarizer failed)" >&2
    else
        echo "TT_TEST: No triage data captured (dispatch timeout may not have fired)" >&2
    fi
    echo "" >&2

    # In dev mode, also dump watcher log
    if [[ "$DEV_MODE" == true && -f "$WATCHER_LOG" ]]; then
        echo "=== WATCHER LOG (last 50 lines) ===" >&2
        tail -50 "$WATCHER_LOG" >&2
        echo "=== END WATCHER LOG ===" >&2
        echo "" >&2
    fi

    exit 2
fi

rm -f "$DIRTY_FLAG"
echo "TT_TEST_RESULT: FAIL (exit code: $EXIT_CODE)" >&2
exit 1
