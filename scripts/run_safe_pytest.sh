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
# Usage: scripts/run_safe_pytest.sh [--dev] [--run-all] [--precompile] <test_path> [extra_pytest_args...]
#
# Options:
#   --dev       Enables polling watcher (NoC sanitizer, waypoints, CB
#               sanitization), lightweight ebreak asserts, and auto-triage
#               on hang with full triage + watcher log dump.
#   --run-all   Run all tests instead of stopping on first failure (-x).
#               Useful for eval scoring where you need full pass/fail counts.
#   --precompile  Warm the JIT cache on the real device, in parallel, before the run, so kernels
#               compile up-front instead of inline & serial. No env vars; always falls back to a
#               cold run on any failure. Hardware only. --precompile-workers N (default: nproc).
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
#
# Total runtime:
#   SAFE_PYTEST_TOTAL_RUNTIME is printed last on every exit path — wall time from device-lock
#   acquired (idle lock-wait excluded) to exit, covering reset + warmup + pytest. Sim: from start.

set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISPATCH_TIMEOUT="${SAFE_PYTEST_DISPATCH_TIMEOUT:-5}"
TRIAGE_SCRIPT="${REPO_DIR}/tools/tt-triage.py"
WATCHER_LOG="${REPO_DIR}/generated/watcher/watcher.log"
LOCK_FILE="/tmp/tt-device.lock"
DIRTY_FLAG="/tmp/tt-device.dirty"
TRIAGE_LOG="/tmp/safe-pytest-triage-$$.log"

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
PRECOMPILE=false
PRECOMPILE_WORKERS="${PRECOMPILE_WORKERS:-$(nproc 2>/dev/null || echo 8)}"
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
        --precompile)
            # Warm the JIT cache (real device, parallel) before the real run. Internal only;
            # always falls back to a cold run on failure (see precompile_warm). Hardware only.
            PRECOMPILE=true
            shift
            ;;
        --precompile-workers)
            PRECOMPILE_WORKERS="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# --- Argument validation ---
if [[ $# -eq 0 ]]; then
    echo "SAFE_PYTEST_ERROR: No test path provided" >&2
    echo "Usage: scripts/run_safe_pytest.sh [--dev] [--run-all] [--precompile] <test_path> [extra_pytest_args...]" >&2
    exit 3
fi

TEST_PATH="$1"
shift
# Remaining args are extra pytest args — the precompile collect must use the SAME selection.
EXTRA_ARGS=("$@")

# Both the warmup and the real run inherit whatever TT_METAL_CACHE / ccache the user has (we never
# override), so they share one cache and a pre-warmed one is reused.
PRECOMPILE_PLUGIN_DIR="$REPO_DIR"

# Precompile (--precompile): warm the JIT cache on the SAME device the tests use (so the build_key
# matches by construction), then let the normal run below hit it. Any failure degrades to a cold run.

precompile_warm() {
    [[ "$SIM_MODE" == false ]] && touch "$DIRTY_FLAG"
    echo "PRECOMPILE: ===== warmup (collect + precompile, real device) =====" >&2
    # Single-process by design: the heavy kernel COMPILE is parallelized in-process by
    # up_front_compile's thread pool; xdist (-n) would only parallelize the cheap collect and
    # measurably loses ~half the cache (concurrent writers + per-worker dedup). REAL_ALLOC gives real
    # addresses so address-baked kernels (pool/move/conv) warm too. ccache state is inherited so it
    # matches the real run — a mismatch would silently miss the cache.
    local clog="/tmp/precompile_collect_$$.log" t0 t1 cstatus
    echo "PRECOMPILE: warming (single proc x ${PRECOMPILE_WORKERS} compile-threads) over: ${TEST_PATH} ${EXTRA_ARGS[*]}" >&2
    t0=$(date +%s)
    UP_FRONT_REAL_ALLOC=1 UP_FRONT_COLLECT_WORKERS="$PRECOMPILE_WORKERS" \
    LOGURU_LEVEL=ERROR PYTHONPATH="$PRECOMPILE_PLUGIN_DIR" \
        pytest "${TEST_PATH}" "${EXTRA_ARGS[@]}" -p tests.plugins.up_front_collect > "$clog" 2>&1
    cstatus=$?
    t1=$(date +%s)
    # A non-zero exit (collection error, plugin failure, OOM, pytest-5 "no tests") means nothing
    # warmed -> say so; the real run still runs cold and correct, just without the speedup.
    if [[ $cstatus -ne 0 ]]; then
        echo "PRECOMPILE: ✗ warmup FAILED (pytest exit $cstatus) after $((t1-t0))s -> warmed NOTHING; running COLD." >&2
        grep -iE "error|unrecognized|no tests ran|no tests collected" "$clog" 2>/dev/null | head -3 | sed 's/^/PRECOMPILE:   /' >&2
        echo "PRECOMPILE:   (full collect log: $clog)" >&2
        return 0
    fi
    echo "PRECOMPILE: ✓ warmup complete in $((t1-t0))s — the real run below reuses it. Log: $clog" >&2
}

# --- Total-run timer ---
# EXIT trap so it always prints last on every path; the RUN_START guard suppresses it for exits
# before testing begins (e.g. missing args).
_print_total_runtime() {
    [[ -z "${RUN_START:-}" ]] && return 0
    local run_end elapsed
    run_end=$(date +%s)
    elapsed=$((run_end - RUN_START))
    echo "========================================" >&2
    printf 'SAFE_PYTEST_TOTAL_RUNTIME: %dm%02ds (%ds total, device-lock-acquired -> exit)\n' \
        $((elapsed / 60)) $((elapsed % 60)) "$elapsed" >&2
}
trap _print_total_runtime EXIT

# --- Acquire flock (hardware only) ---
if [[ "$SIM_MODE" == false ]]; then
    exec 9>"$LOCK_FILE"

    echo "SAFE_PYTEST: Waiting for device lock..." >&2
    flock 9
    echo "SAFE_PYTEST: Device lock acquired" >&2

    # Start the clock once we own the device; the idle lock-wait above is excluded.
    RUN_START=$(date +%s)

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
else
    # Simulator: no lock, so start the clock here.
    RUN_START=$(date +%s)
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

# --- Dev-mode build flags (must precede the warm phase: they feed the JIT build_key) ---
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
fi

# --- Precompile warm phase (opt-in, hardware only; never aborts the real run) ---
if [[ "$PRECOMPILE" == true ]]; then
    if [[ "$SIM_MODE" == true ]]; then
        echo "PRECOMPILE: skipped under simulator (no warm benefit)" >&2
    else
        precompile_warm
    fi
fi

# --- Hang detection setup (hardware only) ---
# On timeout, the dispatch layer runs tt-triage. Fires only on actual hang —
# zero overhead for passing tests. On sim there is no hang detection because
# wall-clock timeouts are meaningless at kHz clock speeds.
rm -f "$TRIAGE_LOG"
if [[ "$SIM_MODE" == false ]]; then
    export TT_METAL_OPERATION_TIMEOUT_SECONDS="$DISPATCH_TIMEOUT"
    # Requires tt-exalens: uv pip install -r tools/triage/requirements.txt
    if ! python3 -c "import ttexalens" 2>/dev/null; then
        echo "SAFE_PYTEST: WARNING: tt-exalens not installed — triage on hang will be unavailable." >&2
        echo "SAFE_PYTEST: Install with: uv pip install -r tools/triage/requirements.txt" >&2
    fi
    export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 ${TRIAGE_SCRIPT} --disable-progress > ${TRIAGE_LOG} 2>&1"
fi

# Dev-mode banner (flags exported above).
if [[ "$DEV_MODE" == true ]]; then
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

"${PYTEST_CMD[@]}"
EXIT_CODE=$?

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

    rm -f "$TRIAGE_LOG"
    exit 2
fi

rm -f "$DIRTY_FLAG"
rm -f "$TRIAGE_LOG"
echo "SAFE_PYTEST_RESULT: FAIL (exit code: $EXIT_CODE)" >&2
exit 1
