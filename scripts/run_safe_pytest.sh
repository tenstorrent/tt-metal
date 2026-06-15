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
# Usage: scripts/run_safe_pytest.sh [--dev] [--run-all] [--sim-workers N] [--precompile|--no-precompile] <test_path> [extra_pytest_args...]
#
# Options:
#   --dev            Enables polling watcher (NoC sanitizer, waypoints, CB
#                    sanitization), lightweight ebreak asserts, and auto-triage
#                    on hang with full triage + watcher log dump.
#   --run-all        Run all tests instead of stopping on first failure (-x).
#                    Useful for eval scoring where you need full pass/fail counts.
#   --sim-workers N  Sim only: pytest-xdist worker count. Each worker dlopens
#                    its own libttsim (no shared device state). Default is 16.
#                    Pass 1 to serialize (e.g. when DPRINT ordering matters or
#                    you suspect cross-worker contention). Errors out if used
#                    outside sim mode.
#   --precompile     Force-on: before the real run, transparently warm the JIT cache on the real
#                    device and in parallel (no env vars, no second command), so kernels
#                    compile up-front in parallel instead of inline & serial. ALWAYS falls
#                    back to a normal cold run if anything goes wrong — it can only make a
#                    run slower, never broken or wrong. Prints a one-line diagnostic. Hardware
#                    only. Tune parallelism with --precompile-workers N (default: nproc).
#   --no-precompile  Force-off: skip the warm pass even on a broad run.
#                    AUTO-ROUTING (default, neither flag given): decided from argv alone (free, no
#                    collect pre-pass). BROAD (a whole directory or a whole test_*.py file, with no
#                    ::nodeid and no -k) -> warm pass pays -> ON. NARROW (::nodeid, -k filter, or a
#                    --dev repro) -> few kernels -> left cold. Explicit flags win. Sim always off.
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
#   Always prints SAFE_PYTEST_TOTAL_RUNTIME as the very last line (on every exit path).
#   It is the wall-clock time from "device lock acquired" (idle lock-wait queueing is
#   deliberately excluded) to script exit, so it covers the whole run — device reset,
#   precompile warm phase, and the pytest run itself — not just pytest. Under simulator
#   there is no lock, so the clock starts at the equivalent point.

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
SIM_WORKERS=""
SIM_WORKERS_GIVEN=false
PRECOMPILE=false
PRECOMPILE_EXPLICIT=false   # true once --precompile/--no-precompile is seen; disables auto-routing
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
        --sim-workers)
            if [[ $# -lt 2 ]]; then
                echo "SAFE_PYTEST_ERROR: --sim-workers requires an integer argument"
                exit 3
            fi
            SIM_WORKERS="$2"
            SIM_WORKERS_GIVEN=true
            shift 2
            ;;
        --precompile)
            # Force-on: transparently warm the JIT cache (real-device, parallel) before the
            # real run, so kernels compile up-front in parallel instead of inline & serial.
            # Everything is internal — no env vars, no second command. Always falls back to a
            # normal cold run if anything goes wrong (see precompile_warm). Hardware only.
            PRECOMPILE=true
            PRECOMPILE_EXPLICIT=true
            shift
            ;;
        --no-precompile)
            # Force-off: skip the warm pass even on a broad run (which auto-routing would warm).
            PRECOMPILE=false
            PRECOMPILE_EXPLICIT=true
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
    echo "Usage: scripts/run_safe_pytest.sh [--dev] [--run-all] [--sim-workers N] [--precompile|--no-precompile] <test_path> [extra_pytest_args...]" >&2
    exit 3
fi

TEST_PATH="$1"
shift
# Remaining args are extra pytest args — the precompile collect must use the SAME selection.
EXTRA_ARGS=("$@")

# --- Auto-route precompile by run BREADTH (unless --precompile/--no-precompile given) ---
# Free, argv-only — no collect pre-pass. The warm pass only pays when there are many distinct
# kernels to compile in parallel; on a narrow run its fixed overhead (a 2nd device open + a collect
# body-run) exceeds the little serial-inline compile it would save. So:
#   BROAD  = a whole directory or a whole test_*.py file, with NO ::nodeid and NO -k filter -> ON
#   NARROW = a ::nodeid, a -k filter, or a --dev repro (single-case debugging)               -> OFF
# Explicit --precompile/--no-precompile win and disable this. Sim is skipped at the warm call below.
if [[ "$PRECOMPILE_EXPLICIT" == false && "$SIM_MODE" == false ]]; then
    _narrow=false
    [[ "$DEV_MODE" == true ]] && _narrow=true        # --dev = single-case repro; warm-up is overhead
    [[ "$TEST_PATH" == *"::"* ]] && _narrow=true      # nodeid selection
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        for _a in "${EXTRA_ARGS[@]}"; do
            case "$_a" in
                -k|-k*|*"::"*) _narrow=true ;;        # -k filter or nodeid in extra args
            esac
        done
    fi
    if [[ "$_narrow" == true ]]; then
        PRECOMPILE=false
        echo "SAFE_PYTEST: precompile off (narrow run: nodeid/-k/--dev; pass --precompile to force)"
    elif [[ -d "$TEST_PATH" ]]; then
        PRECOMPILE=true
        echo "SAFE_PYTEST: precompile auto-enabled (broad run: whole directory; --no-precompile to disable)"
    elif [[ "$TEST_PATH" == *.py && -f "$TEST_PATH" ]]; then
        PRECOMPILE=true
        echo "SAFE_PYTEST: precompile auto-enabled (broad run: whole test file; --no-precompile to disable)"
    else
        PRECOMPILE=false
        echo "SAFE_PYTEST: precompile off (not a whole dir/file: ${TEST_PATH}; pass --precompile to force)"
    fi
fi

# Precompile uses WHATEVER cache the user already has (TT_METAL_CACHE if set, else tt-metal's
# default) — both the warm-collect and the real run inherit the same value (incl. ccache state), so
# they share it and a pre-warmed cache is transparently reused. We never override it.
PRECOMPILE_PLUGIN_DIR="$REPO_DIR"

# ============================================================================
# Precompile (opt-in --precompile): warm the JIT cache on the REAL device, then
# let the normal run below hit it. We open the same device the tests use, so the
# build_key matches by construction — no mock / fingerprint / pre-flight needed.
# Every failure path degrades to a normal cold run — slower at worst, never wrong.
# ============================================================================

precompile_warm() {
    [[ "$SIM_MODE" == false ]] && touch "$DIRTY_FLAG"
    echo "PRECOMPILE: ===== warmup (collect + precompile, real device) =====" >&2
    # Real-device collect over the SAME selection -> warms the shared cache. We open the same device the
    # real run uses, so the build_key matches by construction (no mock / fingerprint / pre-flight needed).
    # SINGLE-PROCESS by design: the heavy kernel COMPILE is parallelized by the plugin's in-process C++
    # thread pool via ttnn.graph.up_front_compile(device, UP_FRONT_COLLECT_WORKERS=N). xdist (-n) would
    # only parallelize the cheap COLLECT body-run and, measured, LOSES ~half the cache (concurrent writers
    # + per-worker dedup: full conv2d 47.6% xdist vs 99.8% single-process). FAST collect (default) keeps
    # real torch tensors with cheap host stand-ins + a SHAPE-ONLY ttnn.from_torch (skips the weight-prep
    # tilize/convert) — works on model tests where storage-free collect collapses on weight prep. REAL_ALLOC
    # gives real buffer addresses so address-baked kernels (pool/move/conv) warm too. ccache state is
    # INHERITED (untouched) so it matches the real run below — a mismatch would silently miss the warm cache.
    local clog="/tmp/precompile_collect_$$.log" t0 t1 cstatus
    echo "PRECOMPILE: warming (single proc x ${PRECOMPILE_WORKERS} compile-threads) over: ${TEST_PATH} ${EXTRA_ARGS[*]}" >&2
    t0=$(date +%s)
    UP_FRONT_COLLECT=1 UP_FRONT_REAL_ALLOC=1 UP_FRONT_COLLECT_WORKERS="$PRECOMPILE_WORKERS" \
    LOGURU_LEVEL=ERROR PYTHONPATH="$PRECOMPILE_PLUGIN_DIR" \
        pytest "${TEST_PATH}" "${EXTRA_ARGS[@]}" -p tests.plugins.up_front_collect > "$clog" 2>&1
    cstatus=$?
    t1=$(date +%s)
    # Don't pretend it warmed if the collect failed. A non-zero exit (pytest usage/collection error,
    # plugin failure, OOM, etc.) means we warmed nothing -> say so plainly; the real run still runs COLD
    # and CORRECT, just without the speedup. (pytest exit 5 = "no tests collected" counts as a failure.)
    if [[ $cstatus -ne 0 ]]; then
        echo "PRECOMPILE: ✗ warmup FAILED (pytest exit $cstatus) after $((t1-t0))s -> warmed NOTHING; running COLD." >&2
        grep -iE "error|unrecognized|no tests ran|no tests collected" "$clog" 2>/dev/null | head -3 | sed 's/^/PRECOMPILE:   /' >&2
        echo "PRECOMPILE:   (full collect log: $clog)" >&2
        return 0
    fi
    echo "PRECOMPILE: ✓ warmup complete in $((t1-t0))s — the real run below reuses it. Log: $clog" >&2
}

# --- Total-run timer ---
# Reports wall-clock time from "device lock acquired" to script exit. Registered as an
# EXIT trap so it ALWAYS prints last, on every exit path (pass, fail, hang, error). The
# RUN_START guard means nothing is printed for exits that happen before testing begins
# (e.g. missing args), since no run took place.
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

    # Start the total-run clock the moment we own the device. The lock-wait above is
    # idle queueing behind other runners and is deliberately excluded.
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
    # Simulator: no device lock to acquire, so start the total-run clock here (the
    # equivalent "start of testing" point).
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
