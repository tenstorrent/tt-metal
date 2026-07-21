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
# Usage: scripts/run_safe_pytest.sh [--dev] [--run-all] [--profile] <test_path> [extra_pytest_args...]
#
# Options:
#   --dev       Enables polling watcher (NoC sanitizer, waypoints, CB
#               sanitization), lightweight ebreak asserts, and auto-triage
#               on hang with full triage + watcher log dump.
#   --run-all   Run all tests instead of stopping on first failure (-x).
#               Useful for eval scoring where you need full pass/fail counts.
#   --profile   Run under the Tracy device profiler (python -m tracy -r). Emits
#               a per-op CSV (generated/profiler/reports/<ts>/ops_perf_results*.csv)
#               and prints its path as "SAFE_PYTEST: PROFILER CSV: <path>" next to
#               the result line. Requires a Tracy-enabled build. NOTE: the tracy
#               wrapper masks pytest's exit code, so a profiled run is reported
#               PASS as long as profiling completed, regardless of the underlying
#               test result. Hangs are still detected and still reset the device.
#
# Modes:
#   default  - Dispatch timeout only. Lean, no debug overhead.
#   --dev    - Debug mode with watcher, asserts, and triage (see above).
#   --profile - Tracy device profiling with per-op CSV report (see above).
#
# Exit codes:
#   0 - All tests passed
#   1 - Test failure (normal pytest failure, no hang)
#   2 - Hang detected (dispatch timeout fired)
#   3 - Setup error (missing args, etc.)

set -o pipefail

# SAFE_PYTEST status lines and triage/watcher dumps go to stdout (plain echo),
# the same stream the metal runtime uses for its logging/DPRINT/JIT output;
# pytest's own stdout/stderr pass through unchanged.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISPATCH_TIMEOUT=5
TRIAGE_SCRIPT="${REPO_DIR}/tools/tt-triage.py"
WATCHER_LOG="${REPO_DIR}/generated/watcher/watcher.log"
TRIAGE_LLM_DIR="${REPO_DIR}/generated/tt-triage"
LOCK_FILE="/tmp/tt-device.lock"
DIRTY_FLAG="/tmp/tt-device.dirty"
TRIAGE_LOG="/tmp/safe-pytest-triage-$$.log"
TRIAGE_LLM="${TRIAGE_LLM_DIR}/triage.csv"
PROFILE_REPORTS_DIR="${REPO_DIR}/generated/profiler/reports"

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
PROFILE_MODE=false
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
        --profile)
            PROFILE_MODE=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# --- Argument validation ---
if [[ $# -eq 0 ]]; then
    echo "SAFE_PYTEST_ERROR: No test path provided"
    echo "Usage: scripts/run_safe_pytest.sh [--dev] [--run-all] [--profile] <test_path> [extra_pytest_args...]"
    exit 3
fi

TEST_PATH="$1"
shift

# --- Deferred warnings ---
# Warnings raised during setup get buried under pytest/triage output, so the
# user never sees them. Buffer them and flush via an EXIT trap so they surface
# as the very last thing printed, regardless of how the script exits.
DEFERRED_WARNINGS=()
defer_warning() { DEFERRED_WARNINGS+=("$1"); }
flush_deferred_warnings() {
    if [[ ${#DEFERRED_WARNINGS[@]} -gt 0 ]]; then
        echo ""
        local w
        for w in "${DEFERRED_WARNINGS[@]}"; do
            echo "$w"
        done
    fi
}
trap flush_deferred_warnings EXIT

# Newest ops_perf_results CSV before the run; set just before pytest (below).
PROFILE_CSV_BEFORE=""

# Print this run's per-op CSV path. `python -m tracy -r` writes a fresh
# reports/<ts>/ subdir per run, so we report the newest only if it differs from
# the pre-run snapshot — otherwise this run produced none and it's a stale leftover.
emit_profiler_csv() {
    [[ "$PROFILE_MODE" == true ]] || return 0
    local csv
    csv=$(ls -t "${PROFILE_REPORTS_DIR}"/*/ops_perf_results*.csv 2>/dev/null | head -1)
    if [[ -n "$csv" && "$csv" != "$PROFILE_CSV_BEFORE" ]]; then
        echo "SAFE_PYTEST: PROFILER CSV: ${csv}"
    else
        echo "SAFE_PYTEST: WARNING: --profile set but this run produced no ops_perf_results CSV"
    fi
}

# --- Acquire flock (hardware only) ---
if [[ "$SIM_MODE" == false ]]; then
    exec 9>"$LOCK_FILE"

    echo "SAFE_PYTEST: Waiting for device lock..."
    flock 9
    echo "SAFE_PYTEST: Device lock acquired"

    # --- Check if device needs reset from previous hang ---
    if [[ -f "$DIRTY_FLAG" ]]; then
        echo "SAFE_PYTEST: Device marked dirty from previous hang, resetting..."
        if ! tt-smi -r; then
            echo "SAFE_PYTEST_ERROR: Device reset (tt-smi -r) failed"
            exit 3
        fi
        rm -f "$DIRTY_FLAG"
        echo "SAFE_PYTEST: Device reset complete"
    fi
fi

# --- Setup environment ---
cd "$REPO_DIR"
if [[ -f python_env/bin/activate ]]; then
    if ! source python_env/bin/activate; then
        defer_warning "SAFE_PYTEST: WARNING: Failed to activate python_env virtual environment"
    fi
else
    defer_warning "SAFE_PYTEST: WARNING: python_env not found; using system Python"
fi

# --- Profiling preflight ---
# `python -m tracy` needs a Tracy-enabled build and tracy deps (e.g. websockets).
# Probe the import it does at startup (tracy.__main__ -> tracy.serve_wasm) so a
# missing dep fails fast here instead of as a confusing mid-run traceback.
if [[ "$PROFILE_MODE" == true ]]; then
    if ! python3 -c "import tracy.serve_wasm" 2>/dev/null; then
        echo "SAFE_PYTEST_ERROR: --profile requested but 'python -m tracy' is unavailable"
        echo "SAFE_PYTEST: Ensure a Tracy-enabled build and tracy deps (e.g. 'pip install websockets')"
        exit 3
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
        defer_warning "SAFE_PYTEST: WARNING: tt-exalens not installed — triage on hang will be unavailable."
        defer_warning "SAFE_PYTEST: Install with: uv pip install -r tools/triage/requirements.txt"
    fi
    # --llm-output-path also writes the triage report as CSV to a persistent file,
    # so machine-readers can consume it directly instead of scraping the log.
    mkdir -p "${TRIAGE_LLM_DIR}"
    export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 ${TRIAGE_SCRIPT} --disable-progress --skip-version-check --llm-output-path=${TRIAGE_LLM} > ${TRIAGE_LOG} 2>&1"
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
        echo "SAFE_PYTEST: [sim+dev] asserts=ebreak llk_asserts=ON watcher=polling (no hang detection on sim)"
    else
        echo "SAFE_PYTEST: [dev] asserts=ebreak llk_asserts=ON watcher=polling triage=ON timeout=${DISPATCH_TIMEOUT}s"
    fi
elif [[ "$SIM_MODE" == true ]]; then
    echo "SAFE_PYTEST: [sim] no hang detection"
else
    echo "SAFE_PYTEST: dispatch_timeout=${DISPATCH_TIMEOUT}s"
fi
echo "SAFE_PYTEST: pytest ${TEST_PATH} $*"
echo "========================================"

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
# --profile: wrap pytest in the Tracy profiler. `python -m tracy -r` runs pytest
#   as a child and post-processes results into ops_perf_results*.csv on pass or
#   fail. Its exit-code masking is handled at the result check below.
if [[ "$PROFILE_MODE" == true ]]; then
    PYTEST_CMD=(python -m tracy -r -m pytest "${TEST_PATH}")
else
    PYTEST_CMD=(pytest "${TEST_PATH}")
fi
if [[ "$FAIL_FAST" == true ]]; then
    PYTEST_CMD+=(-x)
fi
PYTEST_CMD+=("$@")

# Snapshot the newest CSV now, so emit_profiler_csv can tell this run's report
# from a pre-existing one afterward.
if [[ "$PROFILE_MODE" == true ]]; then
    PROFILE_CSV_BEFORE=$(ls -t "${PROFILE_REPORTS_DIR}"/*/ops_perf_results*.csv 2>/dev/null | head -1)
fi

"${PYTEST_CMD[@]}"
EXIT_CODE=$?

echo "========================================"

# --- Handle result ---
# The triage-log guard matters in profile mode: the tracy wrapper exits 0 even
# when the underlying test failed OR hung, so without it a hang would be reported
# PASS and skip the device reset. An empty triage log means no hang fired.
if [[ $EXIT_CODE -eq 0 && ! -s "$TRIAGE_LOG" ]]; then
    rm -f "$DIRTY_FLAG"
    rm -f "$TRIAGE_LOG"
    emit_profiler_csv
    echo "SAFE_PYTEST_RESULT: PASS"
    exit 0
fi

# Pytest setup errors (no device touched):
#   4 = usage error (bad args, nonexistent path)
#   5 = no tests collected (typo in path, bad marker filter, etc.)
if [[ $EXIT_CODE -eq 4 || $EXIT_CODE -eq 5 ]]; then
    rm -f "$DIRTY_FLAG"
    rm -f "$TRIAGE_LOG"
    if [[ $EXIT_CODE -eq 4 ]]; then
        echo "SAFE_PYTEST_ERROR: Pytest usage error (invalid path or arguments)"
    else
        echo "SAFE_PYTEST_ERROR: No tests collected"
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
    echo "SAFE_PYTEST: Resetting device..."
    if tt-smi -r; then
        sleep 2
        rm -f "$DIRTY_FLAG"
        echo "SAFE_PYTEST: Device reset complete"
    else
        echo "SAFE_PYTEST: Device reset FAILED; leaving device marked dirty"
    fi

    echo "SAFE_PYTEST_RESULT: HANG (exit code: $EXIT_CODE)"
    echo ""

    # Dump full triage log
    echo "=== TRIAGE LOG ==="
    cat "$TRIAGE_LOG"
    echo "=== END TRIAGE LOG ==="
    echo ""

    # In dev mode, also dump watcher log
    if [[ "$DEV_MODE" == true && -f "$WATCHER_LOG" ]]; then
        echo "=== WATCHER LOG (last 50 lines) ==="
        tail -50 "$WATCHER_LOG"
        echo "=== END WATCHER LOG ==="
        echo ""
    fi

    # Emit the LLM triage path so machine-readers can find it (grep the prefix).
    if [[ -f "$TRIAGE_LLM" ]]; then
        echo "SAFE_PYTEST: LLM triage: ${TRIAGE_LLM}"
    fi

    rm -f "$TRIAGE_LOG"
    exit 2
fi

rm -f "$DIRTY_FLAG"
rm -f "$TRIAGE_LOG"
echo "SAFE_PYTEST_RESULT: FAIL (exit code: $EXIT_CODE)"
exit 1
