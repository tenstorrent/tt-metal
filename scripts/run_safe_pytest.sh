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
#   Skips flock, device resets, and HW dispatch-timeout triage (these require
#   real hardware).
#   Enables the libttsim hang watchdog via TTSIM_HANG_WATCHDOG_CLOCKS (default
#   50000; pre-existing env wins). On hang the watchdog _Exit(1)'s the child;
#   we classify that as HANG and dump the watchdog message.
#
# Usage: scripts/run_safe_pytest.sh [--dev] [--run-all] [--sim-workers N] <test_path> [extra_pytest_args...]
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
TRIAGE_OUT_DIR="${REPO_DIR}/generated/tt-triage"
LOCK_FILE="/tmp/tt-device.lock"
DIRTY_FLAG="/tmp/tt-device.dirty"
TRIAGE_LOG="/tmp/safe-pytest-triage-$$.log"
TRIAGE_REPORT="${TRIAGE_OUT_DIR}/triage.txt"

# --- Device-lock contention profiling ---
# When $TT_DEVICE_TIMING_LOG is set, on EXIT we append one JSON line:
#   {source, pid, started_at_ms, wait_ms, run_ms, test_path, exit_code}
# wait_ms = script entry → flock acquired (contention)
# run_ms  = flock acquired → script exit  (device occupied)
# On sim, flock is skipped so we seed TT_TIMING_LOCK_ACQUIRED_MS=entry below;
# wait_ms is 0 and run_ms is the full pytest wall-clock. Skipped only when the
# script exits before that seed runs.
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
    # libttsim's own hang watchdog (clocks of no RISC-V / Tensix progress with
    # pending work before the sim _Exit(1)'s). User-set env wins.
    : "${TTSIM_HANG_WATCHDOG_CLOCKS:=50000}"
    export TTSIM_HANG_WATCHDOG_CLOCKS
    # No flock on sim, but we still want device_timings: seed the marker so the
    # exit trap emits with wait_ms=0 and run_ms=full wall-clock.
    TT_TIMING_LOCK_ACQUIRED_MS=$TT_TIMING_ENTRY_MS
fi
PYTEST_STDOUT_LOG="/tmp/safe-pytest-stdout-$$.log"

# --- Parse flags ---
DEV_MODE=false
FAIL_FAST=true
SIM_WORKERS=""
SIM_WORKERS_GIVEN=false
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
        *)
            break
            ;;
    esac
done

# --- Validate --sim-workers ---
if [[ "$SIM_WORKERS_GIVEN" == true ]]; then
    if [[ "$SIM_MODE" == false ]]; then
        echo "SAFE_PYTEST_ERROR: --sim-workers is only valid when TT_METAL_SIMULATOR is set"
        exit 3
    fi
    if ! [[ "$SIM_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
        echo "SAFE_PYTEST_ERROR: --sim-workers must be a positive integer (got: $SIM_WORKERS)"
        exit 3
    fi
fi

# --- Default sim worker count: 16 ---
if [[ "$SIM_MODE" == true && -z "$SIM_WORKERS" ]]; then
    SIM_WORKERS=16
fi

# --- Argument validation ---
if [[ $# -eq 0 ]]; then
    echo "SAFE_PYTEST_ERROR: No test path provided"
    echo "Usage: scripts/run_safe_pytest.sh [--dev] [--run-all] <test_path> [extra_pytest_args...]"
    exit 3
fi

TEST_PATH="$1"
TT_TIMING_TEST_PATH="$TEST_PATH"
shift

# --- Acquire flock (hardware only) ---
if [[ "$SIM_MODE" == false ]]; then
    exec 9>"$LOCK_FILE"

    echo "SAFE_PYTEST: Waiting for device lock..."

    # Find the PID holding an flock on a given path. Tries lslocks first
    # (fast, works in the global namespace); falls back to scanning /proc/*/fd
    # for processes with the lockfile open and an active FLOCK in fdinfo
    # (works inside PID namespaces where lslocks reports holder pid 0).
    _find_lock_holder() {
        local lock_path="$1" pid
        pid=$(lslocks --noheadings --raw --output PID,PATH 2>/dev/null \
            | awk -v p="$lock_path" '$2==p && $1!="0" {print $1; exit}')
        if [[ -n "$pid" ]]; then echo "$pid"; return 0; fi
        local pid_dir fd_link fd_num target
        for pid_dir in /proc/[0-9]*; do
            for fd_link in "$pid_dir"/fd/*; do
                [ -L "$fd_link" ] || continue
                target=$(readlink "$fd_link" 2>/dev/null) || continue
                [ "$target" = "$lock_path" ] || continue
                fd_num=${fd_link##*/}
                if grep -q '^lock:.*FLOCK' "$pid_dir/fdinfo/$fd_num" 2>/dev/null; then
                    echo "${pid_dir##*/}"
                    return 0
                fi
            done
        done
        return 1
    }

    LOCK_WAIT_INTERVAL=20
    LOCK_WAIT_TOTAL=0
    while ! flock -w "$LOCK_WAIT_INTERVAL" 9; do
        LOCK_WAIT_TOTAL=$((LOCK_WAIT_TOTAL + LOCK_WAIT_INTERVAL))
        TS="[$(date '+%Y-%m-%d %H:%M:%S')]"
        HOLDER_PID=$(_find_lock_holder "$LOCK_FILE")
        if [[ -n "$HOLDER_PID" && -d /proc/$HOLDER_PID ]]; then
            HOLDER_CMD=$(tr '\0' ' ' < /proc/$HOLDER_PID/cmdline 2>/dev/null | cut -c1-200)
            HOLDER_PPID=$(awk '{print $4}' /proc/$HOLDER_PID/stat 2>/dev/null)
            if [[ -n "$HOLDER_PPID" && "$HOLDER_PPID" -gt 1 && -d /proc/$HOLDER_PPID ]]; then
                HOLDER_PARENT_CMD=$(tr '\0' ' ' < /proc/$HOLDER_PPID/cmdline 2>/dev/null | cut -c1-150)
                echo "$TS SAFE_PYTEST: waiting for device (${LOCK_WAIT_TOTAL}s) — holder pid=$HOLDER_PID cmd=\"$HOLDER_CMD\" parent pid=$HOLDER_PPID cmd=\"$HOLDER_PARENT_CMD\""
            else
                echo "$TS SAFE_PYTEST: waiting for device (${LOCK_WAIT_TOTAL}s) — holder pid=$HOLDER_PID cmd=\"$HOLDER_CMD\""
            fi
        else
            echo "$TS SAFE_PYTEST: waiting for device (${LOCK_WAIT_TOTAL}s) — holder unknown"
        fi
    done
    TT_TIMING_LOCK_ACQUIRED_MS=$(date +%s%3N)
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
        echo "SAFE_PYTEST: WARNING: Failed to activate python_env virtual environment"
    fi
else
    echo "SAFE_PYTEST: WARNING: python_env not found; using system Python"
fi

# --- Pre-flight: pytest-xdist required for sim parallelism (SIM_WORKERS > 1) ---
if [[ "$SIM_MODE" == true && "$SIM_WORKERS" -gt 1 ]]; then
    if ! python3 -c "import xdist" 2>/dev/null; then
        echo "SAFE_PYTEST_ERROR: --sim-workers=${SIM_WORKERS} requires pytest-xdist."
        echo "                   Install with: pip install pytest-xdist"
        echo "                   Or pass --sim-workers 1 to run serially."
        exit 3
    fi
fi

# --- Hang detection setup (hardware only) ---
# On timeout, the dispatch layer runs tt-triage. Fires only on actual hang —
# zero overhead for passing tests. On sim there is no hang detection because
# wall-clock timeouts are meaningless at kHz clock speeds.
rm -f "$TRIAGE_LOG"
# Also clear any stale triage report from a previous run. Downstream consumers
# (hooks, CI) treat the report's presence as the hang signal — leaving a stale
# file around causes false-positive "hang detected" classification on the
# next ordinary test failure.
rm -f "$TRIAGE_REPORT"
MISSING_TTEXALENS=false
if [[ "$SIM_MODE" == false ]]; then
    export TT_METAL_OPERATION_TIMEOUT_SECONDS="$DISPATCH_TIMEOUT"
    # Requires tt-exalens: uv pip install -r tools/triage/requirements.txt
    # Defer the missing-tool warning to EXIT via trap — otherwise it gets buried
    # in pytest / triage output and users never see it.
    if ! python3 -c "import ttexalens" 2>/dev/null; then
        MISSING_TTEXALENS=true
    fi
    mkdir -p "${TRIAGE_OUT_DIR}"
    export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 ${TRIAGE_SCRIPT} --disable-progress --skip-version-check --llm-output --llm-output-path=${TRIAGE_REPORT} > ${TRIAGE_LOG} 2>&1"
fi

emit_missing_ttexalens_warning() {
    if [[ "$MISSING_TTEXALENS" == true ]]; then
        echo ""
        echo "SAFE_PYTEST: WARNING: tt-exalens not installed — triage on hang is unavailable."
        echo "SAFE_PYTEST: Install with: uv pip install -r tools/triage/requirements.txt"
    fi
}
trap '_emit_device_timing; emit_missing_ttexalens_warning' EXIT

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
        # NoC sanitizer is intentionally disabled on sim — the sanitizer is
        # tuned for HW behavior and hits false positives under libttsim.
        # (Mirrors tt-probe.sh.)
        export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1
        echo "SAFE_PYTEST: [sim+dev] asserts=ebreak llk_asserts=ON watcher=polling(noc_sanitize=OFF) watchdog=${TTSIM_HANG_WATCHDOG_CLOCKS} clocks workers=${SIM_WORKERS}"
    else
        echo "SAFE_PYTEST: [dev] asserts=ebreak llk_asserts=ON watcher=polling triage=ON timeout=${DISPATCH_TIMEOUT}s"
    fi
elif [[ "$SIM_MODE" == true ]]; then
    echo "SAFE_PYTEST: [sim] watchdog=${TTSIM_HANG_WATCHDOG_CLOCKS} clocks workers=${SIM_WORKERS}"
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
PYTEST_CMD=(pytest "${TEST_PATH}")
if [[ "$FAIL_FAST" == true ]]; then
    PYTEST_CMD+=(-x)
fi
# Sim parallelism via pytest-xdist. Each worker dlopens its own libttsim;
# DRAM is MAP_PRIVATE per-process so workers don't interfere. Skip when
# workers=1 to avoid xdist setup overhead.
if [[ "$SIM_MODE" == true && "$SIM_WORKERS" -gt 1 ]]; then
    PYTEST_CMD+=(-n "$SIM_WORKERS")
fi
PYTEST_CMD+=("$@")

# Signal handling: if this script is killed (e.g. parent process gets SIGTERM
# and we get reparented to init, or a watchdog kills us), forward SIGKILL to
# pytest and its descendants. Without this, pytest is orphaned with fd 9 ->
# /tmp/tt-device.lock and /dev/tenstorrent/* held, blocking all future runs.
CHILD_PID=
_signal_cleanup() {
    local sig=$1
    echo ""
    echo "SAFE_PYTEST: Caught SIG${sig} — killing pytest, marking device dirty"
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
# Mirror stdout/stderr to PYTEST_STDOUT_LOG via process substitution so we can
# grep for the libttsim watchdog message after a sim hang. Process substitution
# leaves $! pointing at pytest itself (not tee), so the signal trap still kills
# the right process tree.
"${PYTEST_CMD[@]}" > >(tee "$PYTEST_STDOUT_LOG") 2>&1 &
CHILD_PID=$!
wait "$CHILD_PID"
EXIT_CODE=$?
wait 2>/dev/null  # let tee flush before we grep
CHILD_PID=

echo "========================================"

# --- Handle result ---
if [[ $EXIT_CODE -eq 0 ]]; then
    rm -f "$DIRTY_FLAG"
    rm -f "$TRIAGE_LOG"
    rm -f "$PYTEST_STDOUT_LOG"
    echo "SAFE_PYTEST_RESULT: PASS"
    exit 0
fi

# Pytest setup errors (no device touched):
#   4 = usage error (bad args, nonexistent path)
#   5 = no tests collected (typo in path, bad marker filter, etc.)
if [[ $EXIT_CODE -eq 4 || $EXIT_CODE -eq 5 ]]; then
    rm -f "$DIRTY_FLAG"
    rm -f "$TRIAGE_LOG"
    rm -f "$PYTEST_STDOUT_LOG"
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
#   HW:  Triage log non-empty = dispatch timeout handler ran tt-triage.
#   Sim: pytest stdout contains "hang watchdog fired" = libttsim watchdog _Exit(1)'d.
# In the sim case we stage the watchdog message into TRIAGE_LOG so the HANG
# branch below dumps it identically to a HW triage report.
IS_HANG=false
if [[ -s "$TRIAGE_LOG" ]]; then
    IS_HANG=true
elif [[ "$SIM_MODE" == true ]] && grep -q "hang watchdog fired" "$PYTEST_STDOUT_LOG" 2>/dev/null; then
    IS_HANG=true
    grep -A4 "hang watchdog fired" "$PYTEST_STDOUT_LOG" > "$TRIAGE_LOG"
fi

# Only reset device when the failure might have left it dirty.
# Hangs and crashes corrupt device state. Normal test failures (PCC mismatch,
# assertion errors) and collection errors don't touch the device.
if [[ "$IS_HANG" == true ]]; then
    if [[ "$SIM_MODE" == false ]]; then
        echo "SAFE_PYTEST: Resetting device..."
        if tt-smi -r; then
            sleep 2
            rm -f "$DIRTY_FLAG"
            echo "SAFE_PYTEST: Device reset complete"
        else
            echo "SAFE_PYTEST: Device reset FAILED; leaving device marked dirty"
        fi
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

    # Print the triage report path as the last line so machine-readers can find it.
    if [[ -f "$TRIAGE_REPORT" ]]; then
        echo "SAFE_PYTEST: triage report: ${TRIAGE_REPORT}"
    fi

    rm -f "$TRIAGE_LOG"
    rm -f "$PYTEST_STDOUT_LOG"
    exit 2
fi

rm -f "$DIRTY_FLAG"
rm -f "$TRIAGE_LOG"
rm -f "$PYTEST_STDOUT_LOG"
# Note: $EXIT_CODE here is pytest's internal exit code (e.g. 1 = test failure,
# 2 = collection error / user interrupt). The wrapper's own exit code is
# always 1 for this branch — exit 2 is reserved for real dispatch-timeout
# hangs (handled above). The label keeps both visible to readers and to
# hooks parsing this output.
echo "SAFE_PYTEST_RESULT: FAIL (pytest exit code: $EXIT_CODE; wrapper exit: 1)"
exit 1
