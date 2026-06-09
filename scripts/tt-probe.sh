#!/bin/bash
# tt-probe.sh — Save-and-run inline debug scripts with device safety
#
# Drop-in replacement for raw "python3 << 'PYEOF'" heredocs.
# Reads a Python script from stdin, saves it to disk, and runs it with
# the same device protections as run_safe_pytest.sh (flock, timeout, reset).
#
# Saved scripts persist as debugging artifacts — they document what was
# tried and can be re-run later.
#
# Usage: scripts/tt-probe.sh [--dev] <op_name> << 'PYEOF'
#   import torch, ttnn
#   ...
# PYEOF
#
# Flags:
#   --dev   Enables polling watcher (NoC sanitizer, waypoints, CB sanitization),
#           lightweight ebreak asserts (ASSERT + LLK_ASSERT), and an llm-friendly triage report
#           to generated/tt-triage/triage.txt. Same semantics as run_safe_pytest.sh --dev.
#
# With DPRINT:
#   TT_METAL_DPRINT_CORES=0,0 TT_METAL_DPRINT_RISCVS=TR0 \
#     scripts/tt-probe.sh <op_name> << 'PYEOF'
#   ...
#   PYEOF
#
# Saved to: tests/ttnn/unit_tests/operations/<op_name>/probes/probe_NNN.py
#
# Modes:
#   default  - Dispatch timeout only. Lean, no debug overhead.
#   --dev    - Debug mode: watcher + lightweight asserts + llm-friendly triage on hang.
#              Probes a suspected hang or bad kernel state, letting ASSERT()/
#              LLK_ASSERT() halt at the exact failing instruction for triage.
#
# Exit codes (same as run_safe_pytest.sh):
#   0 - Script completed successfully
#   1 - Script failed (exception, assertion, non-zero exit)
#   2 - Hang detected (dispatch timeout)
#   3 - Setup error

set -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRIAGE_SCRIPT="${REPO_DIR}/tools/tt-triage.py"
TRIAGE_OUT_DIR="${REPO_DIR}/generated/tt-triage"
TRIAGE_REPORT="${TRIAGE_OUT_DIR}/triage.txt"
LOCK_FILE="/tmp/tt-device.lock"
DIRTY_FLAG="/tmp/tt-device.dirty"
TRIAGE_LOG="/tmp/tt-probe-triage-$$.log"

# --- Device-lock contention profiling (see run_safe_pytest.sh for details) ---
TT_TIMING_ENTRY_MS=$(date +%s%3N)
TT_TIMING_LOCK_ACQUIRED_MS=0
TT_TIMING_SOURCE="tt-probe"
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
        esc_path="${TT_TIMING_TEST_PATH//\\/\\\\}"
        esc_path="${esc_path//\"/\\\"}"
        printf '{"source":"%s","pid":%d,"started_at_ms":%s,"wait_ms":%d,"run_ms":%d,"test_path":"%s","exit_code":%d}\n' \
            "$TT_TIMING_SOURCE" "$$" "$TT_TIMING_ENTRY_MS" "$wait_ms" "$run_ms" "$esc_path" "$ec" \
            >> "$TT_DEVICE_TIMING_LOG" 2>/dev/null || true
    fi
    return $ec
}
trap _emit_device_timing EXIT

# --- Parse flags ---
DEV_MODE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV_MODE=true
            shift
            ;;
        -*)
            echo "TT_PROBE_ERROR: Unknown flag: $1"
            echo "Usage: scripts/tt-probe.sh [--dev] <op_name> << 'PYEOF'"
            exit 3
            ;;
        *)
            break
            ;;
    esac
done

# --- Parse positional args ---
if [[ $# -eq 0 ]]; then
    echo "TT_PROBE_ERROR: No op name provided"
    echo "Usage: scripts/tt-probe.sh [--dev] <op_name> << 'PYEOF'"
    echo "       ... python code ..."
    echo "       PYEOF"
    exit 3
fi
OP_NAME="$1"
TT_TIMING_TEST_PATH="$OP_NAME"

# --- Read stdin ---
SCRIPT=$(cat)
if [[ -z "$SCRIPT" ]]; then
    echo "TT_PROBE_ERROR: No script provided on stdin"
    exit 3
fi

# --- Save to disk ---
PROBE_DIR="${REPO_DIR}/tests/ttnn/unit_tests/operations/${OP_NAME}/probes"
mkdir -p "$PROBE_DIR"

NEXT_NUM=1
while [[ -f "${PROBE_DIR}/probe_$(printf '%03d' $NEXT_NUM).py" ]]; do
    ((NEXT_NUM++))
done
PROBE_FILE="${PROBE_DIR}/probe_$(printf '%03d' $NEXT_NUM).py"

printf '%s\n' "$SCRIPT" > "$PROBE_FILE"
PROBE_REL="${PROBE_FILE#$REPO_DIR/}"
echo "TT_PROBE: Saved → ${PROBE_REL}"

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
fi
PROBE_STDOUT_LOG="/tmp/tt-probe-stdout-$$.log"

# --- Timeout config ---
DISPATCH_TIMEOUT=5
if [[ "$SIM_MODE" == false ]]; then
    export TT_METAL_OPERATION_TIMEOUT_SECONDS="$DISPATCH_TIMEOUT"
fi

# --- Hang detection setup ---
# Triage writes an llm-friendly report to generated/tt-triage/triage.txt for
# machine-readable inspection (same location run_safe_pytest.sh uses), plus a
# text log for the stderr dump. Grep targets are documented in CLAUDE.md § "Hang triage".
rm -f "$TRIAGE_LOG"
# Also clear any stale triage report from a previous run. Downstream consumers
# (hooks, CI) treat the report's presence as the hang signal — leaving a stale
# file around causes false-positive "hang detected" classification on the
# next ordinary probe failure.
rm -f "$TRIAGE_REPORT"
MISSING_TTEXALENS=false
# On sim, TT_METAL_OPERATION_TIMEOUT_SECONDS is unset, so the dispatch-timeout
# command never fires — sim hangs are caught by the libttsim watchdog below,
# not this hook.
if [[ "$SIM_MODE" == false ]]; then
    if python3 -c "import ttexalens" 2>/dev/null; then
        mkdir -p "${TRIAGE_OUT_DIR}"
        export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 ${TRIAGE_SCRIPT} --disable-progress --skip-version-check --llm-output --llm-output-path=${TRIAGE_REPORT} > ${TRIAGE_LOG} 2>&1"
    else
        # Defer the missing-tool warning to EXIT via trap — otherwise it gets
        # buried in probe output and users never see it.
        MISSING_TTEXALENS=true
        export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="echo HANG_NO_TRIAGE > ${TRIAGE_LOG}"
    fi
fi

emit_missing_ttexalens_warning() {
    if [[ "$MISSING_TTEXALENS" == true ]]; then
        echo ""
        echo "TT_PROBE: WARNING: tt-exalens not installed — triage on hang is unavailable."
        echo "TT_PROBE: Install with: uv pip install -r tools/triage/requirements.txt"
    fi
}
trap '_emit_device_timing; emit_missing_ttexalens_warning' EXIT

# --- Dev-mode instrumentation ---
# Mirrors run_safe_pytest.sh --dev. On hardware this enables ebreak ASSERTs and
# the polling watcher; on sim only the watcher runs (ebreak asserts are useless
# without a triage mechanism on sim).
if [[ "$DEV_MODE" == true ]]; then
    export TT_METAL_WATCHER=1
    export TT_METAL_WATCHER_NOINLINE=1
    export TT_METAL_WATCHER_DISABLE_DISPATCH=1

    if [[ "$SIM_MODE" == true ]]; then
        # NoC sanitizer is intentionally disabled on sim — see comment in
        # safe_pytest.sh for symmetry. (Kept off here because sim hits false
        # positives the sanitizer is tuned for HW behavior on.)
        export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1
        echo "TT_PROBE: [sim+dev] watcher=polling(+assert,noc_sanitize=OFF) watchdog=${TTSIM_HANG_WATCHDOG_CLOCKS} clocks"
    else
        # Lightweight asserts: compile ASSERT() as ebreak so the core halts at
        # the exact instruction. The dispatch timeout then fires triage, which
        # captures callstacks from ALL cores (assert site + anything waiting on it).
        export TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1
        # LLK asserts: LLK_ASSERT() in the compute API / LLK layer.
        export TT_METAL_LLK_ASSERTS=1
        # Let ebreak asserts halt the core for triage to capture — bypass watcher's
        # own assert handling so all cores' callstacks get collected together.
        export TT_METAL_WATCHER_DISABLE_ASSERT=1
        echo "TT_PROBE: [dev] asserts=ebreak llk_asserts=ON watcher=polling triage=ON timeout=${DISPATCH_TIMEOUT}s"
    fi
fi

# --- Acquire flock (hardware only) ---
if [[ "$SIM_MODE" == false ]]; then
    exec 9>"$LOCK_FILE"
    echo "TT_PROBE: Waiting for device lock..."
    flock 9
    TT_TIMING_LOCK_ACQUIRED_MS=$(date +%s%3N)
    echo "TT_PROBE: Device lock acquired"

    if [[ -f "$DIRTY_FLAG" ]]; then
        echo "TT_PROBE: Device dirty from previous run, resetting..."
        if ! tt-smi -r; then
            echo "TT_PROBE_ERROR: Device reset failed"
            exit 3
        fi
        rm -f "$DIRTY_FLAG"
        echo "TT_PROBE: Device reset complete"
    fi
fi

# --- Activate venv ---
cd "$REPO_DIR"
if [[ -f python_env/bin/activate ]]; then
    source python_env/bin/activate
fi

# --- Mark dirty and run ---
if [[ "$SIM_MODE" == false ]]; then
    touch "$DIRTY_FLAG"
fi

echo "TT_PROBE: python3 ${PROBE_REL}"
echo "========================================"

# Signal handling: forward SIGKILL to python3 child + descendants if this
# script is killed (parent SIGTERM, watchdog, etc.). Without this, python3
# is orphaned holding fd 9 -> /tmp/tt-device.lock and /dev/tenstorrent/*,
# blocking all future runs.
CHILD_PID=
_signal_cleanup() {
    local sig=$1
    echo ""
    echo "TT_PROBE: Caught SIG${sig} — killing probe, marking device dirty"
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

# Run in background + wait so the trap can fire on signal (bash blocks
# signal delivery during synchronous foreground commands).
# Mirror stdout/stderr to PROBE_STDOUT_LOG via process substitution so we can
# grep for the libttsim watchdog message after a sim hang. Process substitution
# leaves $! pointing at python3 itself (not tee).
python3 "$PROBE_FILE" > >(tee "$PROBE_STDOUT_LOG") 2>&1 &
CHILD_PID=$!
wait "$CHILD_PID"
EXIT_CODE=$?
wait 2>/dev/null  # let tee flush before we grep
CHILD_PID=

echo "========================================"

# --- Cleanup: kill orphans ---
if [[ $EXIT_CODE -ne 0 ]]; then
    for child_pid in $(pgrep -P $$ 2>/dev/null); do
        pkill -9 -P "$child_pid" 2>/dev/null || true
        kill -9 "$child_pid" 2>/dev/null || true
    done
fi

# --- Reset device (hardware only) ---
if [[ "$SIM_MODE" == false ]]; then
    echo "TT_PROBE: Resetting device..."
    if tt-smi -r; then
        sleep 2
        rm -f "$DIRTY_FLAG"
        echo "TT_PROBE: Device reset complete"
    else
        echo "TT_PROBE: Device reset FAILED; leaving dirty"
    fi
fi

# --- Detect hang ---
# HW:  dispatch-timeout handler populated TRIAGE_LOG with tt-triage output.
# Sim: libttsim watchdog _Exit(1)'d the child; stage its message into
#      TRIAGE_LOG so the dump below treats it uniformly.
if [[ "$SIM_MODE" == true ]] && grep -q "hang watchdog fired" "$PROBE_STDOUT_LOG" 2>/dev/null; then
    grep -A4 "hang watchdog fired" "$PROBE_STDOUT_LOG" > "$TRIAGE_LOG"
fi
if [[ -s "$TRIAGE_LOG" ]]; then
    echo "TT_PROBE: HANG DETECTED"
    cat "$TRIAGE_LOG"
    if [[ "$SIM_MODE" == false && -s "$TRIAGE_REPORT" ]]; then
        echo "TT_PROBE: triage report: ${TRIAGE_REPORT}"
    fi
    rm -f "$TRIAGE_LOG"
    rm -f "$PROBE_STDOUT_LOG"
    echo "TT_PROBE_RESULT: HANG (probe: ${PROBE_REL})"
    exit 2
fi
rm -f "$TRIAGE_LOG"
rm -f "$PROBE_STDOUT_LOG"

# --- Result ---
# Reserve wrapper exit 2 for the HANG case above. For any other non-zero
# Python exit, normalize the wrapper exit to 1 so that downstream consumers
# (hooks, CI) can rely on `wrapper-exit == 2 → hang`.
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "TT_PROBE_RESULT: PASS"
    exit 0
else
    echo "TT_PROBE_RESULT: FAIL (python exit code: $EXIT_CODE; wrapper exit: 1, probe: ${PROBE_REL})"
    exit 1
fi
