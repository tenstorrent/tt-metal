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
#           lightweight ebreak asserts (ASSERT + LLK_ASSERT), and JSON triage
#           to generated/tt-triage/triage.json. Same semantics as run_safe_pytest.sh --dev.
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
#   --dev    - Debug mode: watcher + lightweight asserts + JSON triage on hang.
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
TRIAGE_JSON_DIR="${REPO_DIR}/generated/tt-triage"
TRIAGE_JSON="${TRIAGE_JSON_DIR}/triage.json"
LOCK_FILE="/tmp/tt-device.lock"
DIRTY_FLAG="/tmp/tt-device.dirty"
TRIAGE_LOG="/tmp/tt-probe-triage-$$.log"

# --- Parse flags ---
DEV_MODE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev)
            DEV_MODE=true
            shift
            ;;
        -*)
            echo "TT_PROBE_ERROR: Unknown flag: $1" >&2
            echo "Usage: scripts/tt-probe.sh [--dev] <op_name> << 'PYEOF'" >&2
            exit 3
            ;;
        *)
            break
            ;;
    esac
done

# --- Parse positional args ---
if [[ $# -eq 0 ]]; then
    echo "TT_PROBE_ERROR: No op name provided" >&2
    echo "Usage: scripts/tt-probe.sh [--dev] <op_name> << 'PYEOF'" >&2
    echo "       ... python code ..." >&2
    echo "       PYEOF" >&2
    exit 3
fi
OP_NAME="$1"

# --- Read stdin ---
SCRIPT=$(cat)
if [[ -z "$SCRIPT" ]]; then
    echo "TT_PROBE_ERROR: No script provided on stdin" >&2
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
echo "TT_PROBE: Saved → ${PROBE_REL}" >&2

# --- Detect simulator mode ---
SIM_MODE=false
if [[ -n "${TT_METAL_SIMULATOR:-}" ]]; then
    SIM_MODE=true
    export TT_METAL_SLOW_DISPATCH_MODE=1
    export TT_METAL_DISABLE_SFPLOADMACRO=1
fi

# --- Timeout config ---
if [[ "$SIM_MODE" == true ]]; then
    DISPATCH_TIMEOUT=0
    SIM_CYCLE_TIMEOUT=${TT_METAL_SIM_CYCLE_TIMEOUT:-100000000}
    SIM_WALL_TIMEOUT=${TT_PROBE_SIM_TIMEOUT_SECONDS:-1800}
    export TT_METAL_SIM_CYCLE_TIMEOUT="$SIM_CYCLE_TIMEOUT"
else
    DISPATCH_TIMEOUT=5
fi
export TT_METAL_OPERATION_TIMEOUT_SECONDS="$DISPATCH_TIMEOUT"

# --- Hang detection setup ---
# Triage writes JSON to generated/tt-triage/triage.json for machine-readable
# inspection (same location run_safe_pytest.sh uses), plus a text log for the stderr
# dump. Grep targets are documented in .claude/CLAUDE.md § "Hang triage".
rm -f "$TRIAGE_LOG"
MISSING_TTEXALENS=false
if [[ "$SIM_MODE" == true ]]; then
    export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="echo HANG > ${TRIAGE_LOG}"
else
    if python3 -c "import ttexalens" 2>/dev/null; then
        mkdir -p "${TRIAGE_JSON_DIR}"
        export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="python3 ${TRIAGE_SCRIPT} --disable-progress --skip-version-check --json-path=${TRIAGE_JSON} > ${TRIAGE_LOG} 2>&1"
    else
        # Defer the missing-tool warning to EXIT via trap — otherwise it gets
        # buried in probe output and users never see it.
        MISSING_TTEXALENS=true
        export TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE="echo HANG_NO_TRIAGE > ${TRIAGE_LOG}"
    fi
fi

emit_missing_ttexalens_warning() {
    if [[ "$MISSING_TTEXALENS" == true ]]; then
        echo "" >&2
        echo "TT_PROBE: WARNING: tt-exalens not installed — triage on hang is unavailable." >&2
        echo "TT_PROBE: Install with: uv pip install -r tools/triage/requirements.txt" >&2
    fi
}
trap emit_missing_ttexalens_warning EXIT

# --- Dev-mode instrumentation ---
# Mirrors run_safe_pytest.sh --dev exactly. On hardware this enables ebreak ASSERTs and
# the polling watcher; on sim only the watcher runs (ebreak asserts are useless
# without a triage mechanism on sim).
if [[ "$DEV_MODE" == true ]]; then
    export TT_METAL_WATCHER=1
    export TT_METAL_WATCHER_NOINLINE=1
    export TT_METAL_WATCHER_DISABLE_DISPATCH=1

    if [[ "$SIM_MODE" == true ]]; then
        export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1
        echo "TT_PROBE: [sim+dev] watcher=polling(+assert,noc_sanitize=OFF) triage=OFF cycle_timeout=${SIM_CYCLE_TIMEOUT} outer_timeout=${SIM_WALL_TIMEOUT}s" >&2
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
        echo "TT_PROBE: [dev] asserts=ebreak llk_asserts=ON watcher=polling triage=ON timeout=${DISPATCH_TIMEOUT}s" >&2
    fi
fi

# --- Acquire flock (hardware only) ---
if [[ "$SIM_MODE" == false ]]; then
    exec 9>"$LOCK_FILE"
    LOCK_TIMEOUT=600
    echo "TT_PROBE: Waiting for device lock..." >&2
    if ! flock -w "$LOCK_TIMEOUT" 9; then
        echo "TT_PROBE_ERROR: Could not acquire device lock after ${LOCK_TIMEOUT}s" >&2
        exit 3
    fi
    echo "TT_PROBE: Device lock acquired" >&2
    export TT_DEVICE_LOCK_HELD=1

    if [[ -f "$DIRTY_FLAG" ]]; then
        echo "TT_PROBE: Device dirty from previous run, resetting..." >&2
        if ! tt-smi -r; then
            echo "TT_PROBE_ERROR: Device reset failed" >&2
            exit 3
        fi
        rm -f "$DIRTY_FLAG"
        echo "TT_PROBE: Device reset complete" >&2
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

echo "TT_PROBE: python3 ${PROBE_REL}" >&2
if [[ "$SIM_MODE" == true ]]; then
    echo "TT_PROBE: [sim] cycle_timeout=${SIM_CYCLE_TIMEOUT} outer_timeout=${SIM_WALL_TIMEOUT}s" >&2
fi
echo "========================================" >&2

if [[ "$SIM_MODE" == true ]]; then
    timeout --signal=TERM --kill-after=10 "$SIM_WALL_TIMEOUT" python3 "$PROBE_FILE"
    EXIT_CODE=$?
    if [[ $EXIT_CODE -eq 124 || $EXIT_CODE -eq 137 ]]; then
        echo "HANG" > "$TRIAGE_LOG"
    fi
else
    python3 "$PROBE_FILE"
    EXIT_CODE=$?
fi

echo "========================================" >&2

# --- Cleanup: kill orphans ---
if [[ $EXIT_CODE -ne 0 ]]; then
    for child_pid in $(pgrep -P $$ 2>/dev/null); do
        pkill -9 -P "$child_pid" 2>/dev/null || true
        kill -9 "$child_pid" 2>/dev/null || true
    done
fi

# --- Reset device (hardware only) ---
if [[ "$SIM_MODE" == false ]]; then
    echo "TT_PROBE: Resetting device..." >&2
    if tt-smi -r; then
        sleep 2
        rm -f "$DIRTY_FLAG"
        echo "TT_PROBE: Device reset complete" >&2
    else
        echo "TT_PROBE: Device reset FAILED; leaving dirty" >&2
    fi
fi

# --- Detect hang ---
if [[ -s "$TRIAGE_LOG" ]]; then
    echo "TT_PROBE: HANG DETECTED" >&2
    if [[ "$SIM_MODE" == false ]]; then
        cat "$TRIAGE_LOG" >&2
        if [[ -s "$TRIAGE_JSON" ]]; then
            echo "TT_PROBE: JSON triage: ${TRIAGE_JSON}" >&2
        fi
    fi
    rm -f "$TRIAGE_LOG"
    echo "TT_PROBE_RESULT: HANG (probe: ${PROBE_REL})" >&2
    exit 2
fi
rm -f "$TRIAGE_LOG"

# --- Result ---
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "TT_PROBE_RESULT: PASS" >&2
else
    echo "TT_PROBE_RESULT: FAIL (exit $EXIT_CODE, probe: ${PROBE_REL})" >&2
fi
exit $EXIT_CODE
