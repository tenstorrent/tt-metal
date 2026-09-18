#!/bin/bash
# tt-probe.sh — Save-and-run inline debug scripts with device safety
#
# Drop-in replacement for raw "python3 << 'PYEOF'" heredocs.
# Reads a Python script from stdin, saves it to disk, and runs it with
# the same device protections as run_safe_pytest.sh (per-card lock, timeout,
# reset). Shared plumbing lives in scripts/lib/tt_safe_run.sh.
#
# Saved scripts persist as debugging artifacts — they document what was
# tried and can be re-run later.
#
# Usage: scripts/tt-probe.sh [--device N|auto] [--mesh] [--dev] <op_name> << 'PYEOF'
#   import torch, ttnn
#   ...
# PYEOF
#
# Flags:
#   --device N|auto  Which card to run on (default auto = lowest FREE card; N = wait for
#                    that UMD card id; see scripts/lib/tt_device_pool.sh). The probe sees
#                    its card as device 0. Logs / triage under generated/dev<N>/.
#   --mesh           Take every card (multi-device probes). Logs under generated/mesh/.
#   --dev   Enables polling watcher (NoC sanitizer, waypoints, CB sanitization),
#           lightweight ebreak asserts (ASSERT + LLK_ASSERT), and an llm-friendly triage report
#           to generated/dev<N>/tt-triage/triage.txt (legacy generated/tt-triage/triage.txt is a
#           symlink to the latest). Same semantics as run_safe_pytest.sh --dev.
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
#
# Exit codes (same as run_safe_pytest.sh):
#   0 - Script completed successfully
#   1 - Script failed (exception, assertion, non-zero exit)
#   2 - Hang detected (dispatch timeout)
#   3 - Setup error

set -o pipefail

TTRUN_PREFIX="TT_PROBE"
# shellcheck source=scripts/lib/tt_safe_run.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/tt_safe_run.sh"
ttrun_init tt-probe
PROBE_STDOUT_LOG="/tmp/tt-probe-stdout-$$.log"
USAGE="Usage: scripts/tt-probe.sh [--device N|auto] [--mesh] [--dev] <op_name> << 'PYEOF'"

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev) DEV_MODE=true; shift ;;
        --device)
            [[ $# -ge 2 ]] || { echo "TT_PROBE_ERROR: --device requires an argument (UMD card id or 'auto')"; exit 3; }
            DEVICE_SELECTOR="$2"; shift 2 ;;
        --device=*) DEVICE_SELECTOR="${1#*=}"; shift ;;
        --mesh) MESH_MODE=true; shift ;;
        -*)
            echo "TT_PROBE_ERROR: Unknown flag: $1"
            echo "$USAGE"
            exit 3
            ;;
        *) break ;;
    esac
done
ttrun_resolve_selector || exit 3

# --- Parse positional args ---
if [[ $# -eq 0 ]]; then
    echo "TT_PROBE_ERROR: No op name provided"
    echo "$USAGE"
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

# --- Acquire a card, set up the environment ---
ttrun_acquire_card || exit 3
ttrun_activate_venv
[[ "$DEV_MODE" == true ]] && ttrun_export_dev_env
ttrun_print_mode_banner
ttrun_setup_hang_hook

# --- Run ---
ttrun_mark_dirty
echo "TT_PROBE: python3 ${PROBE_REL}"
echo "========================================"
ttrun_run_child "$PROBE_STDOUT_LOG" python3 "$PROBE_FILE"
echo "========================================"

# --- Result ---
# Probes always reset the card(s) they held (a probe is by definition exploratory
# and may leave device state behind); the other cards are untouched.
ttrun_reset_cards
ttrun_detect_hang "$PROBE_STDOUT_LOG"
if [[ "$IS_HANG" == true ]]; then
    echo "TT_PROBE: HANG DETECTED"
    ttrun_dump_hang_logs
    ttrun_cleanup_tmp "$PROBE_STDOUT_LOG"
    echo "TT_PROBE_RESULT: HANG (probe: ${PROBE_REL})"
    exit 2
fi
ttrun_cleanup_tmp "$PROBE_STDOUT_LOG"
# Wrapper exit 2 is reserved for HANG; any other non-zero Python exit is normalized
# to 1 so consumers can rely on `wrapper-exit == 2 -> hang`.
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "TT_PROBE_RESULT: PASS"
    exit 0
fi
echo "TT_PROBE_RESULT: FAIL (python exit code: $EXIT_CODE; wrapper exit: 1, probe: ${PROBE_REL})"
exit 1
