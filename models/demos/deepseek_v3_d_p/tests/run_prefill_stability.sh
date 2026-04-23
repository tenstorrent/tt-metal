#!/usr/bin/env bash
# Stability loop for test_prefill_transformer.
#
# Purpose: detect pytest hangs caused by the preceding tt-smi -glx_reset
# leaving the hardware in a bad state.
#
# Each iteration:
#   1. Run the selected pytest node under a 10-minute wall-clock deadline.
#   2. If it finishes cleanly (pass or fail) → run tt-smi -glx_reset and
#      continue to the next iteration.
#   3. If it hangs past the deadline → log "PYTEST HANG DETECTED", leave
#      the pytest process alive for inspection, skip the reset, and exit.
#
# All stdout/stderr is mirrored to a log file.
#
# Usage:
#   ./run_prefill_stability.sh <num_iterations> [log_file]
#
# Must be run from the tt-metal repo root (pytest picks up conftest.py).
#
# Post-hang inspection (pytest pid is logged before exit):
#   py-spy dump --pid <pid>
#   gdb -p <pid>
#   cat /proc/<pid>/status
# When done:
#   kill -9 <pid>
#   tt-smi -glx_reset

set -u

NUM_ITERATIONS="${1:?usage: $0 <num_iterations> [log_file]}"
LOG_FILE="${2:-stability_prefill_$(date +%Y%m%d_%H%M%S).log}"

PYTEST_TIMEOUT_SEC=600
POLL_INTERVAL_SEC=5

export TT_DS_PREFILL_TTNN_CACHE="/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill"
export DEEPSEEK_V3_HF_MODEL="/mnt/models/deepseek-ai/DeepSeek-R1-0528"
export TT_DS_PREFILL_HOST_REF_CACHE="/mnt/models/deepseek-prefill-cache/golden/"

TEST_PATH="models/demos/deepseek_v3_d_p/tests/test_prefill_transformer.py"
TEST_FILTER='pretrained and smoke and e256_cf32_device and mesh-8x4 and 61 and longbook_qa_eng and 1024'

pass_count=0
fail_count=0
hang_count=0
reset_fail_count=0
hang_iter=""
hang_pid=""

log() { printf '%s\n' "$*" | tee -a "${LOG_FILE}"; }

on_exit() {
    log ""
    log "=================================================="
    log "Stability run ended at $(date -Is)"
    log "Pass:            ${pass_count}"
    log "Fail (non-hang): ${fail_count}"
    log "Hangs:           ${hang_count}${hang_iter:+ at iter ${hang_iter}, pytest pid=${hang_pid} (still running)}"
    log "Reset errors:    ${reset_fail_count}"
    log "Total runs:      $((pass_count + fail_count + hang_count))"
    log "Log file:        ${LOG_FILE}"
    log "=================================================="
}
trap on_exit EXIT INT TERM

log "=================================================="
log "Stability run started at $(date -Is)"
log "Iterations:             ${NUM_ITERATIONS}"
log "Pytest deadline (s):    ${PYTEST_TIMEOUT_SEC}"
log "TT_DS_PREFILL_TTNN_CACHE=${TT_DS_PREFILL_TTNN_CACHE}"
log "DEEPSEEK_V3_HF_MODEL=${DEEPSEEK_V3_HF_MODEL}"
log "TT_DS_PREFILL_HOST_REF_CACHE=${TT_DS_PREFILL_HOST_REF_CACHE}"
log "Test path:              ${TEST_PATH}"
log "Filter:                 ${TEST_FILTER}"
log "=================================================="

for (( i=1; i<=NUM_ITERATIONS; i++ )); do
    log ""
    log "------- Iteration ${i}/${NUM_ITERATIONS} @ $(date -Is) -------"

    iter_start=$(date +%s)
    deadline=$(( iter_start + PYTEST_TIMEOUT_SEC ))

    # Launch pytest in background, tee'ing output into the log.
    # Process substitution (> >(tee ...)) lets us capture pytest's PID via $!
    # rather than tee's, so we can poll/leave-alive the right process.
    pytest -xvs "${TEST_PATH}" -k "${TEST_FILTER}" \
        > >(tee -a "${LOG_FILE}") 2>&1 &
    pytest_pid=$!
    # Protect from SIGHUP if the shell ever sets huponexit, so the process
    # can survive the script exit for inspection.
    disown -h "${pytest_pid}" 2>/dev/null || true
    log "Launched pytest pid=${pytest_pid} (deadline ${PYTEST_TIMEOUT_SEC}s)"

    hung=0
    while kill -0 "${pytest_pid}" 2>/dev/null; do
        now=$(date +%s)
        if (( now >= deadline )); then
            hung=1
            break
        fi
        sleep "${POLL_INTERVAL_SEC}"
    done

    iter_end=$(date +%s)
    iter_duration=$(( iter_end - iter_start ))

    if [[ "${hung}" -eq 1 ]]; then
        hang_count=$((hang_count + 1))
        hang_iter="${i}"
        hang_pid="${pytest_pid}"
        log ""
        log "############################################################"
        log "### PYTEST HANG DETECTED at iteration ${i}/${NUM_ITERATIONS}"
        log "### duration=${iter_duration}s (deadline=${PYTEST_TIMEOUT_SEC}s)"
        log "### pytest pid=${pytest_pid} LEFT RUNNING for inspection"
        log "### NOT issuing tt-smi -glx_reset; HW state preserved"
        log "###"
        log "### To inspect the hung process:"
        log "###   py-spy dump --pid ${pytest_pid}"
        log "###   gdb -p ${pytest_pid}"
        log "###   cat /proc/${pytest_pid}/status"
        log "###   ls -l /proc/${pytest_pid}/fd"
        log "### When done:"
        log "###   kill -9 ${pytest_pid}"
        log "###   tt-smi -glx_reset"
        log "############################################################"
        exit 2
    fi

    # Pytest exited on its own — collect status and classify.
    wait "${pytest_pid}"
    pytest_exit=$?

    case "${pytest_exit}" in
        0)
            pass_count=$((pass_count + 1))
            verdict="PASS"
            ;;
        *)
            fail_count=$((fail_count + 1))
            verdict="FAIL (exit=${pytest_exit})"
            ;;
    esac
    log "Iteration ${i} pytest verdict: ${verdict}, duration=${iter_duration}s"

    # Reset runs unconditionally after a clean pytest exit (pass or fail) —
    # stressing the reset path is the whole point of the loop.
    log "Running tt-smi -glx_reset (unconditional after non-hang exit)..."
    tt-smi -glx_reset 2>&1 | tee -a "${LOG_FILE}"
    reset_exit=${PIPESTATUS[0]}
    if [[ ${reset_exit} -ne 0 ]]; then
        reset_fail_count=$((reset_fail_count + 1))
        log "tt-smi -glx_reset FAILED (exit=${reset_exit})"
    else
        log "tt-smi -glx_reset OK"
    fi
done
