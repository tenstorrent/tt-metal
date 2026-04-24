#!/usr/bin/env bash
# Stability loop for test_mla on a multi-host 32x4 quad BH galaxy.
#
# Purpose: detect pytest hangs caused by the preceding tt-smi -glx_reset
# leaving the hardware in a bad state.
#
# Each iteration:
#   1. Launch the test under tt-run + MPI with a wall-clock deadline.
#   2. If it finishes cleanly (pass or fail) -> run
#      `mpirun --host $HOSTS tt-smi -glx_reset` and continue.
#   3. If it hangs past the deadline -> log "PYTEST HANG DETECTED", leave
#      the tt-run process alive for inspection, skip the reset, and exit.
#
# All stdout/stderr is mirrored to a log file.
#
# Usage:
#   HOSTS=host1,host2,host3,host4 \
#     ./run_mla_stability_multihost.sh <num_iterations> [log_file]
#
# Required env vars:
#   HOSTS                comma-separated MPI host list (no default; must be set)
#
# Overridable env vars (exported or one-shot prefixed):
#   RANK_BINDING         rank-binding yaml for tt-run
#                        (default: tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml)
#   RANKFILE             MPI --rankfile path
#                        (default: /data/ipotkonjak/32x4_quad_galaxy_rankfile)
#   TEST_PATH            pytest file path
#                        (default: models/demos/deepseek_v3_d_p/tests/test_mla.py)
#   TEST_FILTER          pytest -k expression
#                        (default: '32x4 and line and random and max_sl and seq100k and skip_check and balanced and loops20')
#   PYTEST_TIMEOUT_SEC   per-iteration deadline in seconds (default: 600)
#   POLL_INTERVAL_SEC    how often to check liveness (default: 5)
#
# Example:
#   HOSTS=h1,h2,h3,h4 PYTEST_TIMEOUT_SEC=1800 \
#     ./run_mla_stability_multihost.sh 20 /tmp/run.log
#
# Must be run from the tt-metal repo root (pytest picks up conftest.py).
#
# Post-hang inspection (tt-run pid is logged before exit):
#   py-spy dump --pid <pid>
#   gdb -p <pid>
#   cat /proc/<pid>/status
# When done:
#   kill -9 <pid>
#   mpirun --host $HOSTS tt-smi -glx_reset

set -u

NUM_ITERATIONS="${1:?usage: $0 <num_iterations> [log_file]}"
LOG_FILE="${2:-stability_mla_multihost_$(date +%Y%m%d_%H%M%S).log}"

: "${HOSTS:?HOSTS must be set (comma-separated MPI host list)}"

PYTEST_TIMEOUT_SEC="${PYTEST_TIMEOUT_SEC:-600}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-5}"

RANK_BINDING="${RANK_BINDING:-tests/tt_metal/distributed/config/32x4_quad_bh_galaxy_rank_bindings.yaml}"
RANKFILE="${RANKFILE:-/data/ipotkonjak/32x4_quad_galaxy_rankfile}"
TEST_PATH="${TEST_PATH:-models/demos/deepseek_v3_d_p/tests/test_mla.py}"
TEST_FILTER="${TEST_FILTER:-32x4 and line and random and max_sl and seq100k and skip_check and balanced and loops20}"

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
    log "Hangs:           ${hang_count}${hang_iter:+ at iter ${hang_iter}, tt-run pid=${hang_pid} (still running)}"
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
log "Poll interval (s):      ${POLL_INTERVAL_SEC}"
log "HOSTS:                  ${HOSTS}"
log "Rank binding:           ${RANK_BINDING}"
log "Rankfile:               ${RANKFILE}"
log "Test path:              ${TEST_PATH}"
log "Filter:                 ${TEST_FILTER}"
log "=================================================="

for (( i=1; i<=NUM_ITERATIONS; i++ )); do
    log ""
    log "------- Iteration ${i}/${NUM_ITERATIONS} @ $(date -Is) -------"

    iter_start=$(date +%s)
    deadline=$(( iter_start + PYTEST_TIMEOUT_SEC ))

    # Launch tt-run in background, tee'ing output into the log.
    # Process substitution (> >(tee ...)) lets us capture tt-run's PID via $!
    # rather than tee's, so we can poll/leave-alive the right process.
    tt-run \
        --rank-binding "${RANK_BINDING}" \
        --mpi-args "--host ${HOSTS} --rankfile ${RANKFILE} --bind-to none --tag-output" \
        python -m pytest "${TEST_PATH}" -k "${TEST_FILTER}" \
        > >(tee -a "${LOG_FILE}") 2>&1 &
    ttrun_pid=$!
    # Protect from SIGHUP if the shell ever sets huponexit, so the process
    # can survive the script exit for inspection.
    disown -h "${ttrun_pid}" 2>/dev/null || true
    log "Launched tt-run pid=${ttrun_pid} (deadline ${PYTEST_TIMEOUT_SEC}s)"

    hung=0
    while kill -0 "${ttrun_pid}" 2>/dev/null; do
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
        hang_pid="${ttrun_pid}"
        log ""
        log "############################################################"
        log "### PYTEST HANG DETECTED at iteration ${i}/${NUM_ITERATIONS}"
        log "### duration=${iter_duration}s (deadline=${PYTEST_TIMEOUT_SEC}s)"
        log "### tt-run pid=${ttrun_pid} LEFT RUNNING for inspection"
        log "### NOT issuing mpirun tt-smi -glx_reset; HW state preserved"
        log "###"
        log "### To inspect the hung process:"
        log "###   py-spy dump --pid ${ttrun_pid}"
        log "###   gdb -p ${ttrun_pid}"
        log "###   cat /proc/${ttrun_pid}/status"
        log "###   ls -l /proc/${ttrun_pid}/fd"
        log "### When done:"
        log "###   kill -9 ${ttrun_pid}"
        log "###   mpirun --host ${HOSTS} tt-smi -glx_reset"
        log "############################################################"
        exit 2
    fi

    # tt-run exited on its own -- collect status and classify.
    wait "${ttrun_pid}"
    ttrun_exit=$?

    case "${ttrun_exit}" in
        0)
            pass_count=$((pass_count + 1))
            verdict="PASS"
            ;;
        *)
            fail_count=$((fail_count + 1))
            verdict="FAIL (exit=${ttrun_exit})"
            ;;
    esac
    log "Iteration ${i} tt-run verdict: ${verdict}, duration=${iter_duration}s"

    # Reset runs unconditionally after a clean exit (pass or fail) --
    # stressing the reset path is the whole point of the loop.
    log "Running mpirun --host ${HOSTS} tt-smi -glx_reset (unconditional after non-hang exit)..."
    mpirun --host "${HOSTS}" tt-smi -glx_reset 2>&1 | tee -a "${LOG_FILE}"
    reset_exit=${PIPESTATUS[0]}
    if [[ ${reset_exit} -ne 0 ]]; then
        reset_fail_count=$((reset_fail_count + 1))
        log "mpirun tt-smi -glx_reset FAILED (exit=${reset_exit})"
    else
        log "mpirun tt-smi -glx_reset OK"
    fi
done
