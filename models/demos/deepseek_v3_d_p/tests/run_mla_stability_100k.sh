#!/usr/bin/env bash
# Stability loop for test_mla_loop (100k scaled_sl, balanced, iter25, random weights).
#
# Alternates device_params between `line_original` and `line_emb` on each
# outer iteration so both fabric variants get exercised back-to-back.
#
# Each outer iteration:
#   1. Run the selected pytest node under a wall-clock deadline.
#   2. If it finishes cleanly (pass or fail) -> run tt-smi -glx_reset and
#      continue to the next iteration (with the other fabric variant).
#   3. If it hangs past the deadline -> log the hang and exit the script
#      immediately, leaving the pytest process alive on live HW for manual
#      inspection. No triage, no kill, no reset.
#
# All stdout/stderr is mirrored to a log file.
#
# Usage:
#   ./run_mla_stability_100k.sh <num_iterations> [log_file]
#
# Must be run from the tt-metal repo root (pytest picks up conftest.py).

set -u

NUM_ITERATIONS="${1:?usage: $0 <num_iterations> [log_file]}"
LOG_FILE="${2:-stability_mla_100k_$(date +%Y%m%d_%H%M%S).log}"

# Generous default — test_mla_loop runs 25 forward passes at 100k seq_len.
# Override via env if needed: PYTEST_TIMEOUT_SEC=2400 ./run_mla_stability_100k.sh ...
PYTEST_TIMEOUT_SEC="${PYTEST_TIMEOUT_SEC:-1800}"
POLL_INTERVAL_SEC=5

# A failed reset cannot leak into the next iteration — running pytest on a
# half-reset mesh produces noise hangs that confuse triage. Retry the reset
# until it succeeds; bail the whole run if the cap is exhausted because at
# that point the HW state cannot be trusted.
RESET_MAX_ATTEMPTS="${RESET_MAX_ATTEMPTS:-20}"
RESET_RETRY_SLEEP_SEC="${RESET_RETRY_SLEEP_SEC:-5}"

export DEEPSEEK_V3_HF_MODEL="${DEEPSEEK_V3_HF_MODEL:-/mnt/models/deepseek-ai/DeepSeek-R1-0528}"

TEST_PATH="models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mla_loop"
# Filter components held fixed across iterations.
# Test IDs (from @pytest.mark.parametrize):
#   mesh_device:    8x4
#   use_pretrained: random
#   scale_down_sl:  scaled_sl
#   seq_len:        seq100k
#   is_balanced:    balanced
#   num_iterations: iter25
TEST_FILTER_FIXED='random and scaled_sl and seq100k and balanced and iter25 and 8x4'
# Variants for device_params, alternated each outer iteration.
FABRIC_VARIANTS=("line_original" "line_emb")

pass_count=0
fail_count=0
hang_count=0
reset_fail_count=0
hang_log=""  # accumulating "iter<i>(<variant>)" tokens, one per hang

log() { printf '%s\n' "$*" | tee -a "${LOG_FILE}"; }

# Retry tt-smi -glx_reset until it succeeds. Each failed attempt bumps
# reset_fail_count. If RESET_MAX_ATTEMPTS is exhausted, exits the whole
# script (status 3) — refusing to run further pytest iterations on bad HW.
reset_until_success() {
    local context="$1"
    local attempt=0
    while (( attempt < RESET_MAX_ATTEMPTS )); do
        attempt=$((attempt + 1))
        log "tt-smi -glx_reset attempt ${attempt}/${RESET_MAX_ATTEMPTS} (${context})..."
        tt-smi -glx_reset 2>&1 | tee -a "${LOG_FILE}"
        local exit_code=${PIPESTATUS[0]}
        if [[ ${exit_code} -eq 0 ]]; then
            if (( attempt > 1 )); then
                log "tt-smi -glx_reset OK after ${attempt} attempts (${context})"
            else
                log "tt-smi -glx_reset OK (${context})"
            fi
            return 0
        fi
        reset_fail_count=$((reset_fail_count + 1))
        log "tt-smi -glx_reset FAILED (exit=${exit_code}, attempt ${attempt}/${RESET_MAX_ATTEMPTS}, ${context})"
        if (( attempt < RESET_MAX_ATTEMPTS )); then
            sleep "${RESET_RETRY_SLEEP_SEC}"
        fi
    done
    log ""
    log "############################################################"
    log "### FATAL: tt-smi -glx_reset failed ${RESET_MAX_ATTEMPTS} times in a row"
    log "### (${context}). Aborting stability run — HW state cannot be trusted."
    log "############################################################"
    exit 3
}

on_exit() {
    log ""
    log "=================================================="
    log "Stability run ended at $(date -Is)"
    log "Pass:            ${pass_count}"
    log "Fail (non-hang): ${fail_count}"
    log "Hangs:           ${hang_count}${hang_log:+ at ${hang_log}}"
    log "Reset errors:    ${reset_fail_count}"
    log "Total runs:      $((pass_count + fail_count + hang_count))"
    log "Log file:        ${LOG_FILE}"
    log "=================================================="
}
trap on_exit EXIT INT TERM

log "=================================================="
log "MLA stability run started at $(date -Is)"
log "Outer iterations:            ${NUM_ITERATIONS}"
log "Pytest deadline (s):         ${PYTEST_TIMEOUT_SEC}"
log "Reset max attempts:          ${RESET_MAX_ATTEMPTS}"
log "Reset retry sleep (s):       ${RESET_RETRY_SLEEP_SEC}"
log "DEEPSEEK_V3_HF_MODEL=${DEEPSEEK_V3_HF_MODEL}"
log "Test path:                   ${TEST_PATH}"
log "Fixed filter:                ${TEST_FILTER_FIXED}"
log "Fabric variants (alternated):${FABRIC_VARIANTS[*]}"
log "=================================================="

for (( i=1; i<=NUM_ITERATIONS; i++ )); do
    variant="${FABRIC_VARIANTS[$(( (i - 1) % ${#FABRIC_VARIANTS[@]} ))]}"
    test_filter="${TEST_FILTER_FIXED} and ${variant}"

    log ""
    log "------- Iteration ${i}/${NUM_ITERATIONS} (${variant}) @ $(date -Is) -------"
    log "Filter: ${test_filter}"

    iter_start=$(date +%s)
    deadline=$(( iter_start + PYTEST_TIMEOUT_SEC ))

    # Launch pytest in background, tee'ing output into the log.
    # Process substitution (> >(tee ...)) lets us capture pytest's PID via $!
    # rather than tee's, so we can poll/leave-alive the right process.
    pytest -xvs "${TEST_PATH}" -k "${test_filter}" \
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
        hang_log+="${hang_log:+, }iter${i}(${variant})"

        log ""
        log "############################################################"
        log "### PYTEST HANG DETECTED at iteration ${i}/${NUM_ITERATIONS} (${variant})"
        log "### duration=${iter_duration}s (deadline=${PYTEST_TIMEOUT_SEC}s)"
        log "### pytest pid=${pytest_pid} (LEFT ALIVE for manual inspection)"
        log "### No triage, no kill, no reset. Exiting stability run."
        log "############################################################"

        # Make sure the backgrounded pytest survives this script's exit so
        # the user can attach/inspect the live HW state.
        disown "${pytest_pid}" 2>/dev/null || true
        exit 2
    fi

    # Pytest exited on its own - collect status and classify.
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
    log "Iteration ${i} (${variant}) verdict: ${verdict}, duration=${iter_duration}s"

    # Reset runs unconditionally after a clean pytest exit (pass or fail) -
    # stressing the reset path is the whole point of the loop.
    reset_until_success "post-${verdict} at iter ${i} (${variant})"
done
