#!/usr/bin/env bash
# Stability loop for test_ring_attention_ag_isolated (100k seq, iter25, 2 links).
#
# Drives the standalone ring_attention_all_gather_async op (no SDPA compute) at
# the same shapes/sharding/mesh/fabric/topology that MLA's ring SDPA feeds it.
# Purpose: determine whether the SDPA-shaped hangs reproduce when the AG path
# runs in isolation. If yes, the bug lives in the AG kernels / fabric they
# drive; if no, the bug lives in SDPA's compute or in the AG↔compute
# interaction inside ring_joint_scaled_dot_product_attention.
#
# Alternates device_params between `line_original` and `line_emb` on each
# outer iteration so both fabric variants get exercised back-to-back.
#
# Each outer iteration:
#   1. Run the selected pytest node under a wall-clock deadline.
#   2. If it finishes cleanly (pass or fail) -> run tt-smi -glx_reset and
#      continue to the next iteration (with the other fabric variant).
#   3. If it hangs past the deadline -> run ./tools/tt-triage.py -vv against
#      the live HW (output captured to a per-hang log), then kill -9 the
#      hung pytest, run tt-smi -glx_reset, and continue to the next iteration.
#
# All stdout/stderr is mirrored to a log file. Per-hang triage output goes
# to a sibling log named like <log>_triage_iter<i>_<variant>.log.
#
# Usage:
#   ./run_ag_stability_100k.sh <num_iterations> [log_file]
#
# Must be run from the tt-metal repo root (pytest picks up conftest.py and
# tt-triage is invoked as ./tools/tt-triage.py).

set -u

NUM_ITERATIONS="${1:?usage: $0 <num_iterations> [log_file]}"
LOG_FILE="${2:-stability_ag_100k_$(date +%Y%m%d_%H%M%S).log}"

# AG-only is materially faster than full SDPA but 100k-seq AG is still
# substantial fabric traffic. Override via env if needed:
#   PYTEST_TIMEOUT_SEC=1800 ./run_ag_stability_100k.sh ...
PYTEST_TIMEOUT_SEC="${PYTEST_TIMEOUT_SEC:-1200}"
POLL_INTERVAL_SEC=5

# Cap triage so a hung tt-triage.py can't stall the outer loop forever.
TRIAGE_TIMEOUT_SEC="${TRIAGE_TIMEOUT_SEC:-600}"
TRIAGE_CMD=("./tools/tt-triage.py" "-vv")

# A failed reset cannot leak into the next iteration — running pytest on a
# half-reset mesh produces noise hangs that confuse triage. Retry the reset
# until it succeeds; bail the whole run if the cap is exhausted because at
# that point the HW state cannot be trusted.
RESET_MAX_ATTEMPTS="${RESET_MAX_ATTEMPTS:-20}"
RESET_RETRY_SLEEP_SEC="${RESET_RETRY_SLEEP_SEC:-5}"

TEST_PATH="models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_ring_attention_ag_isolated.py::test_ring_attention_ag_isolated"
# Filter components held fixed across iterations. Test IDs from
# @pytest.mark.parametrize on test_ring_attention_ag_isolated:
#   kv_dtype:   kv_bf8
#   seq_len:    seq100k
#   n_iters:    iter25
#   num_links:  2link
#   mesh:       8x4
#   rp/up axes: rpxup
TEST_FILTER_FIXED='kv_bf8 and seq100k and iter25 and 2link and 8x4 and rpxup'
# Variants for device_params, alternated each outer iteration.
FABRIC_VARIANTS=("line_original" "line_emb")

pass_count=0
fail_count=0
hang_count=0
reset_fail_count=0
triage_fail_count=0
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
    log "Triage errors:   ${triage_fail_count}"
    log "Reset errors:    ${reset_fail_count}"
    log "Total runs:      $((pass_count + fail_count + hang_count))"
    log "Log file:        ${LOG_FILE}"
    log "=================================================="
}
trap on_exit EXIT INT TERM

log "=================================================="
log "AG-isolated stability run started at $(date -Is)"
log "Outer iterations:            ${NUM_ITERATIONS}"
log "Pytest deadline (s):         ${PYTEST_TIMEOUT_SEC}"
log "Triage deadline (s):         ${TRIAGE_TIMEOUT_SEC}"
log "Triage command:              ${TRIAGE_CMD[*]}"
log "Reset max attempts:          ${RESET_MAX_ATTEMPTS}"
log "Reset retry sleep (s):       ${RESET_RETRY_SLEEP_SEC}"
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

        # Per-hang triage log lives next to the main log.
        log_dir="$(dirname "${LOG_FILE}")"
        log_base="$(basename "${LOG_FILE}" .log)"
        triage_log="${log_dir}/${log_base}_triage_iter${i}_${variant}.log"

        log ""
        log "############################################################"
        log "### PYTEST HANG DETECTED at iteration ${i}/${NUM_ITERATIONS} (${variant})"
        log "### duration=${iter_duration}s (deadline=${PYTEST_TIMEOUT_SEC}s)"
        log "### pytest pid=${pytest_pid} (will be killed after triage)"
        log "### Running triage: ${TRIAGE_CMD[*]}"
        log "### Triage log:    ${triage_log}"
        log "############################################################"

        # Run triage against the still-live HW. timeout(1) caps it so a stuck
        # triage can't stall the outer loop. Output goes to its own log so
        # the main stability log stays readable.
        triage_start=$(date +%s)
        timeout --kill-after=30s "${TRIAGE_TIMEOUT_SEC}" "${TRIAGE_CMD[@]}" \
            > "${triage_log}" 2>&1
        triage_exit=$?
        triage_duration=$(( $(date +%s) - triage_start ))

        case "${triage_exit}" in
            0)
                log "Triage finished OK (${triage_duration}s) -> ${triage_log}"
                ;;
            124|137)
                triage_fail_count=$((triage_fail_count + 1))
                log "Triage TIMED OUT after ${triage_duration}s (exit=${triage_exit}) -> ${triage_log}"
                ;;
            *)
                triage_fail_count=$((triage_fail_count + 1))
                log "Triage FAILED (exit=${triage_exit}, ${triage_duration}s) -> ${triage_log}"
                ;;
        esac

        # Kill the hung pytest tree (children first, then the leader).
        log "Killing hung pytest pid=${pytest_pid} and its descendants..."
        pkill -KILL -P "${pytest_pid}" 2>/dev/null || true
        kill -9 "${pytest_pid}" 2>/dev/null || true
        # Give the kernel a moment to reap.
        for _ in 1 2 3 4 5; do
            kill -0 "${pytest_pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${pytest_pid}" 2>/dev/null; then
            log "WARNING: pytest pid=${pytest_pid} still alive after kill -9"
        fi
        wait "${pytest_pid}" 2>/dev/null || true

        reset_until_success "after hang at iter ${i} (${variant})"

        # Continue to the next iteration (do not run the post-success
        # reset path below).
        continue
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
