#!/bin/bash
# Stress test for non-deterministic hang in reduce_scatter_async_2x4 (deepseek_rs config)
#
# Reproduces the failure from CI job:
#   [WH-T3K][v1] Multi-process reduce scatter test model
#
# Error:
#   RuntimeError: TT_THROW @ tt_metal/impl/dispatch/system_memory_manager.cpp:627
#   TIMEOUT: device timeout, potential hang detected, the device is unrecoverable
#
# The Python script opens the device ONCE and loops the reduce scatter op
# internally for speed. If a hang/failure occurs, this wrapper resets the
# device with tt-smi -r and restarts from where it left off.
#
# Logs:
#   FULL_LOG    - every line of output from every iteration (verbose)
#   SUMMARY_LOG - only milestones, failures, and periodic summaries
#
# Usage: ./run_reduce_scatter_stress.sh

set -uo pipefail

TOTAL_ITERATIONS=50000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_LOG="$(pwd)/reduce_scatter_stress_full_${TIMESTAMP}.log"
SUMMARY_LOG="$(pwd)/reduce_scatter_stress_summary_${TIMESTAMP}.log"

# Track progress across restarts
CURRENT_ITERATION=1
TOTAL_PASS=0
TOTAL_FAIL=0

log_all() {
    echo "$1" | tee -a "$FULL_LOG" >> "$SUMMARY_LOG"
}

log_all "========================================================"
log_all "Reduce Scatter Async Stress Test (deepseek_rs / 2x4)"
log_all "Total iterations: $TOTAL_ITERATIONS"
log_all "Full log:    $FULL_LOG"
log_all "Summary log: $SUMMARY_LOG"
log_all "Started at:  $(date)"
log_all "========================================================"

while (( CURRENT_ITERATION <= TOTAL_ITERATIONS )); do
    log_all ""
    log_all ">>> Starting Python process at iteration $CURRENT_ITERATION [$(date)]"

    # Run the Python stress test -- it loops internally for speed.
    # stdbuf -oL -eL forces line-buffered stdout/stderr so output flows
    # through the tee pipe to the terminal immediately (no frozen screen).
    stdbuf -oL -eL \
        env \
        STRESS_TOTAL_ITERATIONS=$TOTAL_ITERATIONS \
        STRESS_START_ITERATION=$CURRENT_ITERATION \
        STRESS_SUMMARY_LOG="$SUMMARY_LOG" \
        python3 -u run_reduce_scatter_stress_single.py \
        2>&1 | stdbuf -oL tee -a "$FULL_LOG"

    EXIT_CODE=${PIPESTATUS[0]}

    if (( EXIT_CODE == 0 )); then
        # Completed all remaining iterations successfully
        PASSES=$((TOTAL_ITERATIONS - CURRENT_ITERATION + 1))
        TOTAL_PASS=$((TOTAL_PASS + PASSES))
        CURRENT_ITERATION=$((TOTAL_ITERATIONS + 1))
    else
        # Failed -- figure out how far we got from the summary log
        TOTAL_FAIL=$((TOTAL_FAIL + 1))

        # Parse the last successful iteration from stdout
        LAST_PASS=$(grep -oP 'PASS iteration \K[0-9]+' "$FULL_LOG" | tail -1)
        if [[ -n "$LAST_PASS" ]]; then
            PASSES_THIS_RUN=$((LAST_PASS - CURRENT_ITERATION + 1))
            TOTAL_PASS=$((TOTAL_PASS + PASSES_THIS_RUN))
            CURRENT_ITERATION=$((LAST_PASS + 2))  # skip the failed one, start fresh
        else
            # Failed on first iteration of this run
            CURRENT_ITERATION=$((CURRENT_ITERATION + 1))
        fi

        log_all "!!! Failure #$TOTAL_FAIL detected. Total passed so far: $TOTAL_PASS !!!"

        if (( CURRENT_ITERATION <= TOTAL_ITERATIONS )); then
            log_all "Resetting device with tt-smi -r before restart..."
            if ! tt-smi -r >> "$FULL_LOG" 2>&1; then
                log_all "WARNING: tt-smi -r failed. Retrying in 10s..."
                sleep 10
                tt-smi -r >> "$FULL_LOG" 2>&1 || true
            fi
            log_all "Restarting from iteration $CURRENT_ITERATION..."
        fi
    fi
done

log_all ""
log_all "========================================================"
log_all "FINAL RESULTS: $TOTAL_PASS passed, $TOTAL_FAIL failed out of $TOTAL_ITERATIONS"
log_all "Finished at: $(date)"
log_all "========================================================"
