#!/bin/bash
# Stress test for the full vLLM server + benchmark cycle.
#
# Reproduces the CI job:
#   [WH-T3K][v1] Multi-process reduce scatter test model
#
# Each iteration:
#   1. Start vLLM server (MPI, 8 DP workers, dummy reduce scatter model)
#   2. Wait for server health
#   3. Run benchmark (32 prompts, 100 in / 100 out tokens)
#   4. Kill server
#   5. tt-smi -r
#
# The hang was:
#   RuntimeError: TT_THROW @ tt_metal/impl/dispatch/system_memory_manager.cpp:627
#   TIMEOUT: device timeout, potential hang detected, the device is unrecoverable
#
# Logs:
#   FULL_LOG    - all server/benchmark/reset output
#   SUMMARY_LOG - iteration results, failures, periodic summaries
#
# Usage: ./run_vllm_server_stress.sh

set -uo pipefail

TOTAL_ITERATIONS=${1:-50000}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_LOG="$(pwd)/vllm_stress_full_${TIMESTAMP}.log"
SUMMARY_LOG="$(pwd)/vllm_stress_summary_${TIMESTAMP}.log"
SERVER_LOG="$(pwd)/vllm_stress_server_${TIMESTAMP}.log"

# Match CI environment
export TT_METAL_HOME=/tt-metal
export PYTHONPATH="/tt-metal:/tt-metal/vllm"
export LD_LIBRARY_PATH="/tt-metal/build/lib"
export LOGURU_LEVEL=INFO
export VLLM_TARGET_DEVICE="tt"
export VLLM_USE_V1=1
export VLLM_RPC_TIMEOUT=300000
export TT_METAL_OPERATION_TIMEOUT_SECONDS=5.0  # CI default via setup-job action
export MESH_DEVICE="(2, 4)"
export TT_METAL_WATCHER=5  # Poll device state every 5s; watcher.log shows stuck waypoints on hang
export TT_METAL_WATCHER_NOINLINE=1              # Reduce kernel binary size
export TT_METAL_WATCHER_DISABLE_ASSERT=1        # We only need waypoints, disable everything else
export TT_METAL_WATCHER_DISABLE_PAUSE=1         # to fit within idle_erisc code size limit (0x5500)
export TT_METAL_WATCHER_DISABLE_RING_BUFFER=1
export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1
export TT_METAL_WATCHER_DISABLE_STACK_USAGE=1


# Server config (from CI matrix)
MODEL="models/vllm_test_utils/t3000_multiproc_test"
# CI uses meta-llama/Llama-3.1-8B-Instruct with a pre-cached HF hub (HF_HUB_OFFLINE=1).
# We don't have that cache, so use the identical ungated mirror:
TOKENIZER="NousResearch/Meta-Llama-3.1-8B-Instruct"
OVERRIDE_TT_CONFIG='{"rank_binding": "tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml", "mpi_args": "--allow-run-as-root --tag-output", "config_pkl_dir": "vllm", "register_test_models": true}'

SERVER_TIMEOUT_SECONDS=240  # 4 min, matches CI
BENCHMARK_TIMEOUT_SECONDS=240  # 4 min, matches CI

PASS_COUNT=0
FAIL_COUNT=0

log_all() {
    echo "$1" | tee -a "$FULL_LOG" >> "$SUMMARY_LOG"
}

log_full() {
    echo "$1" | tee -a "$FULL_LOG"
}

log_all "========================================================"
log_all "vLLM Server Stress Test (reduce scatter test model)"
log_all "Total iterations: $TOTAL_ITERATIONS"
log_all "Full log:    $FULL_LOG"
log_all "Summary log: $SUMMARY_LOG"
log_all "Server log:  $SERVER_LOG (overwritten each iteration)"
log_all "Started at:  $(date)"
log_all "========================================================"

for i in $(seq 1 $TOTAL_ITERATIONS); do
    ITER_START=$(date +%s)
    log_all ""
    log_all "--- Iteration $i / $TOTAL_ITERATIONS (passed: $PASS_COUNT, failed: $FAIL_COUNT) --- [$(date)]"

    FAILED=false
    FAIL_REASON=""

    # --- 1. Start vLLM server ---
    log_full "Starting vLLM server..."
    > "$SERVER_LOG"  # truncate server log

    python3 vllm/examples/server_example_tt.py \
        --model "$MODEL" \
        --data_parallel_size 8 \
        --tokenizer "$TOKENIZER" \
        --override-tt-config "$OVERRIDE_TT_CONFIG" \
        --num_scheduler_steps 1 \
        > "$SERVER_LOG" 2>&1 &

    SERVER_PID=$!
    log_full "Server started with PID $SERVER_PID"

    # --- 2. Wait for server health ---
    log_full "Waiting for server to be ready (timeout: ${SERVER_TIMEOUT_SECONDS}s)..."
    elapsed=0
    interval=10
    SERVER_READY=false

    while [ $elapsed -lt $SERVER_TIMEOUT_SECONDS ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            FAIL_REASON="Server process died during startup"
            FAILED=true
            break
        fi
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            log_full "Server is up! [${elapsed}s]"
            SERVER_READY=true
            break
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done

    if [ "$FAILED" = false ] && [ "$SERVER_READY" = false ]; then
        FAIL_REASON="Server did not become ready within ${SERVER_TIMEOUT_SECONDS}s"
        FAILED=true
    fi

    # --- 3. Run benchmark (matches CI: vllm bench serve) ---
    if [ "$FAILED" = false ]; then
        log_full "Running benchmark..."

        BENCH_OUTPUT=""
        BENCH_EXIT=0
        BENCH_OUTPUT=$(timeout $BENCHMARK_TIMEOUT_SECONDS \
            vllm bench serve \
                --backend vllm \
                --model "$MODEL" \
                --dataset-name random \
                --random-input-len 100 \
                --random-output-len 100 \
                --num-prompts 32 \
                --ignore-eos \
                --tokenizer "$TOKENIZER" \
                --percentile-metrics ttft,tpot,itl,e2el \
            2>&1) || BENCH_EXIT=$?

        echo "$BENCH_OUTPUT" >> "$FULL_LOG"

        if [ $BENCH_EXIT -eq 124 ]; then
            FAIL_REASON="Benchmark timed out after ${BENCHMARK_TIMEOUT_SECONDS}s"
            FAILED=true
        elif echo "$BENCH_OUTPUT" | grep -q "All requests failed"; then
            FAIL_REASON="Benchmark: all requests failed"
            FAILED=true
        elif echo "$BENCH_OUTPUT" | grep -q "TIMEOUT.*device timeout"; then
            FAIL_REASON="Benchmark: device timeout / hang detected"
            FAILED=true
        fi
    fi

    # --- 4. Kill server (with force-kill fallback) ---
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        # Wait up to 15s for graceful shutdown
        for _w in $(seq 1 15); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        # Force kill if still alive
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log_full "Server $SERVER_PID did not exit gracefully, sending SIGKILL..."
            kill -9 "$SERVER_PID" 2>/dev/null
            sleep 1
        fi
        wait "$SERVER_PID" 2>/dev/null || true
        log_full "Server process $SERVER_PID terminated."
    else
        if [ "$FAILED" = false ]; then
            FAIL_REASON="Server process died unexpectedly"
            FAILED=true
        fi
        log_full "Server process $SERVER_PID already dead."
    fi
    # Kill any stray MPI/worker processes from this iteration
    pkill -9 -f "tt_core_launcher.*tmp_vllm_tt" 2>/dev/null || true
    pkill -9 -f "prterun.*tmp_vllm_tt" 2>/dev/null || true
    sleep 1

    # Save server log and watcher log on failure
    if [ "$FAILED" = true ]; then
        FAIL_SERVER_LOG="$(pwd)/vllm_stress_server_FAIL_iter${i}_${TIMESTAMP}.log"
        cp "$SERVER_LOG" "$FAIL_SERVER_LOG"
        # Copy watcher log if it exists
        WATCHER_LOG=$(find /tt-metal /tmp -path "*/generated/watcher/watcher.log" -newer "$SERVER_LOG" 2>/dev/null | head -1)
        if [ -n "$WATCHER_LOG" ]; then
            FAIL_WATCHER_LOG="$(pwd)/vllm_stress_watcher_FAIL_iter${i}_${TIMESTAMP}.log"
            cp "$WATCHER_LOG" "$FAIL_WATCHER_LOG"
            log_all "Watcher log saved to: $FAIL_WATCHER_LOG"
        fi
        log_all "!!! FAILURE on iteration $i: $FAIL_REASON !!!"
        log_all "Server log saved to: $FAIL_SERVER_LOG"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        PASS_COUNT=$((PASS_COUNT + 1))
        ITER_END=$(date +%s)
        ITER_DURATION=$((ITER_END - ITER_START))
        log_full "PASS iteration $i (${ITER_DURATION}s)"
    fi

    # --- 5. Reset device ---
    log_full "Resetting device with tt-smi -r..."
    if ! tt-smi -r >> "$FULL_LOG" 2>&1; then
        log_all "WARNING: tt-smi -r failed on iteration $i. Retrying in 10s..."
        sleep 10
        tt-smi -r >> "$FULL_LOG" 2>&1 || true
    fi

    # Periodic summary
    if (( i % 10 == 0 )); then
        log_all "=== SUMMARY at iteration $i: $PASS_COUNT passed, $FAIL_COUNT failed ==="
    fi
done

log_all ""
log_all "========================================================"
log_all "FINAL RESULTS: $PASS_COUNT passed, $FAIL_COUNT failed out of $TOTAL_ITERATIONS"
log_all "Finished at: $(date)"
log_all "========================================================"
