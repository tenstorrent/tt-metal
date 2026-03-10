#!/bin/bash
# Autonomous experiment loop for AG+MM 7x8 4-links optimization
# Usage: ./experiments/run_loop.sh [max_experiments]

set -e
cd /home/cust-team/teja/tt-metal

MAX_EXPERIMENTS=${1:-10}
EXPERIMENT_NUM=0
SUCCESS=false

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a experiments/loop.log
}

log "Starting autonomous experiment loop (max=$MAX_EXPERIMENTS)"

# Source venv
source python_env/bin/activate

while [ $EXPERIMENT_NUM -lt $MAX_EXPERIMENTS ] && [ "$SUCCESS" = "false" ]; do
    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
    RUN_NAME="ag_mm_auto_$(date +%Y%m%d_%H%M%S)"

    log "=== Experiment $EXPERIMENT_NUM/$MAX_EXPERIMENTS: $RUN_NAME ==="

    # Reset devices (safe to do every time)
    log "Resetting devices..."
    tt-smi -glx_reset || true
    sleep 60

    # Rebuild (in case code changed)
    log "Rebuilding tt-metal..."
    if ! ./build_metal.sh 2>&1 | tee -a experiments/loop.log; then
        log "BUILD FAILED - stopping loop"
        echo "build_failure" > experiments/last_result.txt
        break
    fi

    # Run profiler
    log "Running profiler..."
    SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=1 \
        ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name "$RUN_NAME" \
        2>&1 | tee experiments/profiler_output.txt || true

    # Check for errors
    if grep -qE 'FATAL|TT_THROW|Aborted' experiments/profiler_output.txt; then
        log "RUNTIME ERROR detected"
        echo "runtime_failure" > experiments/last_result.txt
        continue  # Try next experiment
    fi

    # Check for hang (if output file doesn't exist or is too old)
    PREFILL_CSV="profiler_sweep_results/$RUN_NAME/8k/prefill.csv"
    if [ ! -f "$PREFILL_CSV" ]; then
        log "HANG detected (no prefill.csv)"
        echo "runtime_failure_hang" > experiments/last_result.txt
        continue
    fi

    # Check activation
    if grep -q 'AG+MM.*7x8' experiments/profiler_output.txt; then
        log "4-link path ACTIVE"
        ACTIVE=true
    else
        log "4-link path NOT ACTIVE"
        ACTIVE=false
        echo "not_active" > experiments/last_result.txt
        continue
    fi

    # Parse duration (assuming fused op is in prefill.csv)
    # This is a placeholder - adjust based on actual CSV format
    DURATION=$(grep -i 'all_gather.*matmul\|fused' "$PREFILL_CSV" | head -1 | cut -d',' -f2 || echo "0")
    BASELINE=1851.18  # non-fused baseline in µs

    log "Fused op duration: $DURATION µs (baseline: $BASELINE µs)"

    # Compare to baseline
    if [ "$ACTIVE" = "true" ]; then
        # Use bc for float comparison
        if echo "$DURATION < $BASELINE" | bc -l | grep -q 1; then
            log "SUCCESS! Duration $DURATION < baseline $BASELINE"
            SUCCESS=true
            echo "success" > experiments/last_result.txt
        else
            log "Active but slower: $DURATION >= $BASELINE"
            echo "active_but_slower" > experiments/last_result.txt
        fi
    fi

    # Log to ledger
    cat >> experiments/ledger.md << EOF

---

## Experiment (auto) $RUN_NAME

- **Duration:** $DURATION µs
- **Baseline:** $BASELINE µs
- **Active:** $ACTIVE
- **Result:** $(cat experiments/last_result.txt)

EOF

done

if [ "$SUCCESS" = "true" ]; then
    log "=== LOOP COMPLETED: SUCCESS ==="
    exit 0
else
    log "=== LOOP COMPLETED: No success after $EXPERIMENT_NUM experiments ==="
    exit 1
fi
