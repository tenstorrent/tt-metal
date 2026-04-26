#!/bin/bash
# Autonomous validation loop for GLM-4.7-Flash bring-up
# Usage: ./experiments/run_loop.sh [max_experiments]
#
# Runs the test suite repeatedly after each refactoring step.
# Intended to be called by the agentic workflow, not directly by humans.

set -euo pipefail
cd "$(dirname "$0")/.."

MAX_EXPERIMENTS=${1:-10}
EXPERIMENT_NUM=0
FAILURES=0
MAX_FAILURES=5

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a experiments/loop.log
}

log "Starting validation loop (max=$MAX_EXPERIMENTS, max_failures=$MAX_FAILURES)"

# Source venv
if [ -f python_env/bin/activate ]; then
    source python_env/bin/activate
else
    log "ERROR: python_env not found. Run ./create_venv.sh first."
    exit 1
fi

while [ $EXPERIMENT_NUM -lt $MAX_EXPERIMENTS ] && [ $FAILURES -lt $MAX_FAILURES ]; do
    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
    RUN_NAME="glm4_auto_$(date +%Y%m%d_%H%M%S)"

    log "=== Experiment $EXPERIMENT_NUM/$MAX_EXPERIMENTS: $RUN_NAME ==="

    # Test 1: Layer 0 decode (fastest smoke test)
    log "Running layer 0 decode test..."
    if TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
        pytest models/demos/glm4_moe_lite/tests/test_tt_decoder_layer0_decode_update_cache_optional.py -v \
        2>&1 | tee -a experiments/loop.log; then
        log "PASS: layer 0 decode"
        echo "layer0_pass" > experiments/last_result.txt
    else
        log "FAIL: layer 0 decode"
        echo "layer0_fail" > experiments/last_result.txt
        FAILURES=$((FAILURES + 1))
        continue
    fi

    # Test 2: MoE layer (if layer 0 passed)
    log "Running MoE layer 1 test..."
    if TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
        pytest models/demos/glm4_moe_lite/tests/test_tt_moe_layer1_optional.py -v \
        2>&1 | tee -a experiments/loop.log; then
        log "PASS: MoE layer 1"
        echo "moe_pass" > experiments/last_result.txt
    else
        log "FAIL: MoE layer 1"
        echo "moe_fail" > experiments/last_result.txt
        FAILURES=$((FAILURES + 1))
        continue
    fi

    # Both passed
    log "All tests passed for experiment $EXPERIMENT_NUM"
    echo "all_pass" > experiments/last_result.txt
    FAILURES=0  # Reset consecutive failure counter on success

    # Log to ledger
    cat >> experiments/ledger.md << LEDGER_EOF

---

## Experiment (auto) $RUN_NAME

### Phase
modularize

### Validation
- layer 0 decode: pass
- MoE layer 1: pass

### Verdict
all tests passing

LEDGER_EOF

done

if [ $FAILURES -ge $MAX_FAILURES ]; then
    log "=== LOOP STOPPED: $MAX_FAILURES consecutive failures ==="
    exit 1
else
    log "=== LOOP COMPLETED: $EXPERIMENT_NUM experiments, all passing ==="
    exit 0
fi
