#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Convenience wrapper for generating golden KV cache with memory monitoring
# and proper logging setup.

set -e

# Default paths (override with env vars)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PROMPT_JSON="${PROMPT_JSON:-$TT_METAL_ROOT/prompt.json}"
OUT_DIR="${OUT_DIR:-/tmp/minimax_m3_golden}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"

# Generation parameters
MAX_TOKENS="${MAX_TOKENS:-65536}"
DTYPE="${DTYPE:-bfloat16}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Generate golden KV cache for MiniMax M3 with memory monitoring.

OPTIONS:
    -h, --help              Show this help message
    -p, --prompt-json FILE  Prompt JSON file (default: \$TT_METAL_ROOT/prompt.json)
    -o, --out DIR           Output directory (default: /tmp/minimax_m3_golden)
    -m, --max-tokens N      Max tokens to process (default: 65536)
    -t, --test              Quick test mode (1024 tokens, /tmp output)
    --no-monitor            Skip memory monitoring

ENVIRONMENT:
    HF_MODEL                Path to MiniMax M3 model (required)
    PROMPT_JSON             Override default prompt path
    OUT_DIR                 Override default output path
    MAX_TOKENS              Override default max tokens

EXAMPLES:
    # Full generation with prompt.json
    export HF_MODEL=/data/models/MiniMax-M3
    $0 --prompt-json prompt.json --out /data/golden

    # Quick test
    export HF_MODEL=/data/models/MiniMax-M3
    $0 --test

    # Custom token limit
    export HF_MODEL=/data/models/MiniMax-M3
    MAX_TOKENS=32768 $0
EOF
    exit 0
}

log_info() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $*"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $*" >&2
}

check_memory() {
    local available_gb=$(free -g | awk '/^Mem:/{print $7}')
    log_info "Available memory: ${available_gb}GB"

    if [ "$available_gb" -lt 100 ]; then
        log_error "Insufficient memory! Need ~450GB available, have ${available_gb}GB"
        log_error "Close other applications or add swap space"
        exit 1
    fi

    if [ "$available_gb" -lt 200 ]; then
        log_warn "Low memory (${available_gb}GB). Consider:"
        log_warn "  - Reducing --max-tokens"
        log_warn "  - Enabling swap space"
        log_warn "  - Closing other applications"
    fi
}

monitor_memory() {
    local log_file="$1"
    log_info "Starting memory monitor (logging to $log_file)"

    while true; do
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        mem_used=$(free -g | awk '/^Mem:/{print $3}')
        mem_total=$(free -g | awk '/^Mem:/{print $2}')
        mem_pct=$(awk "BEGIN {printf \"%.1f\", ($mem_used/$mem_total)*100}")

        echo "$timestamp | Used: ${mem_used}GB / ${mem_total}GB (${mem_pct}%)" >> "$log_file"

        sleep 60  # Log every minute
    done
}

# Parse arguments
MONITOR_MEMORY=true
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        -p|--prompt-json)
            PROMPT_JSON="$2"
            shift 2
            ;;
        -o|--out)
            OUT_DIR="$2"
            shift 2
            ;;
        -m|--max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -t|--test)
            TEST_MODE=true
            shift
            ;;
        --no-monitor)
            MONITOR_MEMORY=false
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Test mode overrides
if [ "$TEST_MODE" = true ]; then
    log_info "Test mode enabled"
    MAX_TOKENS=1024
    OUT_DIR="/tmp/minimax_m3_test_$(date +%s)"
fi

# Validate requirements
if [ -z "$HF_MODEL" ]; then
    log_error "HF_MODEL environment variable not set"
    log_error "Example: export HF_MODEL=/path/to/MiniMax-M3"
    exit 1
fi

if [ ! -d "$HF_MODEL" ]; then
    log_error "Model directory not found: $HF_MODEL"
    exit 1
fi

if [ ! -f "$PROMPT_JSON" ]; then
    log_error "Prompt file not found: $PROMPT_JSON"
    log_error "Use --prompt-json to specify a different file"
    exit 1
fi

# Setup
mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
MAIN_LOG="$LOG_DIR/generate_${TIMESTAMP}.log"
MEM_LOG="$LOG_DIR/memory_${TIMESTAMP}.log"

log_info "=========================================="
log_info "MiniMax M3 Golden KV Cache Generation"
log_info "=========================================="
log_info "Model:       $HF_MODEL"
log_info "Prompt:      $PROMPT_JSON"
log_info "Output:      $OUT_DIR"
log_info "Max tokens:  $MAX_TOKENS"
log_info "Dtype:       $DTYPE"
log_info "Main log:    $MAIN_LOG"
log_info "Memory log:  $MEM_LOG"
log_info "=========================================="

# Check memory
check_memory

# Start memory monitor in background
if [ "$MONITOR_MEMORY" = true ]; then
    monitor_memory "$MEM_LOG" &
    MONITOR_PID=$!
    trap "kill $MONITOR_PID 2>/dev/null || true" EXIT
fi

# Run generation
log_info "Starting golden KV cache generation..."
log_info "This will take a LONG time (10-60+ minutes depending on prompt size and CPU)"
log_info "Progress updates show elapsed time - CPU inference has no ETA"
log_info "Follow progress: tail -f $MAIN_LOG"

cd "$TT_METAL_ROOT"

python3 models/demos/minimax_m3/scripts/generate_golden_kv_cache.py \
    --prompt-json "$PROMPT_JSON" \
    --out "$OUT_DIR" \
    --max-tokens "$MAX_TOKENS" \
    --dtype "$DTYPE" \
    --model-path "$HF_MODEL" \
    2>&1 | tee "$MAIN_LOG"

GENERATION_EXIT=$?

# Stop memory monitor
if [ "$MONITOR_MEMORY" = true ]; then
    kill $MONITOR_PID 2>/dev/null || true
fi

# Verify output
if [ $GENERATION_EXIT -eq 0 ]; then
    log_info "Generation completed successfully"
    log_info "Verifying output..."

    python3 models/demos/minimax_m3/scripts/verify_golden_kv.py "$OUT_DIR"
    VERIFY_EXIT=$?

    if [ $VERIFY_EXIT -eq 0 ]; then
        log_info "=========================================="
        log_info "✅ SUCCESS!"
        log_info "=========================================="
        log_info "Golden KV cache ready at: $OUT_DIR"
        log_info ""
        log_info "Usage:"
        log_info "  export PREFILL_TRACE_DIR=$OUT_DIR"
        log_info "  export PREFILL_STANDALONE_PCC=1"
        log_info "  python3 models/demos/deepseek_v3_d_p/tt/runners/prefill_runner.py"
        log_info "=========================================="
        exit 0
    else
        log_error "Verification failed - see output above"
        exit 1
    fi
else
    log_error "Generation failed - check $MAIN_LOG for details"
    exit 1
fi
