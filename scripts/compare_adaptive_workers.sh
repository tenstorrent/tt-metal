#!/bin/bash
#
# Compare model performance with and without adaptive AllGather workers
#
# Usage: ./scripts/compare_adaptive_workers.sh [--prompt-lengths "128,4k,8k"]
#

set -e

# Configuration
PROMPT_LENGTHS="128,4k,8k,16k,32k,64k,128k"
OUTPUT_DIR="perf_comparison_adaptive_workers"
LLAMA_DIR="../../fused_op/tt-metal/Meta-Llama-3.3-70B-Instruct/original"

# Test name mapping
declare -A TEST_NAMES=(
    ["128"]="batch-1"
    ["4k"]="long-4k-b1"
    ["8k"]="long-8k-b1"
    ["16k"]="long-16k-b1"
    ["32k"]="long-32k-b1"
    ["64k"]="long-64k-b1"
    ["128k"]="long-128k-b1"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_header() {
    echo -e "${BLUE}=============================================="
    echo -e "$1"
    echo -e "==============================================${NC}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt-lengths)
            PROMPT_LENGTHS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --prompt-lengths L    Comma-separated prompt lengths (default: all)"
            echo "  --output-dir DIR      Output directory"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup
cd /home/cust-team/llama70b/num_worker/tt-metal
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${OUTPUT_DIR}/${TIMESTAMP}"
mkdir -p "$RUN_DIR"

log_header "Adaptive Workers Performance Comparison"
log_info "Output directory: ${RUN_DIR}"
log_info "Prompt lengths: ${PROMPT_LENGTHS}"

# Save configuration
cat > "${RUN_DIR}/config.txt" << EOF
Timestamp: ${TIMESTAMP}
Prompt Lengths: ${PROMPT_LENGTHS}
LLAMA_DIR: ${LLAMA_DIR}
EOF

# Function to extract performance metrics from output
extract_metrics() {
    local log_file="$1"
    local ttft=$(grep "Average Time to First Token" "$log_file" 2>/dev/null | grep -oP '[\d.]+(?=ms)' | head -1)
    local speed=$(grep "Average speed" "$log_file" 2>/dev/null | grep -oP '[\d.]+(?= tok/s/user)' | head -1)
    local throughput=$(grep "Average speed" "$log_file" 2>/dev/null | grep -oP '[\d.]+(?= tok/s throughput)' | head -1)
    echo "${ttft:-N/A},${speed:-N/A},${throughput:-N/A}"
}

# Function to run a single test
run_test() {
    local prompt_len="$1"
    local config="$2"  # "baseline" or "adaptive"
    local output_file="$3"

    local test_name="${TEST_NAMES[$prompt_len]}"

    log_info "Running ${config} test for ${prompt_len} (test: ${test_name})"

    if [ "$config" == "adaptive" ]; then
        AG_ADAPTIVE_WORKERS=1 LLAMA_DIR="$LLAMA_DIR" \
            pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "$test_name" \
            2>&1 | tee "$output_file"
    else
        LLAMA_DIR="$LLAMA_DIR" \
            pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "$test_name" \
            2>&1 | tee "$output_file"
    fi

    return ${PIPESTATUS[0]}
}

# Parse prompt lengths
IFS=',' read -ra LENGTHS <<< "$PROMPT_LENGTHS"

# Results arrays
declare -A BASELINE_RESULTS
declare -A ADAPTIVE_RESULTS

# Run baseline tests
log_header "Running Baseline Tests (no adaptive workers)"
mkdir -p "${RUN_DIR}/baseline"

for len in "${LENGTHS[@]}"; do
    log_file="${RUN_DIR}/baseline/${len}.log"
    if run_test "$len" "baseline" "$log_file"; then
        BASELINE_RESULTS[$len]=$(extract_metrics "$log_file")
        log_info "Baseline ${len}: ${BASELINE_RESULTS[$len]}"
    else
        BASELINE_RESULTS[$len]="FAILED,FAILED,FAILED"
        log_info "Baseline ${len}: FAILED"
    fi
done

# Run adaptive tests
log_header "Running Adaptive Tests (AG_ADAPTIVE_WORKERS=1)"
mkdir -p "${RUN_DIR}/adaptive"

for len in "${LENGTHS[@]}"; do
    log_file="${RUN_DIR}/adaptive/${len}.log"
    if run_test "$len" "adaptive" "$log_file"; then
        ADAPTIVE_RESULTS[$len]=$(extract_metrics "$log_file")
        log_info "Adaptive ${len}: ${ADAPTIVE_RESULTS[$len]}"
    else
        ADAPTIVE_RESULTS[$len]="FAILED,FAILED,FAILED"
        log_info "Adaptive ${len}: FAILED"
    fi
done

# Generate comparison report
log_header "Generating Comparison Report"

REPORT_FILE="${RUN_DIR}/comparison_report.txt"
CSV_FILE="${RUN_DIR}/comparison_results.csv"

cat > "$REPORT_FILE" << EOF
================================================================================
Adaptive Workers Performance Comparison Report
================================================================================
Timestamp: ${TIMESTAMP}
Prompt Lengths: ${PROMPT_LENGTHS}

================================================================================
Results Summary
================================================================================

EOF

# CSV header
echo "Prompt_Length,Baseline_TTFT_ms,Adaptive_TTFT_ms,TTFT_Change_%,Baseline_TokPerSec,Adaptive_TokPerSec,Speed_Change_%" > "$CSV_FILE"

printf "%-10s | %-15s | %-15s | %-12s | %-15s | %-15s | %-12s\n" \
    "Length" "Base TTFT(ms)" "Adapt TTFT(ms)" "TTFT Change" "Base tok/s" "Adapt tok/s" "Speed Change" >> "$REPORT_FILE"
printf "%s\n" "-----------|-----------------|-----------------|--------------|-----------------|-----------------|-------------" >> "$REPORT_FILE"

for len in "${LENGTHS[@]}"; do
    IFS=',' read -r base_ttft base_speed base_throughput <<< "${BASELINE_RESULTS[$len]}"
    IFS=',' read -r adapt_ttft adapt_speed adapt_throughput <<< "${ADAPTIVE_RESULTS[$len]}"

    # Calculate changes
    if [[ "$base_ttft" != "N/A" && "$adapt_ttft" != "N/A" && "$base_ttft" != "FAILED" && "$adapt_ttft" != "FAILED" ]]; then
        ttft_change=$(python3 -c "print(f'{(($adapt_ttft - $base_ttft) / $base_ttft) * 100:.2f}')")
    else
        ttft_change="N/A"
    fi

    if [[ "$base_speed" != "N/A" && "$adapt_speed" != "N/A" && "$base_speed" != "FAILED" && "$adapt_speed" != "FAILED" ]]; then
        speed_change=$(python3 -c "print(f'{(($adapt_speed - $base_speed) / $base_speed) * 100:.2f}')")
    else
        speed_change="N/A"
    fi

    printf "%-10s | %-15s | %-15s | %-12s | %-15s | %-15s | %-12s\n" \
        "$len" "$base_ttft" "$adapt_ttft" "${ttft_change}%" "$base_speed" "$adapt_speed" "${speed_change}%" >> "$REPORT_FILE"

    echo "${len},${base_ttft},${adapt_ttft},${ttft_change},${base_speed},${adapt_speed},${speed_change}" >> "$CSV_FILE"
done

cat >> "$REPORT_FILE" << EOF

================================================================================
Notes
================================================================================
- TTFT = Time to First Token (lower is better)
- tok/s = tokens per second (higher is better)
- Negative TTFT change = improvement (faster)
- Positive Speed change = improvement (faster)

Baseline: Default AllGather worker selection
Adaptive: AG_ADAPTIVE_WORKERS=1 (more workers for large tensors)
================================================================================
EOF

cat "$REPORT_FILE"

log_header "Comparison Complete"
echo ""
echo "Results saved to:"
echo "  Report: ${REPORT_FILE}"
echo "  CSV: ${CSV_FILE}"
echo "  Logs: ${RUN_DIR}/baseline/ and ${RUN_DIR}/adaptive/"
