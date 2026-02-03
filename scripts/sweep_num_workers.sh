#!/bin/bash
#
# Sweep num_workers_per_link for AllGather operations across multiple prompt lengths
#
# Usage:
#   ./scripts/sweep_num_workers.sh [--workers "1 2 4 6"] [--prompt-lengths "128,4k"]
#

set -e

# Default configuration
WORKER_VALUES=(5 6 7)
SELECTED_PROMPTS="128,4k,8k,16k,32k,64k,128k"
OUTPUT_BASE_DIR="profiler_sweep_results"
BASELINE_DIR="/home/cust-team/llama70b/fused_op/tt-metal/profiler_sweep_results/baseline"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_header() {
    echo -e "${BLUE}=============================================="
    echo -e "$1"
    echo -e "==============================================${NC}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workers)
            IFS=' ' read -ra WORKER_VALUES <<< "$2"
            shift 2
            ;;
        --prompt-lengths)
            SELECTED_PROMPTS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --baseline-dir)
            BASELINE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --workers \"N1 N2 ...\"    Space-separated worker counts to sweep (default: \"1 2 4 6 7\")"
            echo "  --prompt-lengths L        Comma-separated prompt lengths (default: all)"
            echo "  --output-dir DIR          Output directory (default: profiler_sweep_results)"
            echo "  --baseline-dir DIR        Baseline directory for comparison"
            echo ""
            echo "Example:"
            echo "  $0 --workers \"1 2 4\" --prompt-lengths 128,4k"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup environment
cd /home/cust-team/llama70b/num_worker/tt-metal
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)
source python_env/bin/activate

SWEEP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SWEEP_DIR="${OUTPUT_BASE_DIR}/num_workers_sweep_${SWEEP_TIMESTAMP}"
mkdir -p "$SWEEP_DIR"

log_header "NumWorkers Sweep: ${SWEEP_TIMESTAMP}"
log_info "Worker values: ${WORKER_VALUES[*]}"
log_info "Prompt lengths: ${SELECTED_PROMPTS}"
log_info "Output directory: ${SWEEP_DIR}"

# Save sweep configuration
cat > "${SWEEP_DIR}/sweep_config.txt" << EOF
Sweep Timestamp: ${SWEEP_TIMESTAMP}
Date: $(date)
Worker Values: ${WORKER_VALUES[*]}
Prompt Lengths: ${SELECTED_PROMPTS}
Baseline Dir: ${BASELINE_DIR}
EOF

# Run sweep for each worker count
for num_workers in "${WORKER_VALUES[@]}"; do
    log_header "Running with num_workers_per_link=${num_workers}"

    RUN_NAME="workers_${num_workers}"
    RUN_DIR="${SWEEP_DIR}/${RUN_NAME}"

    export AG_NUM_WORKERS_PER_LINK="${num_workers}"

    ./scripts/run_profiler_sweep.sh \
        --output-dir "${SWEEP_DIR}" \
        --run-name "${RUN_NAME}" \
        --prompt-lengths "${SELECTED_PROMPTS}" \
        || log_warn "Sweep for workers=${num_workers} had some failures"

    echo ""
done

# Generate comparison summary
log_header "Generating Comparison Summary"

# Create summary file
SUMMARY_FILE="${SWEEP_DIR}/sweep_summary.txt"
cat > "$SUMMARY_FILE" << EOF
=============================================================================
NumWorkers Sweep Summary
=============================================================================
Sweep Timestamp: ${SWEEP_TIMESTAMP}
Worker Values Tested: ${WORKER_VALUES[*]}
Prompt Lengths: ${SELECTED_PROMPTS}

=============================================================================
Results by Worker Count
=============================================================================

EOF

# Extract AllGatherAsync performance for each configuration
IFS=',' read -ra PROMPT_ARRAY <<< "$SELECTED_PROMPTS"

for prompt_len in "${PROMPT_ARRAY[@]}"; do
    echo "" >> "$SUMMARY_FILE"
    echo "Prompt Length: ${prompt_len}" >> "$SUMMARY_FILE"
    echo "-----------------------------------" >> "$SUMMARY_FILE"
    printf "%-12s %15s %15s %15s\n" "Workers" "AG_Device_us" "AG_Total_us" "AG_Count" >> "$SUMMARY_FILE"

    for num_workers in "${WORKER_VALUES[@]}"; do
        RUN_DIR="${SWEEP_DIR}/workers_${num_workers}/${prompt_len}"
        if [ -f "${RUN_DIR}/prefill.csv" ]; then
            # Extract AllGatherAsync stats from prefill.csv
            AG_STATS=$(python3 << EOF
import pandas as pd
import sys

try:
    df = pd.read_csv("${RUN_DIR}/prefill.csv")
    # Filter for AllGatherAsync operations
    ag_df = df[df['OP CODE'].str.contains('AllGatherAsync', case=False, na=False)]
    if len(ag_df) > 0:
        device_time = ag_df['DEVICE TIME'].sum()
        total_time = ag_df['OP TO OP LATENCY'].sum() if 'OP TO OP LATENCY' in ag_df.columns else device_time
        count = len(ag_df)
        print(f"{device_time:.1f},{total_time:.1f},{count}")
    else:
        print("N/A,N/A,0")
except Exception as e:
    print(f"ERROR,ERROR,0")
EOF
)
            IFS=',' read -r device_time total_time count <<< "$AG_STATS"
            printf "%-12s %15s %15s %15s\n" "${num_workers}" "${device_time}" "${total_time}" "${count}" >> "$SUMMARY_FILE"
        else
            printf "%-12s %15s %15s %15s\n" "${num_workers}" "NO_DATA" "NO_DATA" "0" >> "$SUMMARY_FILE"
        fi
    done
done

# Compare with baseline if available
if [ -d "$BASELINE_DIR" ]; then
    echo "" >> "$SUMMARY_FILE"
    echo "=============================================================================" >> "$SUMMARY_FILE"
    echo "Comparison with Baseline" >> "$SUMMARY_FILE"
    echo "=============================================================================" >> "$SUMMARY_FILE"
    echo "Baseline directory: ${BASELINE_DIR}" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    for prompt_len in "${PROMPT_ARRAY[@]}"; do
        BASELINE_PREFILL="${BASELINE_DIR}/${prompt_len}/prefill.csv"
        if [ -f "$BASELINE_PREFILL" ]; then
            # Get baseline AllGatherAsync stats
            BASELINE_STATS=$(python3 << EOF
import pandas as pd
try:
    df = pd.read_csv("${BASELINE_PREFILL}")
    ag_df = df[df['OP CODE'].str.contains('AllGatherAsync', case=False, na=False)]
    if len(ag_df) > 0:
        device_time = ag_df['DEVICE TIME'].sum()
        total_time = ag_df['OP TO OP LATENCY'].sum() if 'OP TO OP LATENCY' in ag_df.columns else device_time
        count = len(ag_df)
        print(f"{device_time:.1f},{total_time:.1f},{count}")
    else:
        print("N/A,N/A,0")
except Exception as e:
    print(f"ERROR,ERROR,0")
EOF
)
            echo "" >> "$SUMMARY_FILE"
            echo "Prompt Length: ${prompt_len}" >> "$SUMMARY_FILE"
            echo "Baseline AllGatherAsync: ${BASELINE_STATS}" >> "$SUMMARY_FILE"
        fi
    done
fi

cat "$SUMMARY_FILE"

log_header "Sweep Complete"
echo ""
echo "Results saved to: ${SWEEP_DIR}/"
echo "Summary: ${SUMMARY_FILE}"
echo ""
echo "Individual run results:"
for num_workers in "${WORKER_VALUES[@]}"; do
    echo "  - workers_${num_workers}/: ${SWEEP_DIR}/workers_${num_workers}/"
done
