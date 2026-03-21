#!/bin/bash
#
# Run profiler sweep across multiple prompt lengths for GPT-OSS
#
# This script automates profiling the single_layer_profile test with different input
# prompt lengths (128, 1k, 4k, 8k, 16k, 32k tokens). It handles:
#   1. Running tracy profiler for each prompt length
#   2. Parsing raw profiler output into prefill.csv and decode.csv
#   3. Generating comparison summaries across prompt lengths
#   4. Comparing two different runs (baseline vs optimized)
#
# =============================================================================
# USAGE
# =============================================================================
#
#   ./scripts/run_profiler_sweep_gpt_oss.sh [options]
#
# =============================================================================
# OPTIONS
# =============================================================================
#
#   --output-dir DIR      Base output directory (default: profiler_sweep_results_gpt_oss)
#   --run-name NAME       Name for this run (default: timestamp YYYYMMDD_HHMMSS)
#   --parse-only          Skip profiler runs, only parse existing reports
#   --compare-runs R1 R2  Compare two previous runs and generate Excel report
#   --prompt-lengths L    Comma-separated list of prompt lengths to run
#                         (default: all - 128,1k,4k,8k,16k,32k)
#   --hf-model PATH       Path to HuggingFace model (default: /home/cust-team/models/models--openai--gpt-oss-120b/snapshots/gpt-oss-120b)
#   -h, --help            Show help message
#
# =============================================================================
# EXAMPLES
# =============================================================================
#
#   # Run full sweep with auto-generated timestamp name
#   ./scripts/run_profiler_sweep_gpt_oss.sh
#
#   # Run sweep with custom name for later reference
#   ./scripts/run_profiler_sweep_gpt_oss.sh --run-name baseline
#   ./scripts/run_profiler_sweep_gpt_oss.sh --run-name optimized_v2
#
#   # Run only specific prompt lengths (useful for quick tests)
#   ./scripts/run_profiler_sweep_gpt_oss.sh --prompt-lengths 128,4k
#   ./scripts/run_profiler_sweep_gpt_oss.sh --prompt-lengths 32k --run-name large_only
#
#   # Parse existing raw profiler data (if parsing failed or needs re-run)
#   ./scripts/run_profiler_sweep_gpt_oss.sh --parse-only --run-name baseline
#
#   # Compare two runs and generate Excel report
#   ./scripts/run_profiler_sweep_gpt_oss.sh --compare-runs baseline 20260129_022518
#

set -e

# Configuration
OUTPUT_BASE_DIR="profiler_sweep_results_gpt_oss"
RUN_NAME=""
PARSE_ONLY=false
COMPARE_RUNS=""
SELECTED_PROMPTS=""
TEST_FILE="models/demos/gpt_oss/demo/text_demo.py"
SAMPLE_PROMPTS_DIR="models/tt_transformers/demo/sample_prompts"
HF_MODEL_PATH="/home/cust-team/models/models--openai--gpt-oss-120b/snapshots/gpt-oss-120b"
MESH_SHAPE="4x8"  # Default to Galaxy (4x8), can also use "1x8" for LoudBox

# Prompt lengths and their corresponding input files
declare -A PROMPT_FILES=(
    ["128"]="models/demos/gpt_oss/demo/sample_prompts/input_data_questions_prefill_128.json"
    ["1k"]="models/tt_transformers/demo/sample_prompts/input_data_long_1k.json"
    ["2k"]="models/tt_transformers/demo/sample_prompts/input_data_long_2k.json"
    ["4k"]="models/tt_transformers/demo/sample_prompts/input_data_long_4k.json"
    ["8k"]="models/tt_transformers/demo/sample_prompts/input_data_long_8k.json"
    ["16k"]="models/tt_transformers/demo/sample_prompts/input_data_long_16k.json"
    ["32k"]="models/tt_transformers/demo/sample_prompts/input_data_long_32k.json"
)

# Prompt length names to actual sequence lengths (for SDPA filtering)
declare -A SEQ_LENGTHS=(
    ["128"]=128
    ["1k"]=1024
    ["2k"]=2048
    ["4k"]=4096
    ["8k"]=8192
    ["16k"]=16384
    ["32k"]=32768
)

# Max sequence lengths for each prompt (must be >= prompt length, power of 2 preferred)
declare -A MAX_SEQ_LENS=(
    ["128"]=128
    ["1k"]=1024
    ["2k"]=2048
    ["4k"]=4096
    ["8k"]=8192
    ["16k"]=16384
    ["32k"]=32768
)

# Order of execution (smallest to largest)
ALL_PROMPT_ORDER=("128" "1k" "2k" "4k" "8k" "16k" "32k")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${BLUE}=============================================="
    echo -e "$1"
    echo -e "==============================================${NC}"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --output-dir)
                OUTPUT_BASE_DIR="$2"
                shift 2
                ;;
            --run-name)
                RUN_NAME="$2"
                shift 2
                ;;
            --parse-only)
                PARSE_ONLY=true
                shift
                ;;
            --compare-runs)
                COMPARE_RUNS="$2 $3"
                shift 3
                ;;
            --prompt-lengths)
                SELECTED_PROMPTS="$2"
                shift 2
                ;;
            --hf-model)
                HF_MODEL_PATH="$2"
                shift 2
                ;;
            --mesh-shape)
                MESH_SHAPE="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --output-dir DIR      Base output directory (default: profiler_sweep_results_gpt_oss)"
                echo "  --run-name NAME       Name for this run (default: timestamp)"
                echo "  --parse-only          Skip profiler runs, only parse existing reports"
                echo "  --compare-runs R1 R2  Compare two previous runs"
                echo "  --prompt-lengths L    Comma-separated list (e.g., '128,4k,8k')"
                echo "  --hf-model PATH       Path to HuggingFace model"
                echo "  --mesh-shape SHAPE    Mesh shape: '4x8' (Galaxy) or '1x8' (LoudBox). Default: 4x8"
                echo ""
                echo "Examples:"
                echo "  $0                                    # Run full sweep with timestamp"
                echo "  $0 --run-name baseline                # Run with custom name"
                echo "  $0 --prompt-lengths 128,32k           # Run only 128 and 32k"
                echo "  $0 --parse-only --run-name baseline   # Parse existing reports"
                echo "  $0 --compare-runs baseline optimized  # Compare two runs"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# Get prompt order based on selection
get_prompt_order() {
    if [ -n "$SELECTED_PROMPTS" ]; then
        IFS=',' read -ra PROMPT_ORDER <<< "$SELECTED_PROMPTS"
    else
        PROMPT_ORDER=("${ALL_PROMPT_ORDER[@]}")
    fi
}

# Backup the original test file
backup_test_file() {
    if [ ! -f "${TEST_FILE}.backup_profiler_sweep" ]; then
        cp "$TEST_FILE" "${TEST_FILE}.backup_profiler_sweep"
        log_info "Backed up test file"
    fi
}

# Restore the original test file
restore_test_file() {
    if [ -f "${TEST_FILE}.backup_profiler_sweep" ]; then
        cp "${TEST_FILE}.backup_profiler_sweep" "$TEST_FILE"
        log_info "Restored original test file"
    fi
}

# Modify the test file to use a different input file and max_seq_len for single_layer_profile test
set_input_file() {
    local input_file="$1"
    local max_seq_len="$2"

    # Use Python to precisely modify only the single_layer_profile test case
    python3 << EOF
import re

with open("$TEST_FILE", "r") as f:
    content = f.read()

# Pattern to find the single layer profiling test block and replace its input file
pattern = r'(# Single layer profiling[^\n]*\n\s*\(\s*\n\s*)"[^"]+",(\s*#\s*input_prompts)'
replacement = r'\1"${input_file}",\2'
new_content = re.sub(pattern, replacement, content)

# Find and update max_seq_len and page_params for this test case
lines = new_content.split('\n')
in_single_layer_block = False
max_seq_len_updated = False
page_params_updated = False

for i, line in enumerate(lines):
    if '# Single layer profiling' in line:
        in_single_layer_block = True

    if in_single_layer_block:
        # Update max_seq_len line (handles both "8 * 1024," and "8192," formats)
        if '# max_seq_len' in line and not max_seq_len_updated:
            old_line = lines[i]
            lines[i] = re.sub(r'\d+\s*\*\s*1024,', '${max_seq_len},', line)
            if lines[i] == old_line:  # Pattern didn't match, try direct number
                lines[i] = re.sub(r'^\s*\d+,(\s*#\s*max_seq_len)', '            ${max_seq_len},\\1', lines[i])
            max_seq_len_updated = True

        # Update page_params line
        if '# page_params' in line and not page_params_updated:
            # Calculate blocks needed: max_seq_len / block_size
            blocks = ${max_seq_len} // 64
            lines[i] = '            {"page_block_size": 64, "page_max_num_blocks_per_dp": ' + str(blocks) + '},  # page_params'
            page_params_updated = True

        # Stop when we hit num_layers (end of this test case)
        if '# num_layers' in line:
            in_single_layer_block = False
            break

new_content = '\n'.join(lines)

with open("$TEST_FILE", "w") as f:
    f.write(new_content)

print("Modified single_layer_profile test: input_file=${input_file}, max_seq_len=${max_seq_len}")
EOF

    log_info "Set input file to: $input_file, max_seq_len: $max_seq_len"
}

# Find the latest profiler report directory
find_latest_profiler_report() {
    local latest_dir=$(ls -td generated/profiler/reports/*/ 2>/dev/null | head -1)
    if [ -n "$latest_dir" ]; then
        local latest_csv=$(ls -t "${latest_dir}"ops_perf_results_*.csv 2>/dev/null | head -1)
        if [ -n "$latest_csv" ]; then
            echo "$latest_csv"
            return 0
        fi
    fi
    # Fallback to non-timestamped directory
    local latest_csv=$(ls -t generated/profiler/reports/ops_perf_results_*.csv 2>/dev/null | head -1)
    if [ -n "$latest_csv" ]; then
        echo "$latest_csv"
        return 0
    fi
    return 1
}

# Run the profiler
run_profiler() {
    local prompt_len="$1"
    local output_dir="$2"

    mkdir -p "$output_dir"

    log_info "Running profiler for prompt length: ${prompt_len}"
    log_info "Output directory: ${output_dir}"

    # Set environment and run profiler
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)

    # Activate virtual environment and run
    source python_env/bin/activate

    # Use mesh shape directly in filter (e.g., "4x8" not "mesh_4x8")
    # Note: tracy's argument parser has issues with "and" keyword when passed
    # as separate args, so we wrap the entire pytest command in quotes after -m
    local pytest_filter="single_layer_profile and ${MESH_SHAPE}"

    # Run with tracy profiler - wrap entire pytest command in quotes
    HF_MODEL="$HF_MODEL_PATH" \
        python -m tracy -p -v -r --op-support-count 20000 \
        -m "pytest ${TEST_FILE} -k '${pytest_filter}'" \
        2>&1 | tee "${output_dir}/profiler_output.log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        log_error "Profiler run failed for ${prompt_len} with exit code ${exit_code}"
        return $exit_code
    fi

    # Find and copy the latest profiler report
    local latest_csv=$(find_latest_profiler_report)
    if [ -n "$latest_csv" ]; then
        cp "$latest_csv" "${output_dir}/"
        log_info "Copied profiler results: $(basename $latest_csv)"

        # Also copy the full profiler directory for reference
        local profiler_dir=$(dirname "$latest_csv")
        log_info "Profiler report directory: ${profiler_dir}"
    else
        log_warn "No ops_perf_results CSV found for ${prompt_len}"
        return 1
    fi

    return 0
}

# Parse profiler results for a prompt length
parse_results() {
    local prompt_len="$1"
    local output_dir="$2"

    local ops_csv=$(ls -t "${output_dir}"/ops_perf_results_*.csv 2>/dev/null | head -1)
    if [ -z "$ops_csv" ]; then
        log_warn "No ops_perf_results CSV found in ${output_dir}"
        return 1
    fi

    log_info "Parsing results for ${prompt_len}: $(basename $ops_csv)"

    # Get the actual sequence length for SDPA filtering
    local seq_len="${SEQ_LENGTHS[$prompt_len]}"

    python scripts/parse_profiler_report.py \
        "$ops_csv" \
        --device-id 0 \
        --summary --no-outliers \
        --seq-len "$seq_len" \
        --output-dir "${output_dir}" \
        2>&1 | tee "${output_dir}/summary.txt"

    log_info "Generated parsed output and summary"
    return 0
}

# Generate comparison across prompt lengths
generate_comparison() {
    local run_dir="$1"

    log_info "Generating comparison across prompt lengths..."

    # Run aggregated comparison
    python scripts/compare_profiler_results.py \
        "${run_dir}" \
        --output-dir "${run_dir}" \
        2>&1 | tee "${run_dir}/comparison_summary.txt"

    # Run raw comparison
    python scripts/compare_ops_raw.py \
        "${run_dir}" \
        --output-dir "${run_dir}"

    # Generate Excel files with gradient formatting (skip silently if xlsxwriter unavailable)
    python scripts/csv_to_xlsx_with_gradient.py \
        "${run_dir}" \
        --skip-if-unavailable 2>/dev/null || true

    log_info "Generated comparison files"
}

# Compare two runs
compare_two_runs() {
    local run1="$1"
    local run2="$2"
    local run1_dir="${OUTPUT_BASE_DIR}/${run1}"
    local run2_dir="${OUTPUT_BASE_DIR}/${run2}"

    if [ ! -d "$run1_dir" ]; then
        log_error "Run directory not found: $run1_dir"
        return 1
    fi
    if [ ! -d "$run2_dir" ]; then
        log_error "Run directory not found: $run2_dir"
        return 1
    fi

    log_header "Comparing runs: $run1 vs $run2"

    # Use the Python comparison script for detailed Excel output
    python scripts/compare_two_runs.py "$run1" "$run2" --output-dir "$OUTPUT_BASE_DIR"

    echo ""
    log_info "Excel comparison saved to: ${OUTPUT_BASE_DIR}/comparison_${run1}_vs_${run2}.xlsx"
}

# List available runs
list_runs() {
    echo "Available runs in ${OUTPUT_BASE_DIR}:"
    ls -d "${OUTPUT_BASE_DIR}"/*/ 2>/dev/null | while read dir; do
        local run_name=$(basename "$dir")
        local num_prompts=$(ls -d "${dir}"/*/ 2>/dev/null | wc -l)
        echo "  - ${run_name} (${num_prompts} prompt lengths)"
    done
}

# Main execution
main() {
    parse_args "$@"
    get_prompt_order

    # Handle compare mode
    if [ -n "$COMPARE_RUNS" ]; then
        read -r run1 run2 <<< "$COMPARE_RUNS"
        compare_two_runs "$run1" "$run2"
        exit 0
    fi

    # Generate run name with timestamp if not provided
    if [ -z "$RUN_NAME" ]; then
        RUN_NAME=$(date +"%Y%m%d_%H%M%S")
    fi

    local RUN_DIR="${OUTPUT_BASE_DIR}/${RUN_NAME}"

    log_header "GPT-OSS Profiler Sweep: ${RUN_NAME}"
    log_info "Output directory: ${RUN_DIR}"
    log_info "Prompt lengths: ${PROMPT_ORDER[*]}"
    log_info "Parse only: ${PARSE_ONLY}"
    log_info "HF Model: ${HF_MODEL_PATH}"
    log_info "Mesh shape: ${MESH_SHAPE}"

    # Create output directory
    mkdir -p "$RUN_DIR"

    # Save run configuration
    cat > "${RUN_DIR}/run_config.txt" << EOF
Run Name: ${RUN_NAME}
Date: $(date)
Model: GPT-OSS
HF Model Path: ${HF_MODEL_PATH}
Mesh Shape: ${MESH_SHAPE}
Prompt Lengths: ${PROMPT_ORDER[*]}
Parse Only: ${PARSE_ONLY}
EOF

    if [ "$PARSE_ONLY" = false ]; then
        # Backup test file
        backup_test_file

        # Trap to restore test file on exit
        trap restore_test_file EXIT
    fi

    # Track results
    declare -A RESULTS

    for prompt_len in "${PROMPT_ORDER[@]}"; do
        input_file="${PROMPT_FILES[$prompt_len]}"
        max_seq_len="${MAX_SEQ_LENS[$prompt_len]}"
        output_dir="${RUN_DIR}/${prompt_len}"

        log_header "Processing: ${prompt_len}"
        log_info "Input file: ${input_file}"
        log_info "Max seq len: ${max_seq_len}"

        if [ "$PARSE_ONLY" = false ]; then
            # Restore original first, then modify
            restore_test_file
            backup_test_file
            set_input_file "$input_file" "$max_seq_len"

            if run_profiler "$prompt_len" "$output_dir"; then
                if parse_results "$prompt_len" "$output_dir"; then
                    RESULTS[$prompt_len]="SUCCESS"
                else
                    RESULTS[$prompt_len]="PARSE_FAILED"
                fi
            else
                RESULTS[$prompt_len]="RUN_FAILED"
            fi
        else
            # Parse only mode - look for existing results
            if [ -d "$output_dir" ]; then
                if parse_results "$prompt_len" "$output_dir"; then
                    RESULTS[$prompt_len]="SUCCESS"
                else
                    RESULTS[$prompt_len]="PARSE_FAILED"
                fi
            else
                log_warn "No existing results for ${prompt_len} in ${output_dir}"
                RESULTS[$prompt_len]="NOT_FOUND"
            fi
        fi

        echo ""
    done

    # Generate comparison
    generate_comparison "$RUN_DIR"

    # Print summary
    log_header "GPT-OSS Profiler Sweep Complete: ${RUN_NAME}"
    echo ""
    echo "Results:"
    for prompt_len in "${PROMPT_ORDER[@]}"; do
        status="${RESULTS[$prompt_len]}"
        if [ "$status" = "SUCCESS" ]; then
            echo -e "  ${prompt_len}: ${GREEN}${status}${NC}"
        else
            echo -e "  ${prompt_len}: ${RED}${status}${NC}"
        fi
    done
    echo ""
    echo "Output saved to: ${RUN_DIR}/"
    echo ""
    echo "Files generated:"
    echo "  - Per prompt length: ${RUN_DIR}/<length>/prefill.csv, decode.csv, summary.txt"
    echo "  - Comparison: ${RUN_DIR}/prefill_comparison.csv, decode_comparison.csv"
    echo "  - Raw comparison: ${RUN_DIR}/prefill_raw_comparison.csv, decode_raw_comparison.csv"
    echo "  - Comparison summary: ${RUN_DIR}/comparison_summary.txt"
    echo ""
    echo "To view comparison summary:"
    echo "  cat ${RUN_DIR}/comparison_summary.txt"
    echo ""
    echo "To compare with another run:"
    echo "  $0 --compare-runs ${RUN_NAME} <other_run_name>"
    echo ""
    echo "Available runs:"
    list_runs
}

# Run main
main "$@"
