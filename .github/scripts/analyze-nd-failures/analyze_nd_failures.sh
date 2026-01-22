#!/bin/bash

# Main script to analyze non-deterministic failures from GitHub Actions jobs
# Usage: ./analyze_nd_failures.sh [OPTIONS] <job_url_1> [job_url_2] ... [job_url_n]
#
# This script:
# 1. Downloads logs from GitHub Actions job URLs
# 2. Prepares analysis context (logs, test files, code context)
# 3. Uses GitHub Copilot CLI to analyze the failures and suggest fixes

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="${SCRIPT_DIR}/analysis_prompt.md"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_job_logs.sh"
OUTPUT_DIR="${SCRIPT_DIR}/build_analysis_output"
LOG_DIR="${SCRIPT_DIR}/build_downloaded_logs"
REPO_ROOT="${SCRIPT_DIR}/../../.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1" >&2
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    local missing_tools=()

    # Check gh CLI
    if ! command -v gh &> /dev/null; then
        missing_tools+=("gh (GitHub CLI)")
    fi

    # Check jq
    if ! command -v jq &> /dev/null; then
        missing_tools+=("jq")
    fi

    # Check copilot CLI
    if ! command -v github-copilot-cli &> /dev/null && ! command -v copilot &> /dev/null; then
        log_warn "GitHub Copilot CLI not found. Will attempt to use 'copilot' command."
        log_warn "If this fails, please install GitHub Copilot CLI:"
        log_warn "  npm install -g @githubnext/github-copilot-cli"
    fi

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools:"
        for tool in "${missing_tools[@]}"; do
            echo "  - $tool"
        done
        exit 1
    fi

    # Check GitHub authentication
    if ! gh auth status &> /dev/null; then
        log_error "GitHub CLI is not authenticated. Please run 'gh auth login' first."
        exit 1
    fi

    # Check if prompt file exists
    if [[ ! -f "$PROMPT_FILE" ]]; then
        log_error "Prompt file not found: $PROMPT_FILE"
        exit 1
    fi

    # Check if download script exists
    if [[ ! -f "$DOWNLOAD_SCRIPT" ]]; then
        log_error "Download script not found: $DOWNLOAD_SCRIPT"
        exit 1
    fi

    log_info "All prerequisites met"
}

# Function to find copilot CLI command
find_copilot_cmd() {
    if command -v github-copilot-cli &> /dev/null; then
        echo "github-copilot-cli"
    elif command -v copilot &> /dev/null; then
        echo "copilot"
    else
        log_error "GitHub Copilot CLI not found. Please install it:"
        log_error "  npm install -g @githubnext/github-copilot-cli"
        exit 1
    fi
}

# Function to prepare analysis context
prepare_analysis_context() {
    local job_dir=$1
    local context_dir="${OUTPUT_DIR}/context_$(basename "$job_dir")"

    mkdir -p "$context_dir"

    log_info "Preparing analysis context for $(basename "$job_dir")"

    # Copy logs - Copilot will analyze them directly
    cp -r "${job_dir}/logs" "${context_dir}/" 2>/dev/null || true

    echo "$context_dir"
}

# Function to extract failure metadata from logs
extract_failure_metadata() {
    local log_file=$1

    [[ ! -f "$log_file" ]] && return 1

    # Extract just the essential failure information:
    # - Failed test names
    # - Error messages
    # - File paths mentioned in errors
    # - Stack trace snippets (first few lines)

    {
        # Extract failed test names
        grep -E "FAILED|ERROR" "$log_file" | grep -E "test_" | head -10

        # Extract error messages (RuntimeError, Exception, etc.)
        grep -E "(RuntimeError|Exception|Error|TT_THROW)" "$log_file" | head -20

        # Extract file paths mentioned in errors (look for /project/ or file paths)
        grep -E "(/project/|\.py:|\.cpp:|\.hpp:)" "$log_file" | head -10

        # Extract pytest failure summary
        grep -A 5 "FAILURES ==" "$log_file" | head -20

    } | sort -u
}

# Function to create analysis prompt with context
create_analysis_prompt() {
    local context_dir=$1
    local output_file="${context_dir}/full_prompt.md"

    # Start with the base prompt
    cat "$PROMPT_FILE" > "$output_file"

    # Add failure metadata instead of full logs
    cat >> "$output_file" <<EOF

---

## Failure Metadata

The following information was extracted from the failed GitHub Actions job logs.
Use this information to locate and analyze the relevant code files in the repository.

EOF

    # Extract and include only failure metadata
    for log_file in "${context_dir}"/logs/*.log; do
        [[ ! -f "$log_file" ]] && continue
        local log_name=$(basename "$log_file")
        local metadata
        metadata=$(extract_failure_metadata "$log_file" 2>/dev/null)

        if [[ -n "$metadata" ]]; then
            cat >> "$output_file" <<EOF

### Failure Information from: $log_name

\`\`\`
$metadata
\`\`\`

EOF
        fi
    done

    cat >> "$output_file" <<EOF

---

**Note**: you should be tracking exactly what is being done as the test reaches the failure to determine the root instability. Make sure to look through the codebase to figure out what went wrong.

EOF

    echo "$output_file"
}

# Function to run copilot analysis
run_copilot_analysis() {
    local prompt_file=$1
    local output_file=$2
    local copilot_cmd=$3

    log_info "Running Copilot analysis..."
    log_info "Prompt file: $prompt_file"
    log_info "Output file: $output_file"

    # Print the full prompt before sending to Copilot
    echo ""
    log_info "=== Full Prompt to be sent to Copilot ==="
    cat "$prompt_file"
    echo ""
    log_info "=== End of Prompt ==="
    echo ""

    log_info "Note: This may take several minutes. Large prompts can take time to process..."

    # Pipe prompt to copilot (reads from stdin)
    # Use --allow-all-tools for non-interactive mode
    # Use tee to show output in real-time while also saving to file
    if command -v timeout &> /dev/null; then
        log_info "Using timeout (10 minute limit)"
        if timeout 600 sh -c "cat '$prompt_file' | '$copilot_cmd' --allow-all-tools" 2>&1 | tee "$output_file"; then
            local exit_code=0
        else
            local exit_code=$?
            if [[ $exit_code -eq 124 ]]; then
                log_error "Copilot analysis timed out after 10 minutes"
                return 1
            fi
        fi
    else
        # No timeout - run directly
        if cat "$prompt_file" | "$copilot_cmd" --allow-all-tools 2>&1 | tee "$output_file"; then
            local exit_code=0
        else
            local exit_code=$?
        fi
    fi

    # Check if we got valid output
    if [[ $exit_code -eq 0 ]] && [[ -s "$output_file" ]]; then
        log_info "Analysis complete"
        return 0
    else
        log_error "Copilot analysis failed (exit code: $exit_code)"
        log_info "Check the output file for details: $output_file"
        return 1
    fi
}

# Main function
main() {
    local urls=()
    local urls_file=""
    local skip_download=false
    local keep_output=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --file|-f)
                urls_file=$2
                shift 2
                ;;
            --output-dir|-o)
                OUTPUT_DIR=$2
                shift 2
                ;;
            --skip-download)
                skip_download=true
                shift
                ;;
            --keep-output|--no-cleanup)
                keep_output=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS] <job_url_1> [job_url_2] ... [job_url_n]"
                echo ""
                echo "Options:"
                echo "  --file, -f <file>       Read URLs from file (one per line)"
                echo "  --output-dir, -o <dir>  Output directory (default: ./build_analysis_output)"
                echo "  --skip-download         Skip downloading logs (use existing downloads)"
                echo "  --keep-output           Keep output and downloaded folders (don't clean up)"
                echo "  --help, -h              Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0 https://github.com/tenstorrent/tt-metal/actions/runs/1234567890/job/9876543210"
                echo "  $0 --file urls.txt"
                exit 0
                ;;
            *)
                urls+=("$1")
                shift
                ;;
        esac
    done

    # Clean up previous runs by default (unless --keep-output is specified)
    if [[ "$keep_output" == false ]]; then
        log_info "Cleaning up previous run data..."
        if [[ -d "$OUTPUT_DIR" ]]; then
            rm -rf "$OUTPUT_DIR"
            log_info "Removed output directory: $OUTPUT_DIR"
        fi
        if [[ "$skip_download" == false && -d "$LOG_DIR" ]]; then
            rm -rf "$LOG_DIR"
            log_info "Removed downloaded logs directory: $LOG_DIR"
        fi
    else
        log_info "Keeping existing output (--keep-output specified)"
    fi

    # Read URLs from file if provided
    if [[ -n "$urls_file" ]]; then
        if [[ ! -f "$urls_file" ]]; then
            log_error "File not found: $urls_file"
            exit 1
        fi
        while IFS= read -r line || [[ -n "$line" ]]; do
            [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
            urls+=("$line")
        done < "$urls_file"
    fi

    # Check prerequisites
    check_prerequisites

    # Find copilot command
    local copilot_cmd
    copilot_cmd=$(find_copilot_cmd)

    # Download logs if not skipping
    if [[ "$skip_download" == false ]]; then
        if [[ ${#urls[@]} -eq 0 ]]; then
            log_error "No URLs provided. Use --help for usage information."
            exit 1
        fi

        log_info "Downloading logs for ${#urls[@]} job URL(s)..."
        bash "$DOWNLOAD_SCRIPT" "${urls[@]}" || {
            log_error "Failed to download logs"
            exit 1
        }
    else
        log_info "Skipping download, using existing logs in $LOG_DIR"
    fi

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Process each downloaded job directory
    local job_dirs=()
    if [[ -d "$LOG_DIR" ]]; then
        while IFS= read -r job_dir; do
            [[ -d "$job_dir" ]] && job_dirs+=("$job_dir")
        done < <(find "$LOG_DIR" -type d -name "run_*" -maxdepth 1 2>/dev/null || true)
    fi

    if [[ ${#job_dirs[@]} -eq 0 ]]; then
        log_error "No job directories found. Please download logs first."
        exit 1
    fi

    log_info "Found ${#job_dirs[@]} job directory(ies) to analyze"

    # Prepare context and analyze each job
    local analysis_results=()
    for job_dir in "${job_dirs[@]}"; do
        log_info "Processing: $(basename "$job_dir")"

        # Prepare analysis context
        local context_dir
        context_dir=$(prepare_analysis_context "$job_dir")

        # Create full prompt
        local prompt_file
        prompt_file=$(create_analysis_prompt "$context_dir")

        # Run analysis
        local output_file="${context_dir}/analysis_result.md"
        run_copilot_analysis "$prompt_file" "$output_file" "$copilot_cmd"

        analysis_results+=("$output_file")
    done

    # Create combined analysis if multiple jobs
    if [[ ${#analysis_results[@]} -gt 1 ]]; then
        log_info "Creating combined analysis for ${#analysis_results[@]} jobs..."
        local combined_output="${OUTPUT_DIR}/combined_analysis.md"
        {
            echo "# Combined Non-Deterministic Failure Analysis"
            echo ""
            echo "This analysis combines findings from ${#analysis_results[@]} failed job(s)."
            echo ""
            echo "Generated: $(date)"
            echo ""
            for result_file in "${analysis_results[@]}"; do
                echo "---"
                echo ""
                echo "## Analysis for $(basename "$(dirname "$result_file")")"
                echo ""
                cat "$result_file"
                echo ""
            done
        } > "$combined_output"
        log_info "Combined analysis saved to: $combined_output"
    fi

    # Summary
    echo ""
    log_info "=== Analysis Complete ==="
    log_info "Results saved to: $OUTPUT_DIR"
    for result_file in "${analysis_results[@]}"; do
        echo "  - $result_file"
    done
    echo ""
}

main "$@"
