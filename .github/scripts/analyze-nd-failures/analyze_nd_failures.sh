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
REPO_ROOT="${SCRIPT_DIR}/../../.."
BASE_OUTPUT_DIR="${REPO_ROOT}/build_ND_analysis"

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
    local log_dir=$1
    local context_dir="${OUTPUT_DIR}/context"

    mkdir -p "$context_dir"

    log_info "Preparing analysis context"

    # Copy logs - Copilot will analyze them directly
    if [[ -d "${log_dir}/logs" ]]; then
        cp -r "${log_dir}/logs" "${context_dir}/" 2>/dev/null || true
    elif [[ -d "$log_dir" ]]; then
        # If logs directory doesn't exist, copy the entire log_dir structure
        cp -r "$log_dir"/* "${context_dir}/" 2>/dev/null || true
    fi

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

# Function to extract primary error message from logs for folder naming
extract_primary_error() {
    local log_file=$1

    [[ ! -f "$log_file" ]] && echo "" && return 0

    # Try to find the most prominent error message
    # Look for common ND failure patterns first
    local error_msg=""

    # Check for device timeout
    if grep -qi "device.*timeout\|timeout.*device\|timed out" "$log_file" 2>/dev/null; then
        error_msg="device_timeout"
    # Check for initialization failure
    elif grep -qi "failed to initialize\|initialization.*fail\|init.*fail" "$log_file" 2>/dev/null; then
        error_msg="init_failure"
    # Check for connection/discovery issues
    elif grep -qi "connection.*mismatch\|missing.*channel\|missing.*port\|discovery.*fail" "$log_file" 2>/dev/null; then
        error_msg="connection_issue"
    # Check for hardware errors
    elif grep -qi "hardware.*error\|chip.*error\|device.*error" "$log_file" 2>/dev/null; then
        error_msg="hardware_error"
    # Check for resource exhaustion
    elif grep -qi "out of memory\|resource.*exhaust\|handle.*exhaust" "$log_file" 2>/dev/null; then
        error_msg="resource_exhaustion"
    # Try to extract first significant error message
    else
        error_msg=$(grep -E "(RuntimeError|Exception|Error|TT_THROW|FAILED)" "$log_file" 2>/dev/null | head -1 | sed 's/.*\(RuntimeError\|Exception\|Error\|TT_THROW\|FAILED\)[^:]*: *\([^[:space:]]*\).*/\2/' | head -c 30 | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]_' || echo "unknown_error")
    fi

    # Sanitize: replace spaces with underscores, convert to lowercase, remove special chars, limit length
    if [[ -z "$error_msg" ]]; then
        error_msg="unknown_error"
    fi
    echo "$error_msg" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | sed 's/[^[:alnum:]_]//g' | sed 's/__*/_/g' | head -c 30
}

# Function to extract job name from job info JSON
extract_job_name() {
    local job_dir=$1

    # Look for job_info.json in logs directory
    local job_info_file=$(find "$job_dir" -name "*_job_info.json" -type f | head -1)

    # Try to extract job name from job_info.json
    local job_name=""
    if [[ -f "$job_info_file" ]]; then
        job_name=$(jq -r '.name // .labels[]? // empty' "$job_info_file" 2>/dev/null | head -1)

        # If not found, try workflow name
        if [[ -z "$job_name" || "$job_name" == "null" ]]; then
            job_name=$(jq -r '.workflow_name // empty' "$job_info_file" 2>/dev/null)
        fi
    fi

    # If still not found, try to get from workflow_jobs.json
    if [[ -z "$job_name" || "$job_name" == "null" ]]; then
        local workflow_jobs_file=$(find "$job_dir" -name "workflow_jobs.json" -type f | head -1)
        if [[ -f "$workflow_jobs_file" ]]; then
            job_name=$(jq -r '.jobs[0].name // empty' "$workflow_jobs_file" 2>/dev/null)
        fi
    fi

    # Fallback: use job_id from job_info filename
    if [[ -z "$job_name" || "$job_name" == "null" ]]; then
        if [[ -n "$job_info_file" ]]; then
            job_name=$(basename "$job_info_file" .json | sed 's/_job_info//')
        else
            # Last resort: use run_id from directory name
            job_name=$(basename "$job_dir" | sed 's/run_\([0-9]*\).*/\1/')
            [[ -n "$job_name" ]] && job_name="run_${job_name}"
        fi
    fi

    # Sanitize: remove special chars, limit length, replace spaces with underscores
    echo "$job_name" | tr '[:upper:]' '[:lower:]' | sed 's/[^[:alnum:]_]//g' | sed 's/__*/_/g' | sed 's/^_\|_$//g' | head -c 40
}

# Function to create run-specific directory name
create_run_directory_name() {
    local job_dir=$1

    # Extract job name
    local job_name
    job_name=$(extract_job_name "$job_dir")

    # Fallback to run_id if job_name is empty
    if [[ -z "$job_name" ]]; then
        local run_id=$(basename "$job_dir" | sed 's/run_\([0-9]*\).*/\1/')
        job_name="run_${run_id}"
    fi

    # Extract primary error from logs
    local error_abbrev=""
    local log_file=$(find "$job_dir" -name "*.log" -type f | head -1)
    if [[ -n "$log_file" ]]; then
        error_abbrev=$(extract_primary_error "$log_file")
    fi

    # Fallback if no error found
    if [[ -z "$error_abbrev" ]]; then
        error_abbrev="unknown"
    fi

    # Combine: job_name_error_abbrev
    local dir_name="${job_name}_${error_abbrev}"

    # Final sanitization
    dir_name=$(echo "$dir_name" | tr '[:upper:]' '[:lower:]' | sed 's/[^[:alnum:]_]//g' | sed 's/__*/_/g' | sed 's/^_\|_$//g')

    echo "$dir_name"
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
    local custom_output_dir=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --file|-f)
                urls_file=$2
                shift 2
                ;;
            --output-dir|-o)
                custom_output_dir=$2
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
                echo "  --output-dir, -o <dir>  Base output directory (default: <repo_root>/build_ND_analysis)"
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

    # Set base output directory
    if [[ -n "$custom_output_dir" ]]; then
        BASE_OUTPUT_DIR="$custom_output_dir"
    fi
    mkdir -p "$BASE_OUTPUT_DIR"

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

    # Temporary directory for initial downloads (before we know the run name)
    local temp_log_dir="${BASE_OUTPUT_DIR}/.temp_downloads"

    # Download logs if not skipping
    if [[ "$skip_download" == false ]]; then
        if [[ ${#urls[@]} -eq 0 ]]; then
            log_error "No URLs provided. Use --help for usage information."
            exit 1
        fi

        log_info "Downloading logs for ${#urls[@]} job URL(s)..."
        # Download to temp directory first
        bash "$DOWNLOAD_SCRIPT" --output-dir "$temp_log_dir" "${urls[@]}" || {
            log_error "Failed to download logs"
            exit 1
        }
    else
        log_info "Skipping download, looking for existing logs..."
        # Try to find existing logs in BASE_OUTPUT_DIR
        temp_log_dir="$BASE_OUTPUT_DIR"
    fi

    # Process each downloaded job directory
    local job_dirs=()
    if [[ -d "$temp_log_dir" ]]; then
        while IFS= read -r job_dir; do
            [[ -d "$job_dir" ]] && job_dirs+=("$job_dir")
        done < <(find "$temp_log_dir" -type d -name "run_*" -maxdepth 1 2>/dev/null || true)
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

        # Create run-specific directory name
        local run_dir_name
        run_dir_name=$(create_run_directory_name "$job_dir")

        # Create the run-specific directory structure
        local run_output_dir="${BASE_OUTPUT_DIR}/${run_dir_name}/analysis_output"
        local run_log_dir="${BASE_OUTPUT_DIR}/${run_dir_name}/downloaded_logs"

        mkdir -p "$run_output_dir"
        mkdir -p "$run_log_dir"

        # Move/copy logs to the run-specific directory
        log_info "Organizing logs into: ${BASE_OUTPUT_DIR}/${run_dir_name}/"
        cp -r "$job_dir"/* "$run_log_dir/" 2>/dev/null || true

        # Set OUTPUT_DIR and LOG_DIR for this run
        OUTPUT_DIR="$run_output_dir"
        LOG_DIR="$run_log_dir"

        # Prepare analysis context
        local context_dir
        context_dir=$(prepare_analysis_context "$run_log_dir")

        # Create full prompt
        local prompt_file
        prompt_file=$(create_analysis_prompt "$context_dir")

        # Run analysis
        local output_file="${context_dir}/analysis_result.md"
        run_copilot_analysis "$prompt_file" "$output_file" "$copilot_cmd"

        analysis_results+=("$output_file")
    done

    # Clean up temp directory if it exists and we're not keeping output
    if [[ "$keep_output" == false && -d "$temp_log_dir" && "$temp_log_dir" == "${BASE_OUTPUT_DIR}/.temp_downloads" ]]; then
        rm -rf "$temp_log_dir"
    fi

    # Create combined analysis if multiple jobs
    if [[ ${#analysis_results[@]} -gt 1 ]]; then
        log_info "Creating combined analysis for ${#analysis_results[@]} jobs..."
        # Use the first run's output directory for combined analysis
        local first_run_dir=$(dirname "$(dirname "${analysis_results[0]}")")
        local combined_output="${first_run_dir}/combined_analysis.md"
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
    log_info "Results saved to: $BASE_OUTPUT_DIR"
    for result_file in "${analysis_results[@]}"; do
        echo "  - $result_file"
    done
    echo ""
}

main "$@"
