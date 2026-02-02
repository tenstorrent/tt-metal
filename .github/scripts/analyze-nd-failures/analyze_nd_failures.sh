#!/bin/bash

# Main script to analyze non-deterministic failures from GitHub Actions jobs
# Usage: ./analyze_nd_failures.sh [OPTIONS] <job_url_1> [job_url_2] ... [job_url_n]
#
# This script:
# 1. Downloads logs from GitHub Actions job URLs
# 2. Prepares analysis context (logs, test files, code context)
# 3. Uses Claude CLI to analyze the failures and suggest fixes

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="${SCRIPT_DIR}/analysis_prompt.md"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_job_logs.sh"
REPO_ROOT="${SCRIPT_DIR}/../../.."
BASE_OUTPUT_DIR="${REPO_ROOT}/build_ND_analysis"

# Non-root user for running Claude CLI (required because --dangerously-skip-permissions doesn't work as root)
CLAUDE_USER="claude-runner"

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

# Function to ensure non-root user exists for Claude CLI
# Claude CLI's --dangerously-skip-permissions doesn't work as root for security reasons
ensure_claude_user() {
    # Skip if not running as root
    if [[ $(id -u) -ne 0 ]]; then
        CLAUDE_USER=$(whoami)
        log_debug "Not running as root, using current user: $CLAUDE_USER"
        return 0
    fi

    # Check if user already exists
    if id "$CLAUDE_USER" &>/dev/null; then
        log_debug "User $CLAUDE_USER already exists"
    else
        log_info "Creating non-root user '$CLAUDE_USER' for Claude CLI..."
        useradd -m -s /bin/bash "$CLAUDE_USER" 2>/dev/null || {
            log_error "Failed to create user $CLAUDE_USER"
            exit 1
        }
        log_info "User '$CLAUDE_USER' created successfully"
    fi

    # Get the claude user's home directory
    local claude_home
    claude_home=$(eval echo "~$CLAUDE_USER")

    # Copy Claude credentials to the non-root user if they exist in root's home
    if [[ -f /root/.claude.json ]]; then
        cp /root/.claude.json "$claude_home/.claude.json" 2>/dev/null || true
        chown "$CLAUDE_USER:$CLAUDE_USER" "$claude_home/.claude.json" 2>/dev/null || true
        chmod 600 "$claude_home/.claude.json" 2>/dev/null || true
        log_debug "Copied Claude credentials to $CLAUDE_USER"
    fi

    # Copy Claude config directory if it exists
    if [[ -d /root/.claude ]]; then
        cp -r /root/.claude "$claude_home/.claude" 2>/dev/null || true
        chown -R "$CLAUDE_USER:$CLAUDE_USER" "$claude_home/.claude" 2>/dev/null || true
        log_debug "Copied Claude config directory to $CLAUDE_USER"
    fi

    # Ensure the user has read access to the repository
    if [[ -d "$REPO_ROOT" ]]; then
        # Make repo readable by claude user (preserve existing permissions)
        chmod -R a+rX "$REPO_ROOT" 2>/dev/null || true
    fi
}

# Function to run command as claude user (handles both root and non-root cases)
run_as_claude_user() {
    local cmd=$1

    if [[ $(id -u) -ne 0 ]]; then
        # Not root, run directly
        eval "$cmd"
    else
        # Running as root, switch to claude user
        # Use runuser if available, otherwise su
        if command -v runuser &>/dev/null; then
            runuser -u "$CLAUDE_USER" -- bash -c "$cmd"
        else
            su - "$CLAUDE_USER" -c "$cmd"
        fi
    fi
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

    # Check Claude CLI
    if ! command -v claude &> /dev/null; then
        missing_tools+=("claude (Claude CLI)")
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

    # Ensure non-root user exists for Claude CLI
    ensure_claude_user

    log_info "All prerequisites met"
}

# Function to validate Claude model selection
validate_claude_model() {
    local model=$1
    case "$model" in
        haiku|sonnet|sonnet-1M|opus)
            return 0
            ;;
        *)
            log_error "Invalid model: $model"
            log_error "Valid models are: haiku, sonnet, sonnet-1M, opus"
            return 1
            ;;
    esac
}

# Function to get Claude model flag
get_claude_model_flag() {
    local model=$1
    # Map user-friendly names to Claude CLI model identifiers
    case "$model" in
        haiku)
            echo "haiku"
            ;;
        sonnet)
            echo "sonnet"
            ;;
        sonnet-1M)
            echo "sonnet[1m]"
            ;;
        opus)
            echo "opus"
            ;;
        *)
            echo "sonnet"  # Default fallback
            ;;
    esac
}

# Function to prepare analysis context
prepare_analysis_context() {
    local log_dir=$1
    local context_dir="${OUTPUT_DIR}/context"

    mkdir -p "$context_dir"

    log_info "Preparing analysis context"

    # Copy logs - Claude will analyze them directly
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

# Function to ensure unique directory name (append number if exists and no_overwrite is true)
ensure_unique_directory_name() {
    local base_name=$1
    local no_overwrite=$2
    local full_path="${BASE_OUTPUT_DIR}/${base_name}"

    # If overwrite is allowed or directory doesn't exist, return base name
    if [[ "$no_overwrite" == "false" ]] || [[ ! -d "$full_path" ]]; then
        echo "$base_name"
        return 0
    fi

    # Directory exists and no_overwrite is true - find next available number
    local counter=1
    local candidate_name="${base_name}_${counter}"
    local candidate_path="${BASE_OUTPUT_DIR}/${candidate_name}"

    while [[ -d "$candidate_path" ]]; do
        counter=$((counter + 1))
        candidate_name="${base_name}_${counter}"
        candidate_path="${BASE_OUTPUT_DIR}/${candidate_name}"
    done

    log_info "Directory '$base_name' already exists, using '$candidate_name' instead"
    echo "$candidate_name"
}

# Function to create analysis prompt with context
create_analysis_prompt() {
    local context_dir=$1
    local output_file="${context_dir}/full_prompt.md"

    # Start with the base prompt
    cat "$PROMPT_FILE" > "$output_file"

    # List available log files for Claude to read
    cat >> "$output_file" <<EOF

---

## Available Log Files

The following GitHub Actions job log files are available in the repository for you to analyze:

EOF

    # List the log files
    for log_file in "${context_dir}"/logs/*.log; do
        [[ ! -f "$log_file" ]] && continue
        local log_name=$(basename "$log_file")
        # Get the relative path from REPO_ROOT
        local relative_path=$(realpath --relative-to="$REPO_ROOT" "$log_file" 2>/dev/null || echo "$log_file")
        cat >> "$output_file" <<EOF
- \`$relative_path\` (Job ID: ${log_name%.log})
EOF
    done

    cat >> "$output_file" <<EOF

---

## Your Task

Read the log files above to understand what tests failed and why. Track exactly what is being done as the test reaches the failure to determine the root instability. Use the codebase to investigate the failure paths.

Produce a complete analysis document following the exact format specified earlier, starting with "## Failure Summary". Include specific file paths, line numbers, and code excerpts from the repository.

Begin your analysis now:

EOF

    echo "$output_file"
}

# Function to run Claude analysis
run_claude_analysis() {
    local prompt_file=$1
    local output_file=$2
    local model=$3

    log_info "Running Claude analysis with model: $model..."
    log_info "Prompt file: $prompt_file"
    log_info "Output file: $output_file"

    # Get the Claude model flag
    local model_flag
    model_flag=$(get_claude_model_flag "$model")

    # Print the full prompt before sending to Claude
    echo ""
    log_info "=== Full Prompt to be sent to Claude ==="
    cat "$prompt_file"
    echo ""
    log_info "=== End of Prompt ==="
    echo ""

    log_info "Note: This may take several minutes. Large prompts can take time to process..."

    # Ensure output directory is writable by claude user
    local output_dir
    output_dir=$(dirname "$output_file")
    if [[ $(id -u) -eq 0 ]]; then
        chown -R "$CLAUDE_USER:$CLAUDE_USER" "$output_dir" 2>/dev/null || true
        chmod -R u+rwX "$output_dir" 2>/dev/null || true
    fi

    local exit_code=0
    local temp_output
    temp_output=$(mktemp)

    # Make temp file writable by claude user
    chmod 666 "$temp_output"

    log_info "Running Claude as user: $CLAUDE_USER"

    # Build and run the Claude command
    # Use -p flag for non-interactive/pipe mode (print and exit)
    # Use --dangerously-skip-permissions to allow Claude to read source files
    if [[ $(id -u) -eq 0 ]]; then
        # Running as root - run claude directly (without --dangerously-skip-permissions, as root doesn't need it)
        if command -v timeout &> /dev/null; then
            log_info "Using timeout (10 minute limit)"
            if ! timeout 600 bash -c "cd '$REPO_ROOT' && cat '$prompt_file' | claude --model '$model_flag' -p" 2>&1 | tee "$temp_output"; then
                exit_code=$?
            fi

            if [[ $exit_code -eq 124 ]]; then
                log_error "Claude analysis timed out after 10 minutes"
                cp "$temp_output" "$output_file" 2>/dev/null || true
                rm -f "$temp_output"
                return 1
            fi
        else
            # No timeout available
            if ! bash -c "cd '$REPO_ROOT' && cat '$prompt_file' | claude --model '$model_flag' -p" 2>&1 | tee "$temp_output"; then
                exit_code=$?
            fi
        fi
    else
        # Not running as root - run directly
        if command -v timeout &> /dev/null; then
            log_info "Using timeout (10 minute limit)"
            if ! timeout 600 bash -c "cd '$REPO_ROOT' && cat '$prompt_file' | claude --model '$model_flag' -p --dangerously-skip-permissions" 2>&1 | tee "$temp_output"; then
                exit_code=$?
            fi

            if [[ $exit_code -eq 124 ]]; then
                log_error "Claude analysis timed out after 10 minutes"
                cp "$temp_output" "$output_file" 2>/dev/null || true
                rm -f "$temp_output"
                return 1
            fi
        else
            if ! bash -c "cd '$REPO_ROOT' && cat '$prompt_file' | claude --model '$model_flag' -p --dangerously-skip-permissions" 2>&1 | tee "$temp_output"; then
                exit_code=$?
            fi
        fi
    fi

    # Copy output to final location and fix permissions
    cp "$temp_output" "$output_file" 2>/dev/null || true
    rm -f "$temp_output"

    if [[ $(id -u) -eq 0 ]]; then
        chown root:root "$output_file" 2>/dev/null || true
    fi

    # Ensure output is flushed to disk
    sync "$output_file" 2>/dev/null || true

    # Check if we got valid output
    if [[ $exit_code -eq 0 ]] && [[ -s "$output_file" ]]; then
        log_info "Analysis complete"
        return 0
    else
        log_error "Claude analysis failed (exit code: $exit_code)"
        log_info "Check the output file for details: $output_file"
        return 1
    fi
}

# Main function
main() {
    # Record start time for total duration calculation
    local start_time=$(date +%s)

    local urls=()
    local urls_file=""
    local skip_download=false
    local keep_output=false
    local claude_model="sonnet"  # Default model
    local no_overwrite=false
    local create_pr=false
    local pr_base_branch="main"

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
            --model|-m)
                claude_model=$2
                if ! validate_claude_model "$claude_model"; then
                    exit 1
                fi
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
            --no-overwrite)
                no_overwrite=true
                shift
                ;;
            --create-pr)
                create_pr=true
                shift
                ;;
            --pr-base)
                pr_base_branch=$2
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS] <job_url_1> [job_url_2] ... [job_url_n]"
                echo ""
                echo "Options:"
                echo "  --file, -f <file>       Read URLs from file (one per line)"
                echo "  --output-dir, -o <dir>  Base output directory (default: <repo_root>/build_ND_analysis)"
                echo "  --model, -m <model>    Claude model to use: haiku, sonnet, sonnet-1M, opus (default: sonnet)"
                echo "  --skip-download        Skip downloading logs (use existing downloads)"
                echo "  --keep-output          Keep output and downloaded folders (don't clean up)"
                echo "  --no-overwrite         If output folder exists, append number suffix instead of overwriting"
                echo "  --create-pr            After analysis, create a PR with the suggested fixes"
                echo "  --pr-base <branch>     Base branch for PR (default: main)"
                echo "  --help, -h             Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0 https://github.com/tenstorrent/tt-metal/actions/runs/1234567890/job/9876543210"
                echo "  $0 --file urls.txt --model opus"
                echo "  $0 --model sonnet-1M <job_url>"
                echo "  $0 --create-pr --pr-base main <job_url>"
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

    # Validate Claude model
    if ! validate_claude_model "$claude_model"; then
        exit 1
    fi
    log_info "Using Claude model: $claude_model"

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
        # Check if temp_log_dir itself contains downloaded_logs (user pointed to analysis directory directly)
        if [[ -d "$temp_log_dir/downloaded_logs" ]]; then
            job_dirs+=("$temp_log_dir/downloaded_logs")
        else
            # Look for run_* directories (from fresh downloads)
            while IFS= read -r job_dir; do
                [[ -d "$job_dir" ]] && job_dirs+=("$job_dir")
            done < <(find "$temp_log_dir" -type d -name "run_*" -maxdepth 1 2>/dev/null || true)
        fi
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

        # Determine output structure based on whether we're re-analyzing existing logs
        local run_output_dir
        local run_log_dir

        if [[ "$skip_download" == true ]] && [[ -d "$BASE_OUTPUT_DIR/downloaded_logs" ]]; then
            # Re-analyzing existing logs - output to same directory structure
            run_output_dir="${BASE_OUTPUT_DIR}/analysis_output"
            run_log_dir="$job_dir"
            log_info "Re-analyzing existing logs in: $BASE_OUTPUT_DIR"
        else
            # Fresh analysis from downloads - create new run-specific directory
            local base_dir_name
            base_dir_name=$(create_run_directory_name "$job_dir")

            # Ensure unique name if --no-overwrite is set
            local run_dir_name
            run_dir_name=$(ensure_unique_directory_name "$base_dir_name" "$no_overwrite")

            # Create the run-specific directory structure
            run_output_dir="${BASE_OUTPUT_DIR}/${run_dir_name}/analysis_output"
            run_log_dir="${BASE_OUTPUT_DIR}/${run_dir_name}/downloaded_logs"

            # Move/copy logs to the run-specific directory
            log_info "Organizing logs into: ${BASE_OUTPUT_DIR}/${run_dir_name}/"
            cp -r "$job_dir"/* "$run_log_dir/" 2>/dev/null || true
        fi

        mkdir -p "$run_output_dir"
        mkdir -p "$run_log_dir"

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
        run_claude_analysis "$prompt_file" "$output_file" "$claude_model"

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

    # Calculate and display total duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))

    # Summary
    echo ""
    log_info "=== Analysis Complete ==="
    log_info "Total time: ${minutes}m ${seconds}s"
    log_info "Results saved to: $BASE_OUTPUT_DIR"
    for result_file in "${analysis_results[@]}"; do
        echo "  - $result_file"
    done
    echo ""

    # Create PR if requested
    if [[ "$create_pr" == "true" ]]; then
        echo ""
        log_info "=== Creating PR from Analysis ==="

        # Use the first analysis result (or combined if multiple)
        local pr_analysis_file=""
        if [[ ${#analysis_results[@]} -gt 1 ]] && [[ -f "$combined_output" ]]; then
            pr_analysis_file="$combined_output"
            log_info "Using combined analysis for PR"
        elif [[ ${#analysis_results[@]} -gt 0 ]]; then
            pr_analysis_file="${analysis_results[0]}"
            log_info "Using first analysis result for PR"
        else
            log_error "No analysis results available for PR creation"
            exit 1
        fi

        # Call the PR creation script
        local pr_script="${SCRIPT_DIR}/create_pr_from_analysis.sh"
        if [[ ! -f "$pr_script" ]]; then
            log_error "PR creation script not found: $pr_script"
            exit 1
        fi

        log_info "Running PR creation script..."
        bash "$pr_script" --base "$pr_base_branch" --model "$claude_model" "$pr_analysis_file"
    fi
}

main "$@"
