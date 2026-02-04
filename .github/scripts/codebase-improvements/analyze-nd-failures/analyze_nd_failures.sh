#!/bin/bash

# =============================================================================
# Non-Deterministic Failure Analysis Script
# =============================================================================
#
# Analyzes GitHub Actions job failures using Claude CLI to identify root causes
# and suggest code fixes.
#
# Usage: ./analyze_nd_failures.sh [OPTIONS] <job_url_1> [job_url_2] ...
#
# Key options:
#   --name <name>     Human-readable name for the output folder
#   --model <model>   Claude model: haiku, sonnet, sonnet-1M, opus
#   --create-pr       Automatically create a PR with suggested fixes
#
# Run with --help for full options.
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="${SCRIPT_DIR}/analysis_prompt.md"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_job_logs.sh"
REPO_ROOT="${SCRIPT_DIR}/../../.."
BASE_OUTPUT_DIR="${REPO_ROOT}/build_ND_analysis"  # Fixed location, ignored by git via build_* pattern

# User for running Claude CLI when script runs as root
CLAUDE_USER="claude-runner"

# -----------------------------------------------------------------------------
# Logging utilities
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1" >&2; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1" >&2; }
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $1" >&2; }

# -----------------------------------------------------------------------------
# Root user handling
# -----------------------------------------------------------------------------
# Claude CLI's --dangerously-skip-permissions doesn't work as root, so we
# create a non-root user when running in CI/Docker environments.

ensure_claude_user() {
    if [[ $(id -u) -ne 0 ]]; then
        CLAUDE_USER=$(whoami)
        log_debug "Running as user: $CLAUDE_USER"
        return 0
    fi

    # Running as root - create or use existing non-root user
    if ! id "$CLAUDE_USER" &>/dev/null; then
        log_info "Creating user '$CLAUDE_USER' for Claude CLI..."
        useradd -m -s /bin/bash "$CLAUDE_USER" 2>/dev/null || {
            log_error "Failed to create user $CLAUDE_USER"
            exit 1
        }
    fi

    # Copy Claude credentials from root to the new user
    local claude_home
    claude_home=$(eval echo "~$CLAUDE_USER")

    [[ -f /root/.claude.json ]] && {
        cp /root/.claude.json "$claude_home/.claude.json"
        chown "$CLAUDE_USER:$CLAUDE_USER" "$claude_home/.claude.json"
        chmod 600 "$claude_home/.claude.json"
    } 2>/dev/null || true

    [[ -d /root/.claude ]] && {
        cp -r /root/.claude "$claude_home/.claude"
        chown -R "$CLAUDE_USER:$CLAUDE_USER" "$claude_home/.claude"
    } 2>/dev/null || true

    # Grant read access to repository
    chmod -R a+rX "$REPO_ROOT" 2>/dev/null || true
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

# -----------------------------------------------------------------------------
# Model selection
# -----------------------------------------------------------------------------
VALID_MODELS="haiku sonnet sonnet-1M opus"

validate_claude_model() {
    local model=$1
    case "$model" in
        haiku|sonnet|sonnet-1M|opus) return 0 ;;
        *)
            log_error "Invalid model: $model. Valid models: $VALID_MODELS"
            return 1
            ;;
    esac
}

get_claude_model_flag() {
    local model=$1
    case "$model" in
        sonnet-1M) echo "sonnet[1m]" ;;
        *)         echo "$model" ;;
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

# -----------------------------------------------------------------------------
# Error and job name extraction for folder naming
# -----------------------------------------------------------------------------

# Extracts a short error category from log files for use in folder names.
# Returns lowercase strings like "device-timeout", "init-failure", etc.
extract_primary_error() {
    local log_file=$1
    [[ ! -f "$log_file" ]] && echo "unknown" && return 0

    local error_type=""

    # Check for common ND failure patterns (most specific first)
    if grep -qi "device.*timeout\|timeout.*device\|timed out" "$log_file" 2>/dev/null; then
        error_type="device-timeout"
    elif grep -qi "failed to initialize\|initialization.*fail\|init.*fail" "$log_file" 2>/dev/null; then
        error_type="init-failure"
    elif grep -qi "connection.*mismatch\|missing.*channel\|missing.*port\|discovery.*fail" "$log_file" 2>/dev/null; then
        error_type="connection-issue"
    elif grep -qi "hardware.*error\|chip.*error\|device.*error" "$log_file" 2>/dev/null; then
        error_type="hardware-error"
    elif grep -qi "out of memory\|resource.*exhaust\|handle.*exhaust" "$log_file" 2>/dev/null; then
        error_type="resource-exhaustion"
    elif grep -qi "segmentation fault\|segfault\|sigsegv" "$log_file" 2>/dev/null; then
        error_type="segfault"
    elif grep -qi "assertion.*fail\|assert.*fail" "$log_file" 2>/dev/null; then
        error_type="assertion-failure"
    else
        error_type="unknown"
    fi

    echo "$error_type"
}

# Extracts a human-readable job name from job metadata.
# Produces names like "demo-tests-vit" or "unit-tests-wormhole".
extract_job_name() {
    local job_dir=$1
    local job_name=""

    # Try to get job name from job_info.json
    local job_info_file
    job_info_file=$(find "$job_dir" -name "*_job_info.json" -type f 2>/dev/null | head -1)

    if [[ -f "$job_info_file" ]]; then
        job_name=$(jq -r '.name // empty' "$job_info_file" 2>/dev/null | head -1)
        [[ "$job_name" == "null" ]] && job_name=""
    fi

    # Fallback: try workflow_jobs.json
    if [[ -z "$job_name" ]]; then
        local workflow_file
        workflow_file=$(find "$job_dir" -name "workflow_jobs.json" -type f 2>/dev/null | head -1)
        [[ -f "$workflow_file" ]] && job_name=$(jq -r '.jobs[0].name // empty' "$workflow_file" 2>/dev/null)
        [[ "$job_name" == "null" ]] && job_name=""
    fi

    # Fallback: extract from directory name
    if [[ -z "$job_name" ]]; then
        job_name=$(basename "$job_dir" | sed 's/run_//' | sed 's/_attempt.*//')
    fi

    # Clean up the name: convert to lowercase, replace non-alphanumeric with hyphens
    job_name=$(echo "$job_name" | tr '[:upper:]' '[:lower:]')
    job_name=$(echo "$job_name" | sed 's/[^[:alnum:]]/-/g')  # Replace special chars with hyphens
    job_name=$(echo "$job_name" | sed 's/--*/-/g')           # Collapse multiple hyphens
    job_name=$(echo "$job_name" | sed 's/^-\|-$//g')         # Trim leading/trailing hyphens
    job_name=$(echo "$job_name" | cut -c1-50)                # Limit length

    echo "${job_name:-unknown-job}"
}

# Auto-generates a directory name from job metadata.
# Format: <job-name>--<error-type> (e.g., "demo-tests-vit--device-timeout")
create_run_directory_name() {
    local job_dir=$1

    local job_name
    job_name=$(extract_job_name "$job_dir")

    local error_type="unknown"
    local log_file
    log_file=$(find "$job_dir" -name "*.log" -type f 2>/dev/null | head -1)
    [[ -n "$log_file" ]] && error_type=$(extract_primary_error "$log_file")

    echo "${job_name}--${error_type}"
}

# Ensures directory name is unique by appending -1, -2, etc. if needed.
ensure_unique_directory_name() {
    local base_name=$1
    local no_overwrite=$2
    local full_path="${BASE_OUTPUT_DIR}/${base_name}"

    # Return base name if overwrite allowed or directory doesn't exist
    if [[ "$no_overwrite" == "false" ]] || [[ ! -d "$full_path" ]]; then
        echo "$base_name"
        return 0
    fi

    # Find next available number suffix
    local counter=1
    while [[ -d "${BASE_OUTPUT_DIR}/${base_name}-${counter}" ]]; do
        counter=$((counter + 1))
    done

    log_info "Directory exists, using '${base_name}-${counter}' instead"
    echo "${base_name}-${counter}"
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

# -----------------------------------------------------------------------------
# Claude analysis execution
# -----------------------------------------------------------------------------
ANALYSIS_TIMEOUT=600  # 10 minutes

run_claude_analysis() {
    local prompt_file=$1
    local output_file=$2
    local model=$3

    log_info "Running Claude analysis (model: $model)..."

    local model_flag
    model_flag=$(get_claude_model_flag "$model")

    # Show prompt being sent
    echo ""
    log_info "=== Prompt sent to Claude ==="
    cat "$prompt_file"
    log_info "=== End of Prompt ==="
    echo ""

    log_info "This may take several minutes..."

    # Ensure output directory is writable
    local output_dir
    output_dir=$(dirname "$output_file")
    chmod -R a+rwX "$output_dir" 2>/dev/null || true

    local temp_output
    temp_output=$(mktemp)
    chmod 666 "$temp_output"

    # Build Claude command - add --dangerously-skip-permissions only for non-root
    local claude_cmd="cd '$REPO_ROOT' && cat '$prompt_file' | claude --model '$model_flag' -p"
    [[ $(id -u) -ne 0 ]] && claude_cmd+=" --dangerously-skip-permissions"

    # Execute with optional timeout
    local exit_code=0
    if command -v timeout &>/dev/null; then
        log_info "Timeout: ${ANALYSIS_TIMEOUT}s"
        timeout "$ANALYSIS_TIMEOUT" bash -c "$claude_cmd" 2>&1 | tee "$temp_output" || exit_code=$?
    else
        bash -c "$claude_cmd" 2>&1 | tee "$temp_output" || exit_code=$?
    fi

    # Save output
    cp "$temp_output" "$output_file" 2>/dev/null || true
    rm -f "$temp_output"
    sync "$output_file" 2>/dev/null || true

    # Check result
    if [[ $exit_code -eq 124 ]]; then
        log_error "Analysis timed out after $((ANALYSIS_TIMEOUT / 60)) minutes"
        return 1
    elif [[ $exit_code -ne 0 ]] || [[ ! -s "$output_file" ]]; then
        log_error "Analysis failed (exit code: $exit_code)"
        return 1
    fi

    log_info "Analysis complete: $output_file"
    return 0
}

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
main() {
    local start_time=$(date +%s)

    # Configuration variables
    local urls=()
    local urls_file=""
    local claude_model="sonnet"
    local skip_download=false
    local keep_output=false
    local no_overwrite=false
    local create_pr=false
    local pr_base_branch="main"
    USER_PROVIDED_NAME=""  # Global so create_run_directory_name can access it

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --file|-f)
                urls_file=$2
                shift 2
                ;;
            -o)
                USER_PROVIDED_NAME=$2
                shift 2
                ;;
            --model|-m)
                claude_model=$2
                validate_claude_model "$claude_model" || exit 1
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
                cat << 'EOF'
Usage: analyze_nd_failures.sh [OPTIONS] <job_url> [job_url_2] ...

Analyze non-deterministic failures from GitHub Actions jobs using Claude.

Output is saved to: build_ND_analysis/<name>/

OPTIONS:
  -o <name>               Name for the output subfolder (default: auto-generated)
  --model, -m <model>     Claude model: haiku, sonnet, sonnet-1M, opus (default: sonnet)
  --file, -f <file>       Read job URLs from file (one per line)
  --skip-download         Skip log download, use existing logs
  --keep-output           Don't clean up temporary files
  --no-overwrite          Create numbered folders instead of overwriting
  --create-pr             Create a PR with suggested fixes after analysis
  --pr-base <branch>      Base branch for PR (default: main)
  --help, -h              Show this help

EXAMPLES:
  # Basic analysis (auto-named folder)
  ./analyze_nd_failures.sh https://github.com/tenstorrent/tt-metal/actions/runs/123/job/456

  # With custom folder name
  ./analyze_nd_failures.sh -o "vit-demo-timeout" <job_url>

  # Batch analysis with Opus
  ./analyze_nd_failures.sh --file failed_jobs.txt --model opus

  # Full workflow: analyze and create PR
  ./analyze_nd_failures.sh --create-pr --model opus <job_url>
EOF
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                exit 1
                ;;
            *)
                urls+=("$1")
                shift
                ;;
        esac
    done

    # Create output directory
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

    # Determine output directory name
    local run_dir_name
    if [[ -n "$USER_PROVIDED_NAME" ]]; then
        # User provided a name - use it directly
        run_dir_name=$(echo "$USER_PROVIDED_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^[:alnum:]]/-/g' | sed 's/--*/-/g' | sed 's/^-\|-$//g')
        run_dir_name=$(ensure_unique_directory_name "$run_dir_name" "$no_overwrite")
    else
        # Auto-generate from first job directory
        run_dir_name=$(create_run_directory_name "${job_dirs[0]}")
        run_dir_name=$(ensure_unique_directory_name "$run_dir_name" "$no_overwrite")
    fi

    # Set up output directories
    local run_output_dir="${BASE_OUTPUT_DIR}/${run_dir_name}/analysis_output"
    local run_log_dir="${BASE_OUTPUT_DIR}/${run_dir_name}/downloaded_logs"
    mkdir -p "$run_output_dir"
    mkdir -p "$run_log_dir/logs"

    # Consolidate all logs from all job directories into one place
    log_info "Consolidating logs into: ${BASE_OUTPUT_DIR}/${run_dir_name}/"
    for job_dir in "${job_dirs[@]}"; do
        log_info "  Adding logs from: $(basename "$job_dir")"
        # Copy logs
        if [[ -d "${job_dir}/logs" ]]; then
            cp -r "${job_dir}/logs"/* "$run_log_dir/logs/" 2>/dev/null || true
        fi
        # Copy artifacts
        if [[ -d "${job_dir}/artifacts" ]]; then
            mkdir -p "$run_log_dir/artifacts"
            cp -r "${job_dir}/artifacts"/* "$run_log_dir/artifacts/" 2>/dev/null || true
        fi
        # Copy any JSON metadata files
        find "$job_dir" -maxdepth 1 -name "*.json" -exec cp {} "$run_log_dir/" \; 2>/dev/null || true
    done

    # Set OUTPUT_DIR for analysis
    OUTPUT_DIR="$run_output_dir"

    # Prepare analysis context
    local context_dir
    context_dir=$(prepare_analysis_context "$run_log_dir")

    # Create full prompt
    local prompt_file
    prompt_file=$(create_analysis_prompt "$context_dir")

    # Run single analysis on all consolidated logs
    local output_file="${context_dir}/analysis_result.md"
    run_claude_analysis "$prompt_file" "$output_file" "$claude_model"

    local analysis_results=("$output_file")

    # Clean up temp directory if it exists and we're not keeping output
    if [[ "$keep_output" == false && -d "$temp_log_dir" && "$temp_log_dir" == "${BASE_OUTPUT_DIR}/.temp_downloads" ]]; then
        rm -rf "$temp_log_dir"
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
