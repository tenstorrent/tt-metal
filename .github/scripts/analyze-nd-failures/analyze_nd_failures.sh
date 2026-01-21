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
OUTPUT_DIR="${SCRIPT_DIR}/analysis_output"
LOG_DIR="${SCRIPT_DIR}/downloaded_logs"
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
# Note: We let Copilot analyze the logs directly instead of pre-extracting files
prepare_analysis_context() {
    local job_dir=$1
    local context_dir="${OUTPUT_DIR}/context_$(basename "$job_dir")"

    mkdir -p "$context_dir"

    log_info "Preparing analysis context for $(basename "$job_dir")"

    # Copy logs
    cp -r "${job_dir}/logs" "${context_dir}/" 2>/dev/null || true

    # Find and copy relevant test files
    local test_files=()
    for log_file in "${job_dir}"/logs/*.log; do
        [[ ! -f "$log_file" ]] && continue
        while IFS= read -r test_file; do
            [[ -n "$test_file" ]] && test_files+=("$test_file")
        done < <(extract_test_info "$log_file")
    done

    if [[ ${#test_files[@]} -gt 0 ]]; then
        log_info "Found ${#test_files[@]} relevant test file(s)"
        mkdir -p "${context_dir}/test_files"
        for test_file in "${test_files[@]}"; do
            local dest_file="${context_dir}/test_files/$(basename "$test_file")"
            cp "${REPO_ROOT}/${test_file}" "$dest_file" 2>/dev/null || true
        done
    fi

    # Find and copy relevant source files
    local source_files=()
    for log_file in "${job_dir}"/logs/*.log; do
        [[ ! -f "$log_file" ]] && continue
        while IFS= read -r source_file; do
            [[ -n "$source_file" ]] && source_files+=("$source_file")
        done < <(extract_source_files "$log_file")
    done

    if [[ ${#source_files[@]} -gt 0 ]]; then
        log_info "Found ${#source_files[@]} relevant source file(s)"
        mkdir -p "${context_dir}/source_files"
        for source_file in "${source_files[@]}"; do
            local rel_path="${source_file}"
            local dest_dir="${context_dir}/source_files/$(dirname "$rel_path")"
            mkdir -p "$dest_dir"
            cp "${REPO_ROOT}/${source_file}" "${dest_dir}/$(basename "$source_file")" 2>/dev/null || true
        done
    fi

    # Create summary file
    cat > "${context_dir}/summary.txt" <<EOF
Analysis Context Summary
=======================

Job Directory: $job_dir
Prepared: $(date)

Test Files Found: ${#test_files[@]}
Source Files Found: ${#source_files[@]}

Test Files:
$(printf '%s\n' "${test_files[@]}")

Source Files:
$(printf '%s\n' "${source_files[@]}")
EOF

    echo "$context_dir"
}

# Function to create analysis prompt with context
create_analysis_prompt() {
    local context_dir=$1
    local output_file="${context_dir}/full_prompt.md"

    # Start with the base prompt
    cat "$PROMPT_FILE" > "$output_file"

    # Add context information
    cat >> "$output_file" <<EOF

---

## Job Logs

The following logs are from the failed GitHub Actions job(s):

EOF

    # Include full logs - Copilot will analyze them
    for log_file in "${context_dir}"/logs/*.log; do
        [[ ! -f "$log_file" ]] && continue
        local log_name=$(basename "$log_file")
        cat >> "$output_file" <<EOF

### Log File: $log_name

\`\`\`
$(cat "$log_file" 2>/dev/null || echo "[Could not read log file]")
\`\`\`

EOF
    done

    # Note: Copilot will analyze the logs to identify relevant files
    # We don't need to pre-extract files - Copilot can do that analysis

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

    # Check if timeout command is available
    local has_timeout=false
    if command -v timeout &> /dev/null; then
        has_timeout=true
        log_info "Using timeout command (10 minute limit)"
    else
        log_warn "timeout command not available - commands may hang indefinitely"
        log_warn "Consider installing coreutils (timeout) or use Ctrl+C to interrupt"
    fi

    # Try different copilot CLI interfaces
    # The GitHub Copilot CLI can be used in several ways:
    # 1. github-copilot-cli explain <file>
    # 2. github-copilot-cli chat (interactive)
    # 3. github-copilot-cli what-the-shell <command>
    # 4. Direct pipe to copilot (if configured as alias)

    log_info "Using Copilot CLI: $copilot_cmd"

    # Method 1: Try explain command (most common)
    if $copilot_cmd explain --help &> /dev/null; then
        log_info "Using 'explain' command"
        $copilot_cmd explain "$prompt_file" > "$output_file" 2>&1 && {
            log_info "Analysis complete using 'explain' command"
            return 0
        } || log_warn "'explain' command failed, trying alternatives"
    fi

    # Method 2: Try chat command (if available)
    if $copilot_cmd chat --help &> /dev/null 2>&1; then
        log_info "Using 'chat' command (non-interactive)"
        log_info "Note: This may take several minutes. Large prompts can take time to process..."

        # Create a temporary input file for chat
        local chat_input="${OUTPUT_DIR}/chat_input.txt"
        cat > "$chat_input" <<EOF
Please analyze the following GitHub Actions job failure and provide recommendations for code changes that could prevent similar failures in the future.

$(cat "$prompt_file")
EOF
        # Use timeout to prevent hanging (10 minutes) if available
        if [[ "$has_timeout" == true ]]; then
            if timeout 600 bash -c "echo 'Analyze this failure:' | $copilot_cmd chat < '$chat_input'" > "$output_file" 2>&1; then
                local success=true
            else
                local exit_code=$?
                local success=false
                if [[ $exit_code -eq 124 ]]; then
                    log_error "'chat' command timed out after 10 minutes"
                else
                    log_warn "'chat' command failed (exit code: $exit_code)"
                fi
            fi
        else
            # No timeout - run directly
            if bash -c "echo 'Analyze this failure:' | $copilot_cmd chat < '$chat_input'" > "$output_file" 2>&1; then
                local success=true
            else
                local success=false
                log_warn "'chat' command failed"
            fi
        fi

        if [[ "${success:-false}" == true ]]; then
            # Check if we got actual output
            if [[ -s "$output_file" ]] && ! grep -qi "error\|failed\|not found" "$output_file" 2>/dev/null; then
                log_info "Analysis complete using 'chat' command"
                return 0
            else
                log_warn "'chat' command produced no valid output"
            fi
        fi
    fi

    # Method 3: Try direct pipe (if copilot is configured as a command)
    log_info "Trying direct pipe method"
    log_info "Note: This may take several minutes. Large prompts can take time to process..."

    # Use timeout to prevent hanging (10 minutes) if available
    if [[ "$has_timeout" == true ]]; then
        if timeout 600 cat "$prompt_file" | $copilot_cmd > "$output_file" 2>&1; then
            local success=true
        else
            local exit_code=$?
            local success=false
            if [[ $exit_code -eq 124 ]]; then
                log_error "Direct pipe method timed out after 10 minutes"
            else
                log_warn "Direct pipe failed (exit code: $exit_code)"
            fi
        fi
    else
        # No timeout - run directly (user can Ctrl+C if needed)
        if cat "$prompt_file" | $copilot_cmd > "$output_file" 2>&1; then
            local success=true
        else
            local success=false
            log_warn "Direct pipe failed"
        fi
    fi

    if [[ "${success:-false}" == true ]]; then
        # Check if we got actual output (not just empty or error)
        if [[ -s "$output_file" ]] && ! grep -qi "error\|failed\|not found" "$output_file" 2>/dev/null; then
            log_info "Analysis complete using direct pipe"
            return 0
        else
            log_warn "Direct pipe produced no valid output"
        fi
    fi

    # Method 4: Manual fallback - create a script that user can run
    log_warn "Automatic Copilot CLI execution failed. Creating manual analysis script."
    local manual_script="${OUTPUT_DIR}/manual_analysis.sh"
    cat > "$manual_script" <<EOF
#!/bin/bash
# Manual analysis script
# Run this script manually with your preferred Copilot CLI method

echo "To analyze this failure, use one of these methods:"
echo ""
echo "1. Using github-copilot-cli explain:"
echo "   $copilot_cmd explain '$prompt_file'"
echo ""
echo "2. Using github-copilot-cli chat:"
echo "   cat '$prompt_file' | $copilot_cmd chat"
echo ""
echo "3. Copy the prompt and use Copilot in your IDE:"
echo "   Prompt file: $prompt_file"
echo ""
echo "The prompt file contains all the context needed for analysis."
EOF
    chmod +x "$manual_script"

    # Still create a placeholder output file
    cat > "$output_file" <<EOF
# Analysis Pending

Automatic Copilot CLI execution was not successful.

Please run the manual analysis script:
  ${manual_script}

Or use one of these commands directly:

1. \`$copilot_cmd explain "$prompt_file"\`

2. \`cat "$prompt_file" | $copilot_cmd chat\`

3. Open the prompt file in your IDE and use Copilot there:
   $prompt_file

---

## Prompt File Location
$prompt_file

## Context Files
The analysis context has been prepared in:
$(dirname "$output_file")

EOF

    log_warn "Created manual analysis script: $manual_script"
    log_info "You can run the analysis manually using the instructions above"
    return 0
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
                echo "  --output-dir, -o <dir>  Output directory (default: ./analysis_output)"
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
