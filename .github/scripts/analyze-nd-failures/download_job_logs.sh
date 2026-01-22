#!/bin/bash

# Script to download logs from GitHub Actions job links
# Usage: ./download_job_logs.sh <job_url_1> [job_url_2] ... [job_url_n]
# Or: ./download_job_logs.sh --file <file_with_urls>
#
# Each URL should be in format: https://github.com/tenstorrent/tt-metal/actions/runs/<run_id>/job/<job_id>
# Or: https://github.com/tenstorrent/tt-metal/actions/runs/<run_id>

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/build_downloaded_logs"
REPO="tenstorrent/tt-metal"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Function to extract run_id and job_id from a GitHub Actions URL
extract_job_info() {
    local url=$1
    local run_id
    local job_id
    local attempt_number=1

    # Extract run_id from URL
    # Format: https://github.com/tenstorrent/tt-metal/actions/runs/<run_id>/job/<job_id>
    # Or: https://github.com/tenstorrent/tt-metal/actions/runs/<run_id>
    if [[ $url =~ /actions/runs/([0-9]+) ]]; then
        run_id="${BASH_REMATCH[1]}"
    else
        log_error "Could not extract run_id from URL: $url"
        return 1
    fi

    # Extract job_id if present
    if [[ $url =~ /job/([0-9]+) ]]; then
        job_id="${BASH_REMATCH[1]}"
    fi

    # Extract attempt number if present (e.g., /attempts/2)
    if [[ $url =~ /attempts/([0-9]+) ]]; then
        attempt_number="${BASH_REMATCH[1]}"
    fi

    echo "$run_id|$job_id|$attempt_number"
}

# Function to download logs for a specific job
download_job_logs() {
    local run_id=$1
    local job_id=$2
    local attempt_number=${3:-1}
    local job_dir="${OUTPUT_DIR}/run_${run_id}_attempt_${attempt_number}"

    mkdir -p "${job_dir}/logs"
    mkdir -p "${job_dir}/artifacts"

    log_info "Downloading logs for run_id=$run_id, job_id=$job_id, attempt=$attempt_number"

    # If job_id is provided, download logs for that specific job
    if [[ -n "$job_id" ]]; then
        log_info "Downloading logs for job $job_id"
        gh api "/repos/${REPO}/actions/jobs/${job_id}/logs" > "${job_dir}/logs/${job_id}.log" 2>&1 || {
            log_warn "Failed to download logs for job $job_id, continuing..."
        }

        # Download annotations for failed jobs
        local job_info
        job_info=$(gh api "/repos/${REPO}/actions/jobs/${job_id}" 2>&1 || echo "")
        if echo "$job_info" | jq -e '.conclusion == "failure"' > /dev/null 2>&1; then
            log_info "Downloading annotations for failed job $job_id"
            gh api "/repos/${REPO}/check-runs/${job_id}/annotations" > "${job_dir}/logs/${job_id}_annotations.json" 2>&1 || {
                log_warn "Failed to download annotations for job $job_id"
            }
        fi

        # Download job details
        echo "$job_info" > "${job_dir}/logs/${job_id}_job_info.json" 2>&1 || true
    else
        # No specific job_id, download all jobs for this run
        log_info "No job_id specified, downloading all jobs for run $run_id"

        # Use the existing script if available, otherwise do it manually
        if [[ -f "${SCRIPT_DIR}/../../infra/data_collection/github/download_cicd_logs_and_artifacts.sh" ]]; then
            log_info "Using existing download script"
            bash "${SCRIPT_DIR}/../../infra/data_collection/github/download_cicd_logs_and_artifacts.sh" \
                --workflow-run-id "$run_id" \
                --attempt-number "$attempt_number" \
                --repo "$REPO" || {
                log_warn "Existing script failed, trying manual download"
            }

            # Move downloaded files to our output directory
            if [[ -d "generated/cicd/${run_id}" ]]; then
                mv "generated/cicd/${run_id}"/* "${job_dir}/" 2>/dev/null || true
            fi
        else
            # Manual download using gh CLI
            log_info "Downloading all jobs for run $run_id (attempt $attempt_number)"

            # Get all jobs for this run
            local jobs_json
            jobs_json=$(gh api "/repos/${REPO}/actions/runs/${run_id}/attempts/${attempt_number}/jobs" --paginate 2>&1 || {
                log_warn "Failed to get jobs list, trying without pagination"
                gh api "/repos/${REPO}/actions/runs/${run_id}/attempts/${attempt_number}/jobs" 2>&1 || true
            })

            if [[ -n "$jobs_json" ]]; then
                echo "$jobs_json" > "${job_dir}/workflow_jobs.json"

                # Download logs for each job
                echo "$jobs_json" | jq -r '.jobs[]? | "\(.id)|\(.conclusion)"' | while IFS='|' read -r jid conclusion; do
                    if [[ -n "$jid" ]]; then
                        log_info "Downloading logs for job $jid (conclusion: $conclusion)"
                        gh api "/repos/${REPO}/actions/jobs/${jid}/logs" > "${job_dir}/logs/${jid}.log" 2>&1 || {
                            log_warn "Failed to download logs for job $jid"
                        }

                        if [[ "$conclusion" == "failure" ]]; then
                            log_info "Downloading annotations for failed job $jid"
                            gh api "/repos/${REPO}/check-runs/${jid}/annotations" > "${job_dir}/logs/${jid}_annotations.json" 2>&1 || {
                                log_warn "Failed to download annotations for job $jid"
                            }
                        fi
                    fi
                done
            fi
        fi
    fi

    # Download artifacts if available
    log_info "Downloading artifacts for run $run_id"
    gh run download --repo "$REPO" -D "${job_dir}/artifacts" "$run_id" 2>&1 || {
        log_warn "Failed to download artifacts (this is often normal if no artifacts exist)"
    }

    log_info "Completed download for run_id=$run_id, attempt=$attempt_number"
    echo "${job_dir}"
}

# Main function
main() {
    local urls=()
    local urls_file=""

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
            --help|-h)
                echo "Usage: $0 [OPTIONS] <job_url_1> [job_url_2] ..."
                echo ""
                echo "Options:"
                echo "  --file, -f <file>    Read URLs from file (one per line)"
                echo "  --output-dir, -o <dir>  Output directory (default: ./build_downloaded_logs)"
                echo "  --help, -h           Show this help message"
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

    # Read URLs from file if provided
    if [[ -n "$urls_file" ]]; then
        if [[ ! -f "$urls_file" ]]; then
            log_error "File not found: $urls_file"
            exit 1
        fi
        while IFS= read -r line || [[ -n "$line" ]]; do
            # Skip empty lines and comments
            [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
            urls+=("$line")
        done < "$urls_file"
    fi

    if [[ ${#urls[@]} -eq 0 ]]; then
        log_error "No URLs provided. Use --help for usage information."
        exit 1
    fi

    # Check if gh CLI is available
    if ! command -v gh &> /dev/null; then
        log_error "GitHub CLI (gh) is not installed. Please install it first."
        exit 1
    fi

    # Check if jq is available
    if ! command -v jq &> /dev/null; then
        log_error "jq is not installed. Please install it first (e.g., 'sudo apt-get install jq' or 'brew install jq')."
        exit 1
    fi

    # Check GitHub authentication
    if ! gh auth status &> /dev/null; then
        log_error "GitHub CLI is not authenticated. Please run 'gh auth login' first."
        exit 1
    fi

    log_info "Starting download for ${#urls[@]} job URL(s)"
    log_info "Output directory: ${OUTPUT_DIR}"

    mkdir -p "$OUTPUT_DIR"

    local downloaded_dirs=()

    # Process each URL
    for url in "${urls[@]}"; do
        log_info "Processing URL: $url"

        local job_info
        job_info=$(extract_job_info "$url")

        if [[ $? -ne 0 ]]; then
            log_error "Failed to extract job info from URL: $url"
            continue
        fi

        IFS='|' read -r run_id job_id attempt_number <<< "$job_info"

        log_info "Extracted: run_id=$run_id, job_id=$job_id, attempt=$attempt_number"

        local job_dir
        job_dir=$(download_job_logs "$run_id" "$job_id" "$attempt_number")

        if [[ -n "$job_dir" ]]; then
            downloaded_dirs+=("$job_dir")
        fi
    done

    # Summary
    echo ""
    log_info "=== Download Summary ==="
    log_info "Downloaded logs for ${#downloaded_dirs[@]} job(s):"
    for dir in "${downloaded_dirs[@]}"; do
        echo "  - $dir"
    done
    echo ""
    log_info "All logs are available in: ${OUTPUT_DIR}"
}

main "$@"
