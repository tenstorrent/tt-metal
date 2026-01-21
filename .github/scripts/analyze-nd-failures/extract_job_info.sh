#!/bin/bash

# Helper script to extract job information from GitHub Actions URLs
# Usage: ./extract_job_info.sh <url>

set -eo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <github_actions_url>"
    echo ""
    echo "Examples:"
    echo "  $0 https://github.com/tenstorrent/tt-metal/actions/runs/1234567890/job/9876543210"
    echo "  $0 https://github.com/tenstorrent/tt-metal/actions/runs/1234567890"
    exit 1
fi

url=$1

# Extract run_id
if [[ $url =~ /actions/runs/([0-9]+) ]]; then
    run_id="${BASH_REMATCH[1]}"
    echo "run_id: $run_id"
else
    echo "ERROR: Could not extract run_id from URL"
    exit 1
fi

# Extract job_id if present
if [[ $url =~ /job/([0-9]+) ]]; then
    job_id="${BASH_REMATCH[1]}"
    echo "job_id: $job_id"
else
    echo "job_id: (not specified - will download all jobs in run)"
fi

# Extract attempt number if present
if [[ $url =~ /attempts/([0-9]+) ]]; then
    attempt_number="${BASH_REMATCH[1]}"
    echo "attempt_number: $attempt_number"
else
    echo "attempt_number: 1 (default)"
fi

# Output in parseable format
echo "---"
echo "PARSABLE_FORMAT: run_id=$run_id|job_id=${job_id:-}|attempt_number=${attempt_number:-1}"
