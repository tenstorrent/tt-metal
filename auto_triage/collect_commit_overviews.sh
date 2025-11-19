#!/bin/bash
#
# Orchestrate `find_boundaries.sh` and `download_data_between_commits.sh`
# to quickly surface Copilot overview data for the commit range where a
# workflow subjob regressed.
#
# Usage:
#   ./collect_commit_overviews.sh <workflow_name> <subjob_name> [output_file]
#
# Example:
#   ./collect_commit_overviews.sh single-card-demo-tests yolov5x-N150-func \
#     auto_triage/outputs/yolov5x_commit_info.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIND_SCRIPT="${SCRIPT_DIR}/find_boundaries.sh"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_data_between_commits.sh"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <workflow_name> <subjob_name> [output_file]" >&2
    exit 1
fi

WORKFLOW_NAME="$1"
SUBJOB_NAME="$2"
OUTPUT_FILE="${3:-auto_triage_commit_info.json}"

if [ ! -x "$FIND_SCRIPT" ]; then
    echo "Error: ${FIND_SCRIPT} not found or not executable." >&2
    exit 1
fi

if [ ! -x "$DOWNLOAD_SCRIPT" ]; then
    echo "Error: ${DOWNLOAD_SCRIPT} not found or not executable." >&2
    exit 1
fi

cleanup() {
    rm -f "$BOUNDARIES_LOG" "$SUMMARY_JSON"
}

BOUNDARIES_LOG="$(mktemp)"
SUMMARY_JSON="$(mktemp)"
trap cleanup EXIT
export BOUNDARIES_SUMMARY_JSON="$SUMMARY_JSON"

echo "=== Running find_boundaries.sh to locate regression window ==="
"$FIND_SCRIPT" "$WORKFLOW_NAME" "$SUBJOB_NAME" | tee "$BOUNDARIES_LOG"
echo ""

strip_colors() {
    sed -r 's/\x1B\[[0-9;]*[mK]//g'
}

extract_commit_from_log() {
    local marker="$1"
    strip_colors < "$BOUNDARIES_LOG" | awk -v marker="$marker" '
        index($0, marker) {flag=1; next}
        flag && $1 == "Commit:" {print $2; exit}
    '
}

if [ -s "$SUMMARY_JSON" ]; then
    LAST_SUCCESS_COMMIT=$(jq -r '.last_success.commit // ""' "$SUMMARY_JSON")
    FIRST_FAIL_COMMIT=$(jq -r '.first_failure.commit // ""' "$SUMMARY_JSON")
    LAST_SUCCESS_JOB_URL=$(jq -r '.last_success.job_url // .last_success.run_url // ""' "$SUMMARY_JSON")
    FIRST_FAIL_JOB_URL=$(jq -r '.first_failure.job_url // .first_failure.run_url // ""' "$SUMMARY_JSON")
else
    LAST_SUCCESS_COMMIT="$(extract_commit_from_log "LAST SUCCESSFUL RUN")"
    FIRST_FAIL_COMMIT="$(extract_commit_from_log "FIRST FAILING RUN")"
    LAST_SUCCESS_JOB_URL=""
    FIRST_FAIL_JOB_URL=""
fi

if [ -z "$LAST_SUCCESS_COMMIT" ] || [ -z "$FIRST_FAIL_COMMIT" ]; then
    echo "Error: Unable to detect commit boundaries from find_boundaries output." >&2
    exit 1
fi

if [ -n "$LAST_SUCCESS_JOB_URL" ]; then
    echo "Last successful job: $LAST_SUCCESS_JOB_URL"
fi
if [ -n "$FIRST_FAIL_JOB_URL" ]; then
    echo "First failing job:   $FIRST_FAIL_JOB_URL"
fi

echo "Commits identified:"
echo "  Last successful commit: $LAST_SUCCESS_COMMIT"
echo "  First failing commit:   $FIRST_FAIL_COMMIT"
echo ""
echo "=== Downloading Copilot overviews for commits in range ==="

"$DOWNLOAD_SCRIPT" "$LAST_SUCCESS_COMMIT" "$FIRST_FAIL_COMMIT" "$OUTPUT_FILE"

if [ -s "$OUTPUT_FILE" ]; then
    if [ ! -s "$SUMMARY_JSON" ]; then
        jq -n --arg workflow "$WORKFLOW_NAME" --arg subjob "$SUBJOB_NAME" \
            '{workflow: $workflow, subjob: $subjob}' > "$SUMMARY_JSON"
    fi

    tmp_out="$(mktemp)"
    jq -n --slurpfile metadata "$SUMMARY_JSON" --slurpfile commits "$OUTPUT_FILE" \
        '{metadata: ($metadata[0] // {}), commits: $commits[0]}' > "$tmp_out"
    mv "$tmp_out" "$OUTPUT_FILE"
fi

echo ""
echo "Done. Copilot overview JSON saved to: $OUTPUT_FILE"
