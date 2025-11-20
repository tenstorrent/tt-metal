#!/bin/bash
#
# Download logs for a specific GitHub Actions job URL.
# Usage:
#   ./get_logs.sh <job_url> [output_directory]
#
# Example:
#   ./get_logs.sh \
#     https://github.com/tenstorrent/tt-metal/actions/runs/19475473285/job/55735804859 \
#     auto_triage/logs

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <job_url> [output_directory]" >&2
    exit 1
fi

JOB_URL="$1"
OUTPUT_BASE="${2:-auto_triage/logs}"

if ! command -v gh >/dev/null 2>&1; then
    echo "Error: gh CLI is required but not found in PATH." >&2
    exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
    echo "Error: unzip is required to extract the logs." >&2
    exit 1
fi

if [[ "$JOB_URL" =~ github\.com/([^/]+/[^/]+)/actions/runs/([0-9]+)/job/([0-9]+) ]]; then
    REPO="${BASH_REMATCH[1]}"
    RUN_ID="${BASH_REMATCH[2]}"
    JOB_ID="${BASH_REMATCH[3]}"
else
    echo "Error: Unable to parse job URL. Expected format https://github.com/<owner>/<repo>/actions/runs/<run_id>/job/<job_id>" >&2
    exit 1
fi

DEST_DIR="${OUTPUT_BASE%/}/job_${JOB_ID}"
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

TMP_ZIP="$(mktemp --suffix=.zip)"
echo "Downloading logs for job ${JOB_ID} (run ${RUN_ID}) from repo ${REPO}..."
gh api "repos/${REPO}/actions/jobs/${JOB_ID}/logs" --output "$TMP_ZIP" >/dev/null

echo "Extracting logs into ${DEST_DIR}..."
unzip -oq "$TMP_ZIP" -d "$DEST_DIR"
rm -f "$TMP_ZIP"

cat > "${DEST_DIR}/metadata.txt" <<EOF
Job URL: ${JOB_URL}
Repository: ${REPO}
Run ID: ${RUN_ID}
Job ID: ${JOB_ID}
Downloaded: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo "Logs available at: ${DEST_DIR}"
