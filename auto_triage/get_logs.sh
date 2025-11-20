#!/bin/bash
#
# Download logs for a specific GitHub Actions job URL.
# Usage:
#   ./get_logs.sh <job_url> [output_directory]
#
# Example:
#   ./get_logs.sh \
#     https://github.com/tenstorrent/tt-metal/actions/runs/19475473285/job/55735804849

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

if [[ "$JOB_URL" =~ github\.com/([^/]+)/([^/]+)/actions/runs/([0-9]+)/job/([0-9]+) ]]; then
    OWNER="${BASH_REMATCH[1]}"
    REPO="${BASH_REMATCH[2]}"
    RUN_ID="${BASH_REMATCH[3]}"
    JOB_ID="${BASH_REMATCH[4]}"
else
    echo "Error: Unable to parse job URL. Expected format https://github.com/<owner>/<repo>/actions/runs/<run_id>/job/<job_id>" >&2
    exit 1
fi

DEST_DIR="${OUTPUT_BASE%/}/job_${JOB_ID}"
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

echo "Fetching job metadata..."
JOB_INFO=$(gh api "repos/${OWNER}/${REPO}/actions/jobs/${JOB_ID}")
JOB_NAME=$(echo "$JOB_INFO" | jq -r '.name // ""')
JOB_ATTEMPT=$(echo "$JOB_INFO" | jq -r '.run_attempt // 1')

TMP_ZIP="$(mktemp --suffix=.zip)"
TMP_UNZIP="$(mktemp -d)"
echo "Downloading logs for run ${RUN_ID}..."
gh api "repos/${OWNER}/${REPO}/actions/runs/${RUN_ID}/logs" > "$TMP_ZIP"
unzip -oq "$TMP_ZIP" -d "$TMP_UNZIP"

sanitize() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]'
}

echo "Copying full log archive..."
FULL_DIR="${DEST_DIR}/full"
mkdir -p "$FULL_DIR"
cp -R "$TMP_UNZIP"/. "$FULL_DIR"/

JOB_KEY=$(sanitize "$JOB_NAME")
MATCHED=()
while IFS= read -r file; do
    rel="${file#$TMP_UNZIP/}"
    if [ -n "$JOB_KEY" ] && echo "$(sanitize "$rel")" | grep -q "$JOB_KEY"; then
        MATCHED+=("$rel")
    fi
done < <(find "$TMP_UNZIP" -type f -print)

if [ ${#MATCHED[@]} -eq 0 ]; then
    echo "Warning: could not isolate job-specific logs; rely on 'full' directory."
else
    JOB_DIR="${DEST_DIR}/job_specific"
    for rel in "${MATCHED[@]}"; do
        src="$TMP_UNZIP/$rel"
        dest="$JOB_DIR/$rel"
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"
    done
    echo "Extracted ${#MATCHED[@]} file(s) matching job name into ${JOB_DIR}"
fi

rm -f "$TMP_ZIP"
rm -rf "$TMP_UNZIP"

cat > "${DEST_DIR}/metadata.txt" <<EOF
Job URL: ${JOB_URL}
Repository: ${OWNER}/${REPO}
Run ID: ${RUN_ID}
Run Attempt: ${JOB_ATTEMPT}
Job ID: ${JOB_ID}
Job Name: ${JOB_NAME}
Downloaded: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF

echo "Logs available at: ${DEST_DIR}"
