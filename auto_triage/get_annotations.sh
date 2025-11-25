#!/bin/bash

# Fetch check-run annotations for a given job URL (or run ID + optional check-run id)
# Usage: ./get_annotations.sh <job_url> [output_file]

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Missing job URL${NC}"
    echo "Usage: $0 <job_url> [output_file]"
    exit 1
fi

JOB_URL="$1"
RUN_ID=$(echo "$JOB_URL" | sed -n 's#.*/runs/\([0-9]\+\)/job/.*#\1#p')
JOB_ID=$(echo "$JOB_URL" | sed -n 's#.*/job/\([0-9]\+\).*#\1#p')

if [ -z "$RUN_ID" ] || [ -z "$JOB_ID" ]; then
    echo -e "${RED}Error: Unable to parse run/job ID from $JOB_URL${NC}"
    exit 1
fi

OUTPUT_FILE="${2:-auto_triage/logs/job_${JOB_ID}/annotations.json}"
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

OWNER="tenstorrent"
REPO="tt-metal"
ANN_DIR="auto_triage/logs/job_${JOB_ID}"
mkdir -p "$ANN_DIR"
ANN_PATH="$OUTPUT_FILE"

echo -e "${GREEN}Fetching check-run annotations for run ${RUN_ID} (job ${JOB_ID})${NC}"

# List jobs for the workflow run and collect check-run IDs
JOBS_JSON=$(gh api "repos/${OWNER}/${REPO}/actions/runs/${RUN_ID}/jobs?per_page=100" 2>/dev/null || echo '{}')
CHECK_RUN_IDS=($(echo "$JOBS_JSON" | jq -r '.jobs[]?.check_run_url | select(.!=null) | capture("/check-runs/(?<id>[0-9]+)").id' 2>/dev/null || true))

if [ ${#CHECK_RUN_IDS[@]} -eq 0 ]; then
    echo -e "${YELLOW}No check-run IDs found for run ${RUN_ID}.${NC}"
    echo '[]' > "$ANN_PATH"
    exit 0
fi

ALL_ANNOTS='[]'
for CHECK_ID in "${CHECK_RUN_IDS[@]}"; do
    PAGE=1
    while true; do
        RAW=$(gh api "repos/${OWNER}/${REPO}/check-runs/${CHECK_ID}/annotations?per_page=100&page=${PAGE}" 2>/dev/null || echo '[]')
        DATA=$(echo "$RAW" | jq '[.[] | select(((.annotation_level // "") | ascii_downcase) == "failure" or ((.annotation_level // "") | ascii_downcase) == "error")]' 2>/dev/null || echo '[]')
        COUNT=$(echo "$DATA" | jq 'length' 2>/dev/null || echo 0)
        if [ "$COUNT" -eq 0 ]; then
            break
        fi
        ALL_ANNOTS=$(jq -s 'add' <(echo "$ALL_ANNOTS") <(echo "$DATA"))
        if [ "$COUNT" -lt 100 ]; then
            break
        fi
        PAGE=$((PAGE + 1))
    done


done

echo "$ALL_ANNOTS" | jq '.' > "$ANN_PATH"
TOTAL=$(echo "$ALL_ANNOTS" | jq 'length' 2>/dev/null || echo 0)
if [ "$TOTAL" -eq 0 ]; then
    echo -e "${YELLOW}No annotations returned for run ${RUN_ID}.${NC}"
else
    echo -e "${GREEN}Saved ${TOTAL} annotation(s) to ${ANN_PATH}.${NC}"
fi
