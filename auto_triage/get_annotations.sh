#!/bin/bash

# Fetch GitHub Actions annotations for a specific job and save them to disk.
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
JOB_ID=$(echo "$JOB_URL" | grep -oE '/job/[0-9]+' | grep -oE '[0-9]+' || true)
if [ -z "$JOB_ID" ]; then
    echo -e "${RED}Error: Could not extract job ID from URL: $JOB_URL${NC}"
    exit 1
fi

OUTPUT_FILE="${2:-auto_triage/logs/job_${JOB_ID}/annotations.json}"
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

OWNER="tenstorrent"
REPO="tt-metal"

echo -e "${GREEN}Fetching annotations for job ${JOB_ID}${NC}"
ANNOTATIONS=$(gh api "repos/${OWNER}/${REPO}/actions/jobs/${JOB_ID}/annotations" 2>/dev/null || echo "[]")

if [ -z "$ANNOTATIONS" ]; then
    ANNOTATIONS="[]"
fi

echo "$ANNOTATIONS" | jq '.' > "$OUTPUT_FILE"
COUNT=$(echo "$ANNOTATIONS" | jq 'length' 2>/dev/null || echo 0)

if [ "$COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No annotations returned for job ${JOB_ID}.${NC}"
else
    echo -e "${GREEN}Saved ${COUNT} annotation(s) to ${OUTPUT_FILE}.${NC}"
fi
