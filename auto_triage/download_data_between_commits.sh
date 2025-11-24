#!/bin/bash

# Controller script to orchestrate downloading Copilot metadata between two commits.
# Usage: ./download_data_between_commits.sh <start_commit> <end_commit> [output_file]
# - If commit span <= 30, downloads directly.
# - If commit span is between 31 and 200, instructs caller to run the batch script in chunks.
# - If commit span > 200, fails immediately.

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
MAX_BATCHES=200
BATCH_SIZE=30

if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo "Usage: $0 <start_commit> <end_commit> [output_file]"
    exit 1
fi

START_COMMIT="$1"
END_COMMIT="$2"
OUTPUT_FILE="${3:-auto_triage/data/commit_info.json}"

if ! git rev-parse --verify "$START_COMMIT" >/dev/null 2>&1; then
    echo -e "${RED}Error: Start commit '$START_COMMIT' not found${NC}"
    exit 1
fi

if ! git rev-parse --verify "$END_COMMIT" >/dev/null 2>&1; then
    echo -e "${RED}Error: End commit '$END_COMMIT' not found${NC}"
    exit 1
fi

echo -e "${GREEN}Analyzing commits between${NC}"
echo "  Start: $START_COMMIT"
echo "  End:   $END_COMMIT"
echo ""

COMMITS=$(git log --format="%H" --first-parent "$START_COMMIT".."$END_COMMIT")
if ! echo "$COMMITS" | grep -q "^$END_COMMIT$"; then
    COMMITS="$COMMITS"$'\n'"$END_COMMIT"
fi
COMMITS=$(echo "$COMMITS" | sort -u)
COMMIT_COUNT=$(echo "$COMMITS" | grep -c . || echo "0")

echo "Commits in range: $COMMIT_COUNT"

if [ "$COMMIT_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}No commits found between the provided SHAs.${NC}"
    rm -f "$OUTPUT_FILE"
    echo "[]" > "$OUTPUT_FILE"
    exit 0
fi

if [ "$COMMIT_COUNT" -gt "$MAX_BATCHES" ]; then
    echo -e "${RED}Error: too many commits ($COMMIT_COUNT). cannot download${NC}"
    exit 1
fi

# Prepare output file
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_FILE"
echo "[]" > "$OUTPUT_FILE"

if [ "$COMMIT_COUNT" -le "$BATCH_SIZE" ]; then
    echo -e "${GREEN}Commit window <= ${BATCH_SIZE}; downloading in a single batch.${NC}"
    "$(dirname "$0")/download_data_between_commits_batch.sh" "$START_COMMIT" "$END_COMMIT" 0 "$OUTPUT_FILE"
    exit 0
fi

BATCHES=$(( (COMMIT_COUNT + BATCH_SIZE - 1) / BATCH_SIZE ))
echo -e "${YELLOW}Commit window requires $BATCHES batches (limit per call: $BATCH_SIZE).${NC}"
echo -e "Run ./download_data_between_commits_batch.sh with indices 0 through $((BATCHES - 1)) to build the full dataset."
echo -e "Use the same output file ('$OUTPUT_FILE') for each batch; results will be appended."
echo "BATCH_COUNT=$BATCHES"
exit 2
