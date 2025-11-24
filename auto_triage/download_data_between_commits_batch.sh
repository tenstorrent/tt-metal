#!/bin/bash

# Download Copilot PR overview data for a specific batch of commits.
# Usage: ./download_data_between_commits_batch.sh <start_commit> <end_commit> <batch_index> [output_file]
# Each batch processes up to 30 commits. Batches are zero-indexed.

set -euo pipefail

BATCH_SIZE=30
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ $# -lt 3 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo "Usage: $0 <start_commit> <end_commit> <batch_index> [output_file]"
    echo "Example: $0 90336ff5cbacf818e3a20544e5f66b2088757e75 a253cee23e5362d6aba14b716b97f9fe302d6adc 0"
    exit 1
fi

START_COMMIT="$1"
END_COMMIT="$2"
BATCH_INDEX="$3"
OUTPUT_FILE="${4:-auto_triage/data/commit_info.json}"

if ! [[ "$BATCH_INDEX" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: batch_index must be a non-negative integer${NC}"
    exit 1
fi

if ! git rev-parse --verify "$START_COMMIT" >/dev/null 2>&1; then
    echo -e "${RED}Error: Start commit '$START_COMMIT' not found${NC}"
    exit 1
fi

if ! git rev-parse --verify "$END_COMMIT" >/dev/null 2>&1; then
    echo -e "${RED}Error: End commit '$END_COMMIT' not found${NC}"
    exit 1
fi

echo -e "${GREEN}Processing batch $BATCH_INDEX of commits between${NC}"
echo "  Start: $START_COMMIT"
echo "  End:   $END_COMMIT"
echo ""

COMMITS=$(git log --format="%H" --first-parent "$START_COMMIT".."$END_COMMIT")
if ! echo "$COMMITS" | grep -q "^$END_COMMIT$"; then
    COMMITS="$COMMITS"$'\n'"$END_COMMIT"
fi
COMMITS=$(echo "$COMMITS" | sort -u)
mapfile -t COMMIT_ARRAY < <(echo "$COMMITS" | awk 'NF')
TOTAL_COMMITS=${#COMMIT_ARRAY[@]}

if [ "$TOTAL_COMMITS" -eq 0 ]; then
    echo -e "${YELLOW}No commits found between the provided SHAs.${NC}"
    exit 0
fi

START_OFFSET=$((BATCH_INDEX * BATCH_SIZE))
END_OFFSET=$((START_OFFSET + BATCH_SIZE))

if [ "$START_OFFSET" -ge "$TOTAL_COMMITS" ]; then
    echo -e "${RED}Error: batch index $BATCH_INDEX exceeds total commit count ($TOTAL_COMMITS).${NC}"
    exit 1
fi

if [ "$END_OFFSET" -gt "$TOTAL_COMMITS" ]; then
    END_OFFSET="$TOTAL_COMMITS"
fi

SLICE_LEN=$((END_OFFSET - START_OFFSET))
SELECTED_COMMITS=("${COMMIT_ARRAY[@]:START_OFFSET:SLICE_LEN}")
BATCH_COUNT=${#SELECTED_COMMITS[@]}

echo "Total commits in range: $TOTAL_COMMITS"
echo "This batch covers commits $((START_OFFSET + 1)) to $END_OFFSET (count $BATCH_COUNT)."
echo ""

OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "[]" > "$OUTPUT_FILE"
fi

PROCESSED=0
SKIPPED=0
ERRORS=0

for commit_sha in "${SELECTED_COMMITS[@]}"; do
    [ -z "$commit_sha" ] && continue
    commit_short="${commit_sha:0:8}"
    echo -n "[$((PROCESSED + SKIPPED + ERRORS + 1))/$BATCH_COUNT] Processing $commit_short... "

    commit_msg=$(git log -1 --format="%B" "$commit_sha" 2>/dev/null || echo "")
    pr_number=$(echo "$commit_msg" | grep -oP '\(#\K\d+' | head -1 || echo "")

    if [ -z "$pr_number" ]; then
        echo -e "${YELLOW}No PR found${NC}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo -n "PR #$pr_number... "

    commit_date=$(git log -1 --format="%ai" "$commit_sha" 2>/dev/null || echo "")
    commit_subject=$(git log -1 --format="%s" "$commit_sha" 2>/dev/null || echo "")
    pr_info=$(gh api "repos/tenstorrent/tt-metal/pulls/$pr_number" 2>/dev/null || echo "{}")

    overview="wasn't found"
    reviews_json=$(gh api "repos/tenstorrent/tt-metal/pulls/$pr_number/reviews" 2>/dev/null || echo "[]")

    if [ "$reviews_json" != "[]" ] && [ -n "$reviews_json" ]; then
        copilot_review=$(echo "$reviews_json" | jq -r ".[] | select(.user.login == \"copilot-pull-request-reviewer\" or .user.login == \"copilot-pull-request-reviewer[bot]\") | .body" 2>/dev/null || echo "")
        if [ -n "$copilot_review" ]; then
            overview=$(echo "$copilot_review" | python3 -c "import sys
content = sys.stdin.read()
start_marker = '## Pull Request Overview'
end_markers = ['### Reviewed Changes', '## ', '---']
if start_marker in content:
    start_idx = content.find(start_marker)
    end_idx = len(content)
    for marker in end_markers:
        marker_idx = content.find(marker, start_idx + len(start_marker))
        if marker_idx != -1 and marker_idx < end_idx:
            end_idx = marker_idx
    overview = content[start_idx:end_idx].strip()
    overview = overview.replace(start_marker, '', 1).strip()
    print(overview)")
            if [ -z "$overview" ]; then
                overview=$(echo "$copilot_review" | sed -n '/## Pull Request Overview/,/### Reviewed Changes/p' | sed '$d' | sed '1s/## Pull Request Overview//' | sed 's/^[[:space:]]*//' | head -c 5000 || echo "")
            fi
            if [ -z "$overview" ]; then
                overview=$(echo "$copilot_review" | grep -A 50 "## Pull Request Overview" | tail -n +2 | head -n 30 | head -c 2000 || echo "")
            fi
            if [ -z "$overview" ]; then
                overview="wasn't found"
            fi
        fi
    fi

    if [ "$overview" = "wasn't found" ]; then
        echo -e "${YELLOW}No Copilot overview${NC}"
    else
        echo -e "${GREEN}Found overview${NC}"
    fi

    pr_title=$(echo "$pr_info" | jq -r '.title // ""' 2>/dev/null || echo "")
    pr_url=$(echo "$pr_info" | jq -r '.html_url // ""' 2>/dev/null || echo "")
    pr_description=$(echo "$pr_info" | jq -r '.body // ""' 2>/dev/null || echo "")
    pr_author=$(echo "$pr_info" | jq -r '.user.login // ""' 2>/dev/null || echo "")
    commit_author=$(git log -1 --format="%an" "$commit_sha" 2>/dev/null || echo "")
    co_author_names=$(git log -1 --format="%B" "$commit_sha" 2>/dev/null |
        awk '/^Co-authored-by:/ { sub(/^Co-authored-by:[[:space:]]*/, ""); sub(/<.*>/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print }' |
        jq -R -s -c 'split("\n") | map(select(length > 0))' 2>/dev/null)
    if [ -z "$co_author_names" ]; then
        co_author_names="[]"
    fi

    authors_array=$(jq -n --arg pr_author "$pr_author" --arg commit_author "$commit_author" --argjson co_authors "$co_author_names" '[$pr_author, $commit_author] + $co_authors | map(select(length > 0 and (. | contains("bot") | not) and (. | contains("[bot]") | not))) | unique' 2>/dev/null || echo "[]")

    entry=$(jq -n \
        --arg commit "$commit_sha" \
        --arg commit_short "$commit_short" \
        --arg commit_date "$commit_date" \
        --arg commit_subject "$commit_subject" \
        --arg pr_number "$pr_number" \
        --arg pr_title "$pr_title" \
        --arg pr_url "$pr_url" \
        --arg pr_description "$pr_description" \
        --argjson authors "$authors_array" \
        --arg overview "$overview" \
        '{commit: $commit, commit_short: $commit_short, commit_date: $commit_date, commit_subject: $commit_subject, pr_number: $pr_number, pr_title: $pr_title, pr_url: $pr_url, pr_description: $pr_description, authors: $authors, copilot_overview: $overview}' 2>/dev/null || echo "{}")

    if [ "$entry" != "{}" ]; then
        jq ". += [$entry]" "$OUTPUT_FILE" > "${OUTPUT_FILE}.tmp" && mv "${OUTPUT_FILE}.tmp" "$OUTPUT_FILE" 2>/dev/null || {
            echo -e "${RED}Error updating JSON${NC}"
            ERRORS=$((ERRORS + 1))
            continue
        }
        echo -e "${GREEN}✓${NC}"
        PROCESSED=$((PROCESSED + 1))
    else
        echo -e "${YELLOW}Failed to create entry${NC}"
        ERRORS=$((ERRORS + 1))
    fi

done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Batch Summary:${NC}"
echo -e "  Processed: $PROCESSED"
echo -e "  Skipped:   $SKIPPED"
echo -e "  Errors:    $ERRORS"
echo -e "${GREEN}Output saved to: $OUTPUT_FILE${NC}"
echo -e "${GREEN}========================================${NC}"
