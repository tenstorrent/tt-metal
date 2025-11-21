#!/bin/bash

# Script to download Copilot Pull Request Overview comments for commits between two commits
# Usage: ./download_data_between_commits.sh <start_commit> <end_commit> [output_file]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo "Usage: $0 <start_commit> <end_commit> [output_file]"
    echo "Example: $0 90336ff5cbacf818e3a20544e5f66b2088757e75 a253cee23e5362d6aba14b716b97f9fe302d6adc"
    exit 1
fi

START_COMMIT="$1"
END_COMMIT="$2"
OUTPUT_FILE="${3:-auto_triage/data/commit_info.json}"

# Validate commits exist
if ! git rev-parse --verify "$START_COMMIT" >/dev/null 2>&1; then
    echo -e "${RED}Error: Start commit '$START_COMMIT' not found${NC}"
    exit 1
fi

if ! git rev-parse --verify "$END_COMMIT" >/dev/null 2>&1; then
    echo -e "${RED}Error: End commit '$END_COMMIT' not found${NC}"
    exit 1
fi

echo -e "${GREEN}Downloading Copilot PR overview comments between commits${NC}"
echo "Start: $START_COMMIT"
echo "End:   $END_COMMIT"
echo ""

# Get all commits between start and end (including end commit)
echo "Getting commit list..."
COMMITS=$(git log --format="%H" --first-parent "$START_COMMIT".."$END_COMMIT")
# Add the end commit itself if not already included
if ! echo "$COMMITS" | grep -q "^$END_COMMIT$"; then
    COMMITS="$COMMITS"$'\n'"$END_COMMIT"
fi
# Remove duplicates and sort
COMMITS=$(echo "$COMMITS" | sort -u)

COMMIT_COUNT=$(echo "$COMMITS" | grep -c . || echo "0")
echo -e "${GREEN}Found $COMMIT_COUNT commits${NC}"
echo ""

if [ "$COMMIT_COUNT" -gt 100 ]; then
    echo -e "${RED}Error: too many commits. cannot download${NC}"
    exit 1
fi

# Initialize JSON output
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_FILE"
echo "[]" > "$OUTPUT_FILE"

# Process each commit
PROCESSED=0
SKIPPED=0
ERRORS=0

while IFS= read -r commit_sha; do
    [ -z "$commit_sha" ] && continue

    commit_short="${commit_sha:0:8}"
    echo -n "[$((PROCESSED + SKIPPED + ERRORS + 1))/$COMMIT_COUNT] Processing $commit_short... "

    # Get commit message to extract PR number
    commit_msg=$(git log -1 --format="%B" "$commit_sha" 2>/dev/null || echo "")

    # Extract PR number from commit message (format: (#12345))
    pr_number=$(echo "$commit_msg" | grep -oP '\(#\K\d+' | head -1 || echo "")

    if [ -z "$pr_number" ]; then
        echo -e "${YELLOW}No PR found${NC}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo -n "PR #$pr_number... "

    # Get commit details first (we'll need these regardless)
    commit_date=$(git log -1 --format="%ai" "$commit_sha" 2>/dev/null || echo "")
    commit_subject=$(git log -1 --format="%s" "$commit_sha" 2>/dev/null || echo "")

    # Get PR details (we'll need these regardless)
    pr_info=$(gh api "repos/tenstorrent/tt-metal/pulls/$pr_number" 2>/dev/null || echo "{}")

    # Try to get Copilot review
    overview="wasn't found"
    reviews_json=$(gh api "repos/tenstorrent/tt-metal/pulls/$pr_number/reviews" 2>/dev/null || echo "[]")

    if [ "$reviews_json" != "[]" ] && [ -n "$reviews_json" ]; then
        # Find Copilot review
        copilot_review=$(echo "$reviews_json" | jq -r ".[] | select(.user.login == \"copilot-pull-request-reviewer\" or .user.login == \"copilot-pull-request-reviewer[bot]\") | .body" 2>/dev/null || echo "")

        if [ -n "$copilot_review" ]; then
            # Extract Pull Request Overview section
            # The overview is everything from "## Pull Request Overview" until "### Reviewed Changes" or end
            overview=$(echo "$copilot_review" | python3 -c "
import sys
content = sys.stdin.read()
start_marker = '## Pull Request Overview'
end_markers = ['### Reviewed Changes', '## ', '---']

if start_marker in content:
    start_idx = content.find(start_marker)
    # Find the earliest end marker after start
    end_idx = len(content)
    for marker in end_markers:
        marker_idx = content.find(marker, start_idx + len(start_marker))
        if marker_idx != -1 and marker_idx < end_idx:
            end_idx = marker_idx

    overview = content[start_idx:end_idx].strip()
    # Remove the header line
    overview = overview.replace(start_marker, '', 1).strip()
    print(overview)
" 2>/dev/null || echo "")

            # If Python extraction failed, try simpler approach
            if [ -z "$overview" ] || [ "$overview" = "" ]; then
                # Extract everything between "## Pull Request Overview" and "### Reviewed Changes"
                overview=$(echo "$copilot_review" | sed -n '/## Pull Request Overview/,/### Reviewed Changes/p' | sed '$d' | sed '1s/## Pull Request Overview//' | sed 's/^[[:space:]]*//' | head -c 5000 || echo "")
            fi

            # Final fallback: get first 2000 chars after the header
            if [ -z "$overview" ] || [ "$overview" = "" ]; then
                overview=$(echo "$copilot_review" | grep -A 50 "## Pull Request Overview" | tail -n +2 | head -n 30 | head -c 2000 || echo "")
            fi

            if [ -z "$overview" ] || [ "$overview" = "" ]; then
                overview="wasn't found"
            fi
        fi
    fi

    # Set status message
    if [ "$overview" = "wasn't found" ]; then
        echo -e "${YELLOW}No Copilot overview${NC}"
    else
        echo -e "${GREEN}Found overview${NC}"
    fi
    pr_title=$(echo "$pr_info" | jq -r '.title // ""' 2>/dev/null || echo "")
    pr_url=$(echo "$pr_info" | jq -r '.html_url // ""' 2>/dev/null || echo "")
    pr_description=$(echo "$pr_info" | jq -r '.body // ""' 2>/dev/null || echo "")

    # Get PR author
    pr_author=$(echo "$pr_info" | jq -r '.user.login // ""' 2>/dev/null || echo "")

    # Get commit author and co-authors
    commit_author=$(git log -1 --format="%an" "$commit_sha" 2>/dev/null || echo "")
    commit_author_email=$(git log -1 --format="%ae" "$commit_sha" 2>/dev/null || echo "")

    # Extract co-authors from commit message (Co-authored-by: Name <email>)
    # Get just the names, not emails
    co_author_names=$(git log -1 --format="%B" "$commit_sha" 2>/dev/null \
        | awk '/^Co-authored-by:/ { sub(/^Co-authored-by:[[:space:]]*/, ""); sub(/<.*>/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print }' \
        | jq -R -s -c 'split("\n") | map(select(length > 0))' 2>/dev/null)
    if [ -z "$co_author_names" ]; then
        co_author_names="[]"
    fi

    # Combine authors: PR author + commit author + co-authors
    # Filter out bots and empty strings, then get unique values
    authors_array=$(jq -n \
        --arg pr_author "$pr_author" \
        --arg commit_author "$commit_author" \
        --argjson co_authors "$co_author_names" \
        '[$pr_author, $commit_author] + $co_authors | map(select(length > 0 and (. | contains("bot") | not) and (. | contains("[bot]") | not))) | unique' 2>/dev/null || echo "[]")

    # Create JSON entry
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

    # Add to output JSON
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

done <<< "$COMMITS"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Summary:${NC}"
echo -e "  Processed: $PROCESSED"
echo -e "  Skipped:   $SKIPPED"
echo -e "  Errors:    $ERRORS"
echo -e "${GREEN}Output saved to: $OUTPUT_FILE${NC}"
echo -e "${GREEN}========================================${NC}"
