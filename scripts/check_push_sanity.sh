#!/usr/bin/env bash
# check_push_sanity.sh — Pre-push sanity checks for tt-metal
# Detects merge pollution (merge commits absorbed into branch) and large diffs
# that indicate an improper rebase.

set -euo pipefail

REMOTE="${1:-origin}"
BRANCH="${2:-HEAD}"

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

MERGE_BASE=$(git merge-base "${REMOTE}/main" HEAD 2>/dev/null) || {
    echo -e "${YELLOW}[pre-push] Could not determine merge base with ${REMOTE}/main — skipping checks.${NC}"
    exit 0
}

EXIT_CODE=0

# --- Check 1: Merge commits in branch history (HARD BLOCK) ---
MERGE_COMMITS=$(git log "${MERGE_BASE}..HEAD" --merges --oneline 2>/dev/null)
if [ -n "$MERGE_COMMITS" ]; then
    echo -e "${RED}[pre-push] ERROR: Your branch contains merge commits:${NC}"
    echo "$MERGE_COMMITS"
    echo ""
    echo -e "${RED}Merge commits are banned in tt-metal (see CONTRIBUTING.md).${NC}"
    echo -e "${RED}Fix with: git rebase origin/main${NC}"
    echo -e "${RED}Push aborted.${NC}"
    EXIT_CODE=1
fi

# --- Check 2: Files changed vs main ---
FILE_COUNT=$(git diff --name-only "${MERGE_BASE}..HEAD" | wc -l | tr -d ' ')
if [ "$FILE_COUNT" -gt 200 ]; then
    echo -e "${RED}[pre-push] WARNING: Your branch changes ${FILE_COUNT} files vs main.${NC}"
    echo -e "${RED}This likely indicates merge pollution (accidental 'git merge main' instead of 'git rebase main').${NC}"
    echo -e "${RED}Check with: git log --oneline ${REMOTE}/main..HEAD${NC}"
    echo -e "${RED}If intentional, push with: git push --no-verify${NC}"
    echo ""
elif [ "$FILE_COUNT" -gt 50 ]; then
    echo -e "${YELLOW}[pre-push] WARNING: Your branch changes ${FILE_COUNT} files vs main.${NC}"
    echo -e "${YELLOW}If you ran 'git merge main', consider 'git rebase main' instead.${NC}"
    echo ""
fi

# --- Check 3: Commit count ---
COMMIT_COUNT=$(git rev-list --count "${MERGE_BASE}..HEAD" 2>/dev/null)
if [ "$COMMIT_COUNT" -gt 30 ]; then
    echo -e "${YELLOW}[pre-push] WARNING: Your branch has ${COMMIT_COUNT} commits ahead of main.${NC}"
    echo -e "${YELLOW}Consider squashing or reviewing before pushing.${NC}"
    echo ""
fi

if [ "$EXIT_CODE" -eq 0 ] && [ "$FILE_COUNT" -le 50 ]; then
    echo -e "${GREEN}[pre-push] ✓ Sanity checks passed (${FILE_COUNT} files, ${COMMIT_COUNT} commits).${NC}"
fi

exit $EXIT_CODE
