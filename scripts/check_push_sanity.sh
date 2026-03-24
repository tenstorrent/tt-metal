#!/usr/bin/env bash
# check_push_sanity.sh — Pre-push sanity checks for tt-metal
# Detects merge pollution (merge commits absorbed into branch) and large diffs
# that indicate an improper rebase.

set -euo pipefail

REMOTE="${1:-origin}"

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Collect local SHAs being pushed from pre-push stdin (local_ref local_sha remote_ref remote_sha).
# If nothing is read (e.g., script run manually), fall back to checking HEAD.
LOCAL_SHAS=()
while read -r local_ref local_sha remote_ref remote_sha; do
    # Ignore delete refs (all-zero SHA) and empty lines.
    if [ -n "${local_sha:-}" ] && [ "$local_sha" != "0000000000000000000000000000000000000000" ]; then
        LOCAL_SHAS+=("$local_sha")
    fi
done || true

if [ "${#LOCAL_SHAS[@]}" -eq 0 ]; then
    LOCAL_SHAS=("HEAD")
fi

EXIT_CODE=0

run_checks_for_sha() {
    local TARGET_SHA="$1"

    local MERGE_BASE
    MERGE_BASE=$(git merge-base "${REMOTE}/main" "$TARGET_SHA" 2>/dev/null) || {
        echo -e "${YELLOW}[pre-push] Could not determine merge base with ${REMOTE}/main for ${TARGET_SHA} — skipping checks for this ref.${NC}"
        return 0
    }

    # --- Check 1: Merge commits in branch history (HARD BLOCK) ---
    local MERGE_COMMITS
    MERGE_COMMITS=$(git log "${MERGE_BASE}..${TARGET_SHA}" --merges --oneline 2>/dev/null)
    if [ -n "$MERGE_COMMITS" ]; then
        echo -e "${RED}[pre-push] ERROR: Your branch contains merge commits (up to ${TARGET_SHA}):${NC}"
        echo "$MERGE_COMMITS"
        echo ""
        echo -e "${RED}Merge commits are banned in tt-metal (see CONTRIBUTING.md).${NC}"
        echo -e "${RED}Fix with: git rebase ${REMOTE}/main${NC}"
        echo -e "${RED}Push aborted.${NC}"
        EXIT_CODE=1
    fi

    # --- Check 2: Files changed vs main ---
    local FILE_COUNT
    FILE_COUNT=$(git diff --name-only "${MERGE_BASE}..${TARGET_SHA}" | wc -l | tr -d ' ')
    if [ "$FILE_COUNT" -gt 200 ]; then
        echo -e "${RED}[pre-push] WARNING: Your branch changes ${FILE_COUNT} files vs main (up to ${TARGET_SHA}).${NC}"
        echo -e "${RED}This likely indicates merge pollution (accidental 'git merge main' instead of 'git rebase main').${NC}"
        echo -e "${RED}Check with: git log --oneline ${REMOTE}/main..${TARGET_SHA}${NC}"
        echo -e "${RED}If intentional, push with: git push --no-verify${NC}"
        echo ""
    elif [ "$FILE_COUNT" -gt 50 ]; then
        echo -e "${YELLOW}[pre-push] WARNING: Your branch changes ${FILE_COUNT} files vs main (up to ${TARGET_SHA}).${NC}"
        echo -e "${YELLOW}If you ran 'git merge main', consider 'git rebase ${REMOTE}/main' instead.${NC}"
        echo ""
    fi

    # --- Check 3: Commit count ---
    local COMMIT_COUNT
    COMMIT_COUNT=$(git rev-list --count "${MERGE_BASE}..${TARGET_SHA}" 2>/dev/null)
    if [ "$COMMIT_COUNT" -gt 30 ]; then
        echo -e "${YELLOW}[pre-push] WARNING: Your branch has ${COMMIT_COUNT} commits ahead of main (up to ${TARGET_SHA}).${NC}"
        echo -e "${YELLOW}Consider squashing or reviewing before pushing.${NC}"
        echo ""
    fi

    if [ "$EXIT_CODE" -eq 0 ] && [ "$FILE_COUNT" -le 50 ]; then
        echo -e "${GREEN}[pre-push] ✓ Sanity checks passed (${FILE_COUNT} files, ${COMMIT_COUNT} commits) for ${TARGET_SHA}.${NC}"
    fi
}

for sha in "${LOCAL_SHAS[@]}"; do
    run_checks_for_sha "$sha"
done

exit $EXIT_CODE
