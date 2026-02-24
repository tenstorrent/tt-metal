#!/bin/bash
# Auto-commit script for TTNN agents
# Called by agent Stop hooks to ensure work is committed before handoff
#
# Usage: auto_commit.sh <agent_name>
#
# This script:
# 1. Checks if there are uncommitted changes
# 2. Determines the operation name from modified files
# 3. Creates a commit with proper format

set -e

AGENT_NAME="${1:-unknown-agent}"

# Find repo root (look for .git directory)
find_repo_root() {
    local dir="$PWD"
    while [[ "$dir" != "/" ]]; do
        if [[ -d "$dir/.git" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    echo ""
    return 1
}

REPO_ROOT=$(find_repo_root)
if [[ -z "$REPO_ROOT" ]]; then
    echo "[auto_commit] Not in a git repository, skipping commit"
    exit 0
fi

cd "$REPO_ROOT"

# Check if there are any changes to commit
if git diff --quiet && git diff --cached --quiet; then
    # Check for untracked files in ttnn/cpp/ttnn/operations
    UNTRACKED=$(git status --porcelain | grep "^??" | grep -E "ttnn/cpp/ttnn/operations" || true)
    if [[ -z "$UNTRACKED" ]]; then
        echo "[auto_commit] No changes to commit"
        exit 0
    fi
fi

# Try to determine operation name from modified files
OPERATION_NAME=""

# Look for operation name in modified/staged files
MODIFIED_FILES=$(git status --porcelain | grep -E "ttnn/cpp/ttnn/operations" | head -1 || true)
if [[ -n "$MODIFIED_FILES" ]]; then
    # Extract operation name from path like ttnn/cpp/ttnn/operations/reduction/my_op/...
    OPERATION_NAME=$(echo "$MODIFIED_FILES" | sed -E 's|.*ttnn/cpp/ttnn/operations/[^/]+/([^/]+)/.*|\1|' | head -1)
fi

# Fallback if we couldn't determine operation name
if [[ -z "$OPERATION_NAME" || "$OPERATION_NAME" == "$MODIFIED_FILES" ]]; then
    OPERATION_NAME="unknown"
fi

# Stage all changes
git add -A

# Check again if there's anything staged
if git diff --cached --quiet; then
    echo "[auto_commit] No changes to commit after staging"
    exit 0
fi

# Get summary of changes
CHANGED_FILES=$(git diff --cached --name-only | wc -l)
CHANGED_SUMMARY=$(git diff --cached --stat | tail -1)

# Create commit message
COMMIT_MSG="[$AGENT_NAME] auto-commit before handoff

- $CHANGED_FILES files changed
- $CHANGED_SUMMARY

operation: $OPERATION_NAME
auto_commit: true"

# Commit
git commit -m "$COMMIT_MSG" --no-verify 2>/dev/null || {
    echo "[auto_commit] Commit failed (possibly no changes or hook rejection)"
    exit 0
}

COMMIT_SHA=$(git rev-parse --short HEAD)
echo "[auto_commit] Created commit $COMMIT_SHA for $AGENT_NAME (operation: $OPERATION_NAME)"
