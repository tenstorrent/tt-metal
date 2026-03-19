#!/bin/bash
# worktree-remove-hook.sh - WorktreeRemove hook wrapper
#
# Reads JSON input from stdin with { worktree_path, ... } fields.
# Removes the git worktree and cleans up the branch.

set -euo pipefail

INPUT=$(cat)
WORKTREE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('worktree_path',''))" 2>/dev/null)

if [[ -n "$WORKTREE_PATH" && -d "$WORKTREE_PATH" ]]; then
    # Kill any lingering build processes in the worktree
    pkill -f "build_metal.*${WORKTREE_PATH}" 2>/dev/null || true

    # Remove the git worktree
    git worktree remove "$WORKTREE_PATH" --force 2>/dev/null || true

    echo "Removed worktree: $WORKTREE_PATH"
else
    echo "Worktree path not found or empty: '$WORKTREE_PATH'" >&2
fi
