#!/bin/bash
# worktree-create-hook.sh - WorktreeCreate hook wrapper
#
# Reads JSON input from stdin with { name, cwd, session_id } fields.
# Creates a worktree via worktree-setup.sh and prints the path to stdout.

set -euo pipefail

INPUT=$(cat)
NAME=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('name',''))" 2>/dev/null)

if [[ -z "$NAME" ]]; then
    echo "Error: no 'name' field in hook input" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE_PATH="${REPO_DIR}/.claude/worktrees/${NAME}"

# Create worktree + start background build
exec "${REPO_DIR}/.claude/scripts/worktree-setup.sh" "$WORKTREE_PATH"
