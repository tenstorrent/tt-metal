#!/bin/bash
# worktree-create-hook.sh - WorktreeCreate hook
#
# Reads JSON input from stdin with { name, cwd, session_id } fields.
# Creates a git worktree + inits submodules, then prints the path.
# The C++ build is NOT done here — CLAUDE.md instructs the agent to
# kick it off as a background task once the session starts.

set -euo pipefail

INPUT=$(cat)
NAME=$(echo "$INPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('name',''))" 2>/dev/null)

if [[ -z "$NAME" ]]; then
    echo "Error: no 'name' field in hook input" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE_PATH="${REPO_DIR}/.claude/worktrees/${NAME}"
BASE_BRANCH="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)"
WT_BRANCH="${BASE_BRANCH}-wt-${NAME}"

# Create worktree
git -C "$REPO_DIR" worktree add -b "$WT_BRANCH" "$WORKTREE_PATH" "$BASE_BRANCH" >&2

# Init submodules (uses shared object store, fast)
git -C "$WORKTREE_PATH" submodule update --init --recursive >&2

# Hook contract: print the absolute worktree path to stdout
echo "$WORKTREE_PATH"
