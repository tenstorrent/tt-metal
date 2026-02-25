#!/bin/bash
# worktree-setup.sh - Create an isolated git worktree with full C++ build + Python venv
#
# Usage: .claude/scripts/worktree-setup.sh <worktree-path> [--branch <branch>] [--foreground]
#
# Creates a git worktree, then kicks off build_metal.sh + create_venv.sh.
# By default the build runs in the background; use --foreground to block.
#
# Marker files (created in the worktree root):
#   .worktree_building  - present while build is running
#   .worktree_ready     - created when build + venv complete successfully
#   .worktree_setup.log - full build output

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- Parse arguments ---
WORKTREE_PATH=""
BASE_BRANCH=""
FOREGROUND=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch)
            BASE_BRANCH="$2"; shift 2 ;;
        --foreground)
            FOREGROUND=true; shift ;;
        -*)
            echo "Unknown option: $1" >&2; exit 1 ;;
        *)
            if [[ -z "$WORKTREE_PATH" ]]; then
                WORKTREE_PATH="$1"; shift
            else
                echo "Unexpected argument: $1" >&2; exit 1
            fi
            ;;
    esac
done

if [[ -z "$WORKTREE_PATH" ]]; then
    echo "Usage: $0 <worktree-path> [--branch <branch>] [--foreground]" >&2
    exit 1
fi

# Resolve to absolute path
WORKTREE_PATH="$(cd "$(dirname "$WORKTREE_PATH")" 2>/dev/null && pwd)/$(basename "$WORKTREE_PATH")" \
    || WORKTREE_PATH="$(mkdir -p "$(dirname "$WORKTREE_PATH")" && cd "$(dirname "$WORKTREE_PATH")" && pwd)/$(basename "$WORKTREE_PATH")"

# Default base branch: current branch of main repo
if [[ -z "$BASE_BRANCH" ]]; then
    BASE_BRANCH="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD)"
fi

# Derive worktree branch name from the directory name
WT_NAME="$(basename "$WORKTREE_PATH")"
WT_BRANCH="${BASE_BRANCH}-wt-${WT_NAME}"

# --- Create git worktree ---
echo "Creating worktree at ${WORKTREE_PATH} (branch: ${WT_BRANCH})..."
git -C "$REPO_DIR" worktree add -b "$WT_BRANCH" "$WORKTREE_PATH" "$BASE_BRANCH"

# --- Build function ---
do_build() {
    cd "$WORKTREE_PATH"
    touch .worktree_building

    # Run build in a subshell so set -e doesn't kill the parent on failure
    local rc=0
    (
        set -e
        echo "=== Build started at $(date) ==="
        echo "Worktree: $WORKTREE_PATH"
        echo "Base branch: $BASE_BRANCH"
        echo ""

        echo "=== Running build_metal.sh ==="
        ./build_metal.sh --debug --enable-ccache --cpm-source-cache "${REPO_DIR}/.cpmcache" 2>&1

        echo ""
        echo "=== Running create_venv.sh ==="
        ./create_venv.sh --force 2>&1

        echo ""
        echo "=== Build completed successfully at $(date) ==="
    ) > .worktree_setup.log 2>&1 || rc=$?

    rm -f .worktree_building

    if [[ $rc -eq 0 ]]; then
        touch .worktree_ready
        echo "Worktree ready: $WORKTREE_PATH"
    else
        echo "BUILD FAILED (exit $rc) — see ${WORKTREE_PATH}/.worktree_setup.log" >&2
    fi

    return $rc
}

# --- Kick off build ---
if [[ "$FOREGROUND" == "true" ]]; then
    do_build
else
    do_build &
    disown
    echo "Build running in background (PID $!) — check ${WORKTREE_PATH}/.worktree_setup.log"
fi

# Print the worktree path (hook contract: stdout = path)
echo "$WORKTREE_PATH"
