#!/bin/bash
# worktree-setup.sh - Create an isolated git worktree with full C++ build + Python venv
#
# Usage: .claude/scripts/worktree-setup.sh <worktree-path> [--branch <branch>]
#
# For manual use outside of Claude Code. Creates a git worktree, inits
# submodules, builds C++, and creates the Python venv. Blocks until done.
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch)
            BASE_BRANCH="$2"; shift 2 ;;
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
    echo "Usage: $0 <worktree-path> [--branch <branch>]" >&2
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

# --- Build ---
cd "$WORKTREE_PATH"
touch .worktree_building

rc=0
(
    set -e
    echo "=== Build started at $(date) ==="
    echo "Worktree: $WORKTREE_PATH"
    echo ""

    echo "=== Initializing git submodules ==="
    git submodule update --init --recursive 2>&1

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

exit $rc
