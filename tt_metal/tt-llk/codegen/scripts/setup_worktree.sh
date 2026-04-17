#!/bin/bash
# Worktree setup for codegen agents.
#
# Creates a git branch from origin/main, sets up a worktree, and symlinks
# the codegen infrastructure (agents, scripts, references, config) so that
# Claude Code agents running in the worktree can read their playbooks.
#
# Usage (sourced by other scripts or the orchestrator):
#   source codegen/scripts/setup_worktree.sh
#   setup_worktree issue-123             # issue solving
#   setup_worktree generate-gelu-quasar  # kernel generation
#   cleanup_worktree issue-123
#
# Or standalone:
#   ./codegen/scripts/setup_worktree.sh create issue-123
#   ./codegen/scripts/setup_worktree.sh cleanup issue-123
#
# Exports:
#   WORKTREE_BRANCH  — the branch name (e.g., ai-code-gen/issue-123-v1)
#   WORKTREE_DIR     — absolute path to the worktree

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# tt-llk root (parent of codegen/)
LLK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# git repo root
REPO_ROOT="$(cd "$LLK_ROOT" && git rev-parse --show-toplevel)"
# Relative path from repo root to tt-llk (needed for worktree paths)
LLK_REL="${LLK_ROOT#"$REPO_ROOT/"}"

GIT_USER="ai-code-gen"

# ── Helpers ──────────────────────────────────────────────────────────────

# Find the next available branch version for a task identifier.
# e.g., if ai-code-gen/issue-123-v1 exists, returns 2.
next_branch_version() {
  local task_id="$1"
  local pattern="${GIT_USER}/${task_id}-v"
  local max=0

  while IFS= read -r branch; do
    branch="${branch#remotes/origin/}"
    if [[ "$branch" =~ ${pattern}([0-9]+)$ ]]; then
      local v="${BASH_REMATCH[1]}"
      (( v > max )) && max=$v
    fi
  done < <(git -C "$REPO_ROOT" branch -a 2>/dev/null | sed 's/^[ *]*//')

  echo $(( max + 1 ))
}

# ── Main functions ───────────────────────────────────────────────────────

# Create a branch + worktree for the given task, with codegen infra symlinked in.
# Args: task_id — a slug like "issue-123" or "generate-gelu-quasar"
# Sets: WORKTREE_BRANCH, WORKTREE_DIR
setup_worktree() {
  local task_id="$1"

  # ── Create branch from origin/main ──
  local version
  version="$(next_branch_version "$task_id")"
  WORKTREE_BRANCH="${GIT_USER}/${task_id}-v${version}"

  cd "$REPO_ROOT"
  echo "[worktree] Creating branch $WORKTREE_BRANCH from origin/main"
  git fetch origin main --quiet 2>/dev/null || true
  git branch "$WORKTREE_BRANCH" origin/main

  # ── Create worktree ──
  WORKTREE_DIR="/tmp/codegen_worktree_${task_id}"

  if [[ -d "$WORKTREE_DIR" ]]; then
    echo "[worktree] Cleaning up stale worktree at $WORKTREE_DIR"
    git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
  fi

  echo "[worktree] Creating worktree at $WORKTREE_DIR"
  git worktree add "$WORKTREE_DIR" "$WORKTREE_BRANCH"

  # ── Symlink codegen infrastructure ──
  # The codegen/ directory doesn't exist on main — symlink read-only parts
  # from the source branch so agents can read their playbooks.
  local wt_llk="${WORKTREE_DIR}/${LLK_REL}"

  echo "[worktree] Symlinking codegen infrastructure into worktree"
  mkdir -p "${wt_llk}/codegen"

  # Read-only: agent playbooks, references, scripts, config
  # Use -snf so symlinks replace any existing dirs/files from main
  ln -snf "${LLK_ROOT}/codegen/agents"     "${wt_llk}/codegen/agents"
  ln -snf "${LLK_ROOT}/codegen/references" "${wt_llk}/codegen/references"
  ln -snf "${LLK_ROOT}/codegen/scripts"    "${wt_llk}/codegen/scripts"
  ln -snf "${LLK_ROOT}/codegen/config"     "${wt_llk}/codegen/config"
  ln -sf  "${LLK_ROOT}/codegen/CLAUDE.md"  "${wt_llk}/codegen/CLAUDE.md"
  # __init__.py is required so `import codegen.config.settings` (Step -1 env
  # validation, agent_tools imports) works inside the worktree.
  ln -sf  "${LLK_ROOT}/codegen/__init__.py" "${wt_llk}/codegen/__init__.py"

  # Writable: artifacts dir is per-worktree (no cross-contamination)
  mkdir -p "${wt_llk}/codegen/artifacts"

  # Top-level config files (use -sf to overwrite any that exist on main)
  ln -sf "${LLK_ROOT}/CLAUDE.md"    "${wt_llk}/CLAUDE.md"
  ln -sf "${LLK_ROOT}/.mcp.json"   "${wt_llk}/.mcp.json"
  [[ -d "${LLK_ROOT}/.claude" ]] && ln -snf "${LLK_ROOT}/.claude" "${wt_llk}/.claude"

  # ── Hide symlinks from git ──
  # Files that existed on main were replaced with symlinks — git sees a typechange.
  # Mark them assume-unchanged so git status stays clean.
  git -C "$WORKTREE_DIR" update-index --assume-unchanged "${LLK_REL}/CLAUDE.md"

  # Append to .gitignore then mark it assume-unchanged so
  # the .gitignore change itself is invisible to git status/add/commit.
  cat >> "${wt_llk}/.gitignore" <<'GITIGNORE'

# Codegen infrastructure (symlinked from feature branch, do not commit)
codegen/
.claude/
.mcp.json
GITIGNORE
  git -C "$WORKTREE_DIR" update-index --assume-unchanged "${LLK_REL}/.gitignore"

  echo "[worktree] Ready: $WORKTREE_DIR"
  echo "[worktree] Branch: $WORKTREE_BRANCH"
  echo "[worktree] LLK root in worktree: $wt_llk"

  export WORKTREE_BRANCH
  export WORKTREE_DIR
}

# Remove worktree and optionally the branch for the given task.
cleanup_worktree() {
  local task_id="$1"
  local delete_branch="${2:-false}"
  local wt_dir="/tmp/codegen_worktree_${task_id}"

  cd "$REPO_ROOT"

  if git worktree list 2>/dev/null | grep -q "$wt_dir"; then
    echo "[worktree] Removing worktree at $wt_dir"
    git worktree remove --force "$wt_dir" 2>/dev/null || true
  fi

  # Fallback: if worktree remove failed, clean up manually
  if [[ -d "$wt_dir" ]]; then
    echo "[worktree] Force-removing leftover directory $wt_dir"
    rm -rf "$wt_dir"
    git worktree prune 2>/dev/null || true
  fi

  if [[ "$delete_branch" == "true" && -n "${WORKTREE_BRANCH:-}" ]]; then
    echo "[worktree] Deleting branch $WORKTREE_BRANCH"
    git branch -D "$WORKTREE_BRANCH" 2>/dev/null || true
  fi

  echo "[worktree] Cleanup complete for $task_id"
}

# ── Standalone mode ──────────────────────────────────────────────────────

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cmd="${1:-}"
  task_id="${2:-}"

  if [[ -z "$cmd" || -z "$task_id" ]]; then
    echo "Usage: $0 {create|cleanup} TASK_ID"
    echo ""
    echo "TASK_ID examples:"
    echo "  issue-123                  — for issue solving"
    echo "  generate-gelu-quasar       — for kernel generation"
    exit 1
  fi

  case "$cmd" in
    create)  setup_worktree "$task_id" ;;
    cleanup) cleanup_worktree "$task_id" ;;
    *)       echo "Unknown command: $cmd (use create or cleanup)"; exit 1 ;;
  esac
fi
