#!/bin/bash
# Worktree setup for codegen agents. Creates a branch from origin/main
#
# Usage (sourced):
#   worktree-name example: issue-123
#   source codegen/scripts/setup_worktree.sh
#   setup_worktree {worktree-name}     # create; exports WORKTREE_DIR / WORKTREE_BRANCH
#   cleanup_worktree {worktree-name}   # remove this run's worktree (default)
#   prune_worktrees 14                 # GC leftover worktrees older than 14d
# Standalone:
#   ./codegen/scripts/setup_worktree.sh {create|cleanup|prune|list} [ARG]
#
# Env:
#   CODEGEN_WORKTREE_ROOT  — worktree parent dir (default: /proj_sw/user_dev/llk-codegen-worktrees)
#   CODEGEN_KEEP_WORKTREE  — "false" (default) removes the worktree after the run;
#                            "true" keeps the live checkout
# Exports: WORKTREE_BRANCH, WORKTREE_DIR

# Strict mode: Avoid strict mode if user sourced the script
[[ "${BASH_SOURCE[0]}" == "$0" ]] && set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# tt-llk root (parent of codegen/)
LLK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# git repo root
REPO_ROOT="$(cd "$LLK_ROOT" && git rev-parse --show-toplevel)"
# Relative path from repo root to tt-llk (needed for worktree paths)
LLK_REL="${LLK_ROOT#"$REPO_ROOT/"}"

GIT_USER="llk_code_gen"

# Prepare paths
CODEGEN_WORKTREE_ROOT="${CODEGEN_WORKTREE_ROOT:-/proj_sw/user_dev/llk-codegen-worktrees}"
CODEGEN_GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-dir)"

CODEGEN_SETUP_LOCK="${CODEGEN_GIT_DIR}/codegen-worktree-setup.lock"

# Remove the worktree after the run (the fix survives as the branch commit +
# generated.patch); "true" keeps it for inspection. `prune` GCs crashed runs.
CODEGEN_KEEP_WORKTREE="${CODEGEN_KEEP_WORKTREE:-false}"

# ── Helpers ──────────────────────────────────────────────────────────────

# Next free branch version for a task (e.g. returns 2 if ...-v1 exists).
# Call with the setup lock held — read-then-create races otherwise.
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

# List every worktree directory this tooling owns (under CODEGEN_WORKTREE_ROOT).
codegen_worktree_dirs() {
  git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null \
    | awk '/^worktree /{print $2}' \
    | grep -F "${CODEGEN_WORKTREE_ROOT}/" || true
}

# ── Main functions ───────────────────────────────────────────────────────

# Create a branch + worktree for the given task, with codegen infra symlinked in.
# Args: task_id — a slug like "issue-123" or "generate-gelu-quasar"
# Sets: WORKTREE_BRANCH, WORKTREE_DIR
setup_worktree() {
  local -; set -euo pipefail
  local task_id="$1"

  mkdir -p "$CODEGEN_WORKTREE_ROOT"
  cd "$REPO_ROOT"
  git fetch origin main --quiet 2>/dev/null || true

  # Reserve a unique branch + dir under a lock (concurrency-safe).
  local lock_fd
  exec {lock_fd}>"$CODEGEN_SETUP_LOCK"
  flock "$lock_fd"

  local version
  version="$(next_branch_version "$task_id")"
  WORKTREE_BRANCH="${GIT_USER}/${task_id}-v${version}"
  # Version in the dir name too, so concurrent same-task runs don't collide.
  WORKTREE_DIR="${CODEGEN_WORKTREE_ROOT}/${task_id}-v${version}"

  echo "[worktree] Creating branch $WORKTREE_BRANCH from origin/main"
  git branch "$WORKTREE_BRANCH" origin/main

  # Clean only THIS exact dir (a crashed prior run of this version).
  if [[ -d "$WORKTREE_DIR" ]] || git -C "$REPO_ROOT" worktree list 2>/dev/null | grep -q "$WORKTREE_DIR"; then
    echo "[worktree] Cleaning up stale worktree at $WORKTREE_DIR"
    git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
    git -C "$REPO_ROOT" worktree prune 2>/dev/null || true
    rm -rf "$WORKTREE_DIR" 2>/dev/null || true
  fi

  echo "[worktree] Creating worktree at $WORKTREE_DIR"
  git worktree add "$WORKTREE_DIR" "$WORKTREE_BRANCH"

  flock -u "$lock_fd"
  exec {lock_fd}>&-

  # ── Symlink codegen infrastructure ──

  # tt-llk worktree path
  local wt_llk="${WORKTREE_DIR}/${LLK_REL}"

  echo "[worktree] Symlinking codegen infrastructure into worktree"
  mkdir -p "${wt_llk}/codegen"

  # Read-only symlink: agent playbooks, references, scripts, config, hooks...
  ln -snf "${LLK_ROOT}/codegen/agents"     "${wt_llk}/codegen/agents"
  ln -snf "${LLK_ROOT}/codegen/references" "${wt_llk}/codegen/references"
  ln -snf "${LLK_ROOT}/codegen/scripts"    "${wt_llk}/codegen/scripts"
  ln -snf "${LLK_ROOT}/codegen/config"     "${wt_llk}/codegen/config"
  ln -snf "${LLK_ROOT}/codegen/hooks"      "${wt_llk}/codegen/hooks"
  ln -snf "${LLK_ROOT}/codegen/skills"     "${wt_llk}/codegen/skills"
  ln -sf  "${LLK_ROOT}/codegen/CLAUDE.md"  "${wt_llk}/codegen/CLAUDE.md"
  ln -sf  "${LLK_ROOT}/codegen/__init__.py" "${wt_llk}/codegen/__init__.py"

  # Writable: artifacts dir is per-worktree (no cross-contamination)
  mkdir -p "${wt_llk}/codegen/artifacts"

  # run_test.sh lives outside codegen/ but is codegen infra: symlink it to the
  # source copy so every worktree runs the current test harness, not the base
  # commit's. Marked --skip-worktree below so git ignores the override.
  mkdir -p "${wt_llk}/.claude/scripts"
  ln -sf "${LLK_ROOT}/.claude/scripts/run_test.sh"  "${wt_llk}/.claude/scripts/run_test.sh"
  ln -sf "${LLK_ROOT}/.claude/scripts/llk_triage.py" "${wt_llk}/.claude/scripts/llk_triage.py"

  # Share the source checkout's Python venv across worktrees instead of rebuilding
  # it every run — apt + pip install of requirements.txt is the slow part, and the
  # deps are arch-independent. Symlink it like the other codegen infra; the venv's
  # activate hardcodes its real path, so activation through the symlink (run_test.sh
  # sources tests/.venv/bin/activate) resolves to the real venv.
  # The tt-metal Docker image has no venv (deps are system-wide) — skip the link there.
  if [[ -d "${LLK_ROOT}/tests/.venv" ]]; then
    echo "[worktree] Linking shared test venv from ${LLK_ROOT}/tests/.venv"
    ln -snf "${LLK_ROOT}/tests/.venv" "${wt_llk}/tests/.venv"
  else
    echo "[worktree] No shared venv at ${LLK_ROOT}/tests/.venv — using ambient python (Docker image)"
  fi

  # Fetch only the arch-specific SFPI toolchain per worktree (setup_testing_env.sh
  # is idempotent: skips if already at the pinned version). test_config resolves it
  # at tests/sfpi/ relative to each worktree, so it cannot be shared.
  echo "[worktree] Fetching SFPI toolchain in worktree"
  (
    cd "${wt_llk}/tests"
    if [[ ! -e /dev/tenstorrent ]]; then
      # No real device on this host (e.g. used for emulation) — export explicitly
      # rather than relying on a prefix assignment surviving `source`.
      export CHIP_ARCH=quasar
    fi
    [[ -f .venv/bin/activate ]] && source .venv/bin/activate
    ./setup_testing_env.sh
  )

  cat >> "${wt_llk}/.gitignore" <<'GITIGNORE'

# Codegen infrastructure (symlinked from feature branch, do not commit)
codegen/
.claude/
CLAUDE.md
.mcp.json
# Shared test venv: **/.venv/** ignores its contents but not the symlink itself
tests/.venv
GITIGNORE

  # .gitignore doesn't hide files already tracked on the base commit (e.g. .mcp.json
  # on origin/main): the symlink shows up as a typechange. Mark such tracked paths
  # --skip-worktree so git ignores the worktree symlink. (.gitignore is included so
  # its own appended lines stay hidden too.)
  for rel in CLAUDE.md .mcp.json .gitignore .claude/scripts/run_test.sh; do
    p="${LLK_REL}/${rel}"
    if git -C "$WORKTREE_DIR" ls-files --error-unmatch -- "$p" >/dev/null 2>&1; then
      git -C "$WORKTREE_DIR" update-index --skip-worktree -- "$p" 2>/dev/null || true
    fi
  done

  echo "[worktree] Ready: $WORKTREE_DIR"
  echo "[worktree] Branch: $WORKTREE_BRANCH"
  echo "[worktree] LLK root in worktree: $wt_llk"

  export WORKTREE_BRANCH
  export WORKTREE_DIR
}

# Remove this run's worktree after the run (default).
cleanup_worktree() {
  local -; set -euo pipefail
  local task_id="$1"
  local delete_branch="${2:-false}"

  cd "$REPO_ROOT"

  if [[ "$CODEGEN_KEEP_WORKTREE" == "true" ]]; then
    echo "[worktree] Keeping worktree for '$task_id' (CODEGEN_KEEP_WORKTREE=true). GC later: $0 prune"
    return 0
  fi

  if [[ -n "${WORKTREE_DIR:-}" ]]; then
    # Normal path: only this run's worktree.
    echo "[worktree] Removing worktree at $WORKTREE_DIR (fix preserved on branch + patch)"
    git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
    [[ -d "$WORKTREE_DIR" ]] && rm -rf "$WORKTREE_DIR"
  else
    # Standalone admin (no WORKTREE_DIR): remove all of this task's worktrees
    # (may hit a concurrent same-task run — prefer `prune` for routine GC).
    echo "[worktree] WORKTREE_DIR unset — removing ALL worktrees for '$task_id' (may affect a concurrent same-task run)."
    local wt
    while IFS= read -r wt; do
      case "$wt" in
        "${CODEGEN_WORKTREE_ROOT}/${task_id}-v"*)
          echo "[worktree] Removing worktree at $wt"
          git worktree remove --force "$wt" 2>/dev/null || true
          [[ -d "$wt" ]] && rm -rf "$wt" ;;
      esac
    done < <(codegen_worktree_dirs)
  fi
  git worktree prune 2>/dev/null || true

  if [[ "$delete_branch" == "true" && -n "${WORKTREE_BRANCH:-}" ]]; then
    echo "[worktree] Deleting branch $WORKTREE_BRANCH"
    git branch -D "$WORKTREE_BRANCH" 2>/dev/null || true
  fi

  echo "[worktree] Cleanup complete for $task_id"
}

# GC worktree dirs older than N days (default 14); branches are left intact.
prune_worktrees() {
  local -; set -euo pipefail
  local days="${1:-14}"
  mkdir -p "$CODEGEN_WORKTREE_ROOT"
  cd "$REPO_ROOT"
  echo "[worktree] Pruning worktrees under $CODEGEN_WORKTREE_ROOT older than ${days}d"
  local wt pruned=0
  while IFS= read -r wt; do
    [[ -z "$wt" ]] && continue
    case "$wt" in "${CODEGEN_WORKTREE_ROOT}/"*) ;; *) continue ;; esac
    # -newermt returns the dir if it is NEWER than the cutoff; empty ⇒ old.
    if [[ -d "$wt" ]] && [[ -z "$(find "$wt" -maxdepth 0 -newermt "-${days} days" 2>/dev/null)" ]]; then
      echo "[worktree]   removing old worktree $wt"
      git worktree remove --force "$wt" 2>/dev/null || rm -rf "$wt"
      pruned=$((pruned + 1))
    fi
  done < <(codegen_worktree_dirs)
  git worktree prune 2>/dev/null || true
  echo "[worktree] Pruned $pruned worktree(s)"
}

# List codegen worktrees.
list_worktrees() {
  echo "[worktree] Root: $CODEGEN_WORKTREE_ROOT"
  local wt
  while IFS= read -r wt; do
    [[ -z "$wt" ]] && continue
    echo "  $wt"
  done < <(codegen_worktree_dirs)
}

# ── Standalone mode ──────────────────────────────────────────────────────

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cmd="${1:-}"
  arg="${2:-}"

  case "$cmd" in
    create)
      [[ -z "$arg" ]] && { echo "Usage: $0 create TASK_ID" >&2; exit 1; }
      setup_worktree "$arg" ;;
    cleanup)
      [[ -z "$arg" ]] && { echo "Usage: $0 cleanup TASK_ID" >&2; exit 1; }
      cleanup_worktree "$arg" ;;
    prune)
      prune_worktrees "${arg:-14}" ;;
    list)
      list_worktrees ;;
    *)
      echo "Usage: $0 {create|cleanup|prune|list} [ARG]"
      echo ""
      echo "  create TASK_ID   Create a durable worktree + branch"
      echo "  cleanup TASK_ID  Remove this run's worktree (default; kept if"
      echo "                   CODEGEN_KEEP_WORKTREE=true)"
      echo "  prune [DAYS]     GC leftover worktrees older than DAYS (default 14)"
      echo "  list             List codegen worktrees"
      echo ""
      echo "TASK_ID examples:"
      echo "  issue-123                  — for issue solving"
      echo "  generate-gelu-quasar       — for kernel generation"
      echo ""
      echo "Env: CODEGEN_WORKTREE_ROOT (default /proj_sw/user_dev/llk-codegen-worktrees),"
      echo "     CODEGEN_KEEP_WORKTREE (default false)"
      [[ -z "$cmd" ]] && exit 1 || { [[ "$cmd" == "--help" || "$cmd" == "-h" ]] && exit 0 || exit 1; }
      ;;
  esac
fi
