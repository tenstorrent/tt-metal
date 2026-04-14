#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Batch LLK kernel generation for Blackhole, driven by GitHub P2 issues.
#
# Fetches open Blackhole P2 issues from tenstorrent/tt-llk, then runs codegen
# for each issue's kernel. Issues are categorized by type (SFPU, math, pack,
# unpack) and can be filtered by label or issue number.
#
# Usage:
#   ./scripts/batch_generate_bh.sh                          # list all BH P2 issues
#   ./scripts/batch_generate_bh.sh --run                    # run all sequentially
#   ./scripts/batch_generate_bh.sh --run --parallel          # run all in parallel
#   ./scripts/batch_generate_bh.sh --run -j 4               # max 4 concurrent
#   ./scripts/batch_generate_bh.sh --issue 1153              # run single issue
#   ./scripts/batch_generate_bh.sh --label LLK --run        # only LLK-labeled issues
#   ./scripts/batch_generate_bh.sh --refresh                 # re-fetch issues from GitHub
#   ./scripts/batch_generate_bh.sh --run --dry-run           # show prompts without running
#   ./scripts/batch_generate_bh.sh --run --model sonnet      # use a different model
#   ./scripts/batch_generate_bh.sh --run --no-review         # skip automated review
#   ./scripts/batch_generate_bh.sh --run --auto-fix          # auto-fix review findings

set -euo pipefail
_ORIG_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODEGEN_DIR="$(dirname "$_ORIG_SCRIPT_DIR")"
ISSUES_JSON="${CODEGEN_DIR}/artifacts/bh_p2_issues.json"
LOG_DIR="/tmp/codegen_bh_logs_$(date +%Y%m%d_%H%M%S)"
GIT_USER="$(whoami)"
REPO_ROOT="$(cd "$CODEGEN_DIR/.." && pwd)"

# Copy scripts to a temp directory so they survive branch switches (git checkout
# removes codegen-bh/ when switching to branches based on main).
SCRIPT_DIR="$(mktemp -d /tmp/codegen_bh_scripts_XXXXXX)"
cp "$_ORIG_SCRIPT_DIR"/*.py "$SCRIPT_DIR/"
trap 'rm -rf "$SCRIPT_DIR"' EXIT
RUNS_BASE="$(cd "$REPO_ROOT/../../llk_code_gen/blackhole_issue_solver" 2>/dev/null && pwd || echo "$REPO_ROOT/../../llk_code_gen/blackhole_issue_solver")"
RUNS_JSONL="${RUNS_BASE}/runs.jsonl"

# CI environment
export CODEGEN_BATCH_ID="${CODEGEN_BATCH_ID:-$(date +%Y-%m-%d)_bh_batch}"
export CODEGEN_MODEL="${CODEGEN_MODEL:-claude-opus-4-6}"

# --- Parse args ---
RUN=false
ISSUE_NUM=""
EXTRA_LABEL=""
REFRESH=false
DRY_RUN=false
PARALLEL=false
MAX_JOBS=0
MODEL="${CODEGEN_MODEL}"
NO_REVIEW=false
AUTO_FIX=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --run)      RUN=true; shift ;;
    --issue)    ISSUE_NUM="$2"; RUN=true; shift 2 ;;
    --label)    EXTRA_LABEL="$2"; shift 2 ;;
    --refresh)  REFRESH=true; shift ;;
    --dry-run)  DRY_RUN=true; shift ;;
    --parallel) PARALLEL=true; shift ;;
    -j)         MAX_JOBS="$2"; PARALLEL=true; shift 2 ;;
    --model)    MODEL="$2"; shift 2 ;;
    --no-review) NO_REVIEW=true; shift ;;
    --auto-fix)  AUTO_FIX=true; shift ;;
    --help|-h)
      echo "Usage: $0 [--run] [--issue NUM] [--label LABEL] [--refresh] [--parallel] [-j N] [--model MODEL] [--no-review] [--auto-fix] [--dry-run]"
      echo ""
      echo "  --run         Run codegen for fetched issues (without this, just lists)"
      echo "  --issue NUM   Run codegen for a single issue number"
      echo "  --label LABEL Filter by additional GitHub label (e.g., LLK, feature)"
      echo "  --refresh     Re-fetch issues from GitHub before listing/running"
      echo "  --parallel    Run issues in parallel"
      echo "  -j N          Max concurrent jobs (default: unlimited)"
      echo "  --model MODEL Claude model to use (default: claude-opus-4-6)"
      echo "  --no-review   Skip the automated code review step"
      echo "  --auto-fix    Auto-fix errors found by the reviewer"
      echo "  --dry-run     Show prompts without running"
      echo ""
      echo "First run will auto-fetch issues from GitHub. Use --refresh to update."
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export CODEGEN_MODEL="$MODEL"

# --- Fetch issues if needed ---
fetch_issues() {
  local extra_args=()
  if [[ -n "$EXTRA_LABEL" ]]; then
    extra_args+=(--extra-label "$EXTRA_LABEL")
  fi
  echo "Fetching Blackhole P2 issues from GitHub..."
  python3 "${SCRIPT_DIR}/fetch_bh_issues.py" -o "$ISSUES_JSON" --summary "${extra_args[@]}"
}

if [[ "$REFRESH" == true ]] || [[ ! -f "$ISSUES_JSON" ]]; then
  fetch_issues
fi

# --- Parse issues from JSON ---
# Extract issue list as tab-separated: "number\ttitle\tlabels\tassignees"
parse_issues() {
  python3 -c "
import json, sys

with open('$ISSUES_JSON') as f:
    data = json.load(f)

issue_num_filter = '$ISSUE_NUM'
extra_label = '$EXTRA_LABEL'

for issue in data['issues']:
    num = issue['number']
    if issue_num_filter and str(num) != issue_num_filter:
        continue
    if extra_label and extra_label not in issue['labels']:
        continue
    title = issue['title'].replace('\t', ' ')
    labels = ','.join(l for l in issue['labels'] if l not in ('blackhole', 'P2'))
    assignees = ','.join(issue['assignees']) if issue['assignees'] else '(none)'
    print(f'{num}\t{title}\t{labels}\t{assignees}')
"
}

mapfile -t ISSUE_LINES < <(parse_issues)

# If --issue was given and not found in the P2+blackhole cache, fetch it directly
if [[ ${#ISSUE_LINES[@]} -eq 0 && -n "$ISSUE_NUM" ]]; then
  echo "Issue #${ISSUE_NUM} not in P2+blackhole cache. Fetching directly from GitHub..."
  _DIRECT_ISSUE="$(gh issue view "$ISSUE_NUM" -R "$REPO" --json number,title,labels,assignees 2>/dev/null)"
  if [[ -n "$_DIRECT_ISSUE" ]]; then
    _DIRECT_LINE="$(python3 -c "
import json, sys
issue = json.loads(sys.argv[1])
num = issue['number']
title = issue['title'].replace('\t', ' ')
labels = ','.join(l['name'] for l in issue.get('labels', []))
assignees = ','.join(a.get('login', '?') for a in issue.get('assignees', [])) or '(none)'
print(f'{num}\t{title}\t{labels}\t{assignees}')
" "$_DIRECT_ISSUE")"
    ISSUE_LINES=("$_DIRECT_LINE")
    echo "  Found: #${ISSUE_NUM}"
  fi
fi

if [[ ${#ISSUE_LINES[@]} -eq 0 ]]; then
  echo "No matching issues found."
  if [[ -n "$ISSUE_NUM" ]]; then
    echo "Issue #${ISSUE_NUM} may not exist or you may not have access."
  fi
  exit 1
fi

# --- List mode (no --run) ---
if [[ "$RUN" == false ]]; then
  echo ""
  echo "=== Blackhole P2 Issues: ${#ISSUE_LINES[@]} issue(s) ==="
  echo "=== Model: $MODEL ==="
  echo ""
  printf "  %-6s %-55s %-15s %s\n" "#" "Title" "Assignee" "Labels"
  printf "  %-6s %-55s %-15s %s\n" "------" "-------" "--------" "------"
  for entry in "${ISSUE_LINES[@]}"; do
    IFS=$'\t' read -r num title labels assignees <<< "$entry"
    title_short="${title:0:53}"
    [[ ${#title} -gt 55 ]] && title_short="${title_short}.."
    assignee_short="${assignees:0:13}"
    [[ ${#assignees} -gt 13 ]] && assignee_short="${assignee_short}.."
    printf "  #%-5s %-55s %-15s %s\n" "$num" "$title_short" "$assignee_short" "$labels"
  done
  echo ""
  echo "Run with: $0 --run                    # all issues sequentially"
  echo "          $0 --run --parallel          # all in parallel"
  echo "          $0 --run -j 4               # max 4 concurrent"
  echo "          $0 --issue 1153              # single issue"
  echo "          $0 --label LLK --run         # filter by label"
  echo "          $0 --refresh                 # re-fetch from GitHub"
  echo "          $0 --run --dry-run           # preview prompts"
  exit 0
fi

# --- Branch management ---
# Find the next available version for a branch: {user}/issue-{num}-codegen-v{N}
next_branch_version() {
  local num="$1"
  local v=1
  cd "$REPO_ROOT"
  while git rev-parse --verify "${GIT_USER}/issue-${num}-codegen-v${v}" &>/dev/null \
     || git rev-parse --verify "origin/${GIT_USER}/issue-${num}-codegen-v${v}" &>/dev/null; do
    v=$((v + 1))
  done
  echo "$v"
}

create_issue_branch() {
  local num="$1"
  local v
  v="$(next_branch_version "$num")"
  local branch="${GIT_USER}/issue-${num}-codegen-v${v}"

  cd "$REPO_ROOT"
  echo "  Creating branch $branch from main"
  git branch "$branch" origin/main

  # Export so log_run can pick it up
  CURRENT_BRANCH="$branch"
}

# Create a worktree for the given issue branch. Sets CURRENT_WORKTREE.
# Copies codegen-bh/ and .claude/ into the worktree (they don't exist on main).
create_issue_worktree() {
  local num="$1"
  local branch="$CURRENT_BRANCH"
  local wt_dir="/tmp/codegen_bh_worktree_${num}"

  cd "$REPO_ROOT"
  if [[ -d "$wt_dir" ]]; then
    git worktree remove --force "$wt_dir" 2>/dev/null || rm -rf "$wt_dir"
  fi
  git worktree add "$wt_dir" "$branch"

  # Copy orchestration infrastructure into worktree (not on main)
  [[ -d "${REPO_ROOT}/codegen-bh" ]] && cp -r "${REPO_ROOT}/codegen-bh" "${wt_dir}/codegen-bh"
  [[ -d "${REPO_ROOT}/.claude" ]] && cp -r "${REPO_ROOT}/.claude" "${wt_dir}/.claude"
  # Hide copied dirs from git: append to .gitignore then mark it assume-unchanged
  # so the .gitignore modification itself is invisible to git status/add/commit.
  printf '\n# batch-runner infrastructure (do not commit)\ncodegen-bh/\n.claude/\n' >> "${wt_dir}/.gitignore"
  git -C "$wt_dir" update-index --assume-unchanged .gitignore

  CURRENT_WORKTREE="$wt_dir"
}

# Remove worktree for the given issue
cleanup_issue_worktree() {
  local num="$1"
  local wt_dir="/tmp/codegen_bh_worktree_${num}"
  cd "$REPO_ROOT"
  git worktree remove --force "$wt_dir" 2>/dev/null || rm -rf "$wt_dir" 2>/dev/null || true
}

restore_branch() {
  local original="$1"
  cd "$REPO_ROOT"
  git checkout "$original" 2>/dev/null || true
}

# --- Prompt template ---
make_prompt() {
  local num="$1" title="$2"
  # Count existing runs for this issue to get the version number
  local existing
  existing=$(ls -d "${RUNS_BASE}/blackhole_issue_${num}_v"* 2>/dev/null | wc -l)
  local version=$((existing + 1))
  local run_log_dir="${RUNS_BASE}/blackhole_issue_${num}_v${version}"
  echo "Investigate and fix Blackhole issue #${num}: ${title}. Work autonomously -- use superpowers skills, do not ask questions. Test your changes thoroughly before committing -- compile, run existing tests, and add new tests if none exist. Commit your changes when done.

IMPORTANT: After all tests finish, write a file at ${run_log_dir}/test_results.json summarizing test results in this exact format:
{\"tests_total\": <total number of tests run>, \"tests_passed\": <number that passed>, \"tests_details\": [{\"test\": \"<test name or file>\", \"passed\": <N>, \"total\": <N>}]}
Include ALL tests you ran (compile tests, HW tests, python tests). If a test compiled and ran successfully, count it as passed. If you ran no tests, write {\"tests_total\": 0, \"tests_passed\": 0, \"tests_details\": []}.

LOG_DIR=${run_log_dir} CODEGEN_BATCH_ID=${CODEGEN_BATCH_ID} CODEGEN_MODEL=${MODEL}"
}

# --- Log run result ---
log_run() {
  local issue_num="$1" title="$2" branch="$3" status="$4" start_time="$5" end_time="$6" \
        tmp_log_dir="$7" eval_json="${8:-}" review_json="${9:-}" repo_root_override="${10:-$REPO_ROOT}"

  local log_args=(
    --issue "$issue_num" --title "$title" --branch "$branch"
    --status "$status" --start "$start_time" --end "$end_time"
    --log-dir "$tmp_log_dir" --model "$MODEL"
    --repo-root "$repo_root_override" --runs-base "$RUNS_BASE"
  )

  [[ -n "$CODEGEN_BATCH_ID" ]] && log_args+=(--batch-id "$CODEGEN_BATCH_ID")
  [[ -f "$ISSUES_JSON" ]] && log_args+=(--issues-json "$ISSUES_JSON")
  [[ -n "$eval_json" && -f "$eval_json" ]] && log_args+=(--evaluation "$eval_json")
  [[ -n "$review_json" && -f "$review_json" ]] && log_args+=(--review "$review_json")

  python3 "${SCRIPT_DIR}/log_run.py" "${log_args[@]}"
}

# --- Run a single issue (used by parallel mode) ---
run_one_issue() {
  local num="$1" title="$2" total="$3"
  local prompt
  prompt="$(make_prompt "$num" "$title")"
  local logfile="${LOG_DIR}/issue_${num}.log"
  local jsonfile="${LOG_DIR}/issue_${num}.json"
  local start_time
  start_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo "[#${num}/${total}] START: ${title} (model: $MODEL)"

  create_issue_branch "$num"
  local branch="$CURRENT_BRANCH"
  echo "  Branch: $branch"

  cd "$CODEGEN_DIR"
  claude -p "$prompt" --dangerously-skip-permissions --effort max --verbose --model "$MODEL" --output-format json > "$jsonfile" 2>"$logfile"
  local exit_code=$?
  local end_time
  end_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  local status="completed"
  if [[ $exit_code -ne 0 ]]; then
    status="crashed"
  fi

  # --- Evaluate results ---
  local eval_json="${LOG_DIR}/eval_${num}.json"
  echo "  Evaluating results..."
  python3 "${SCRIPT_DIR}/evaluate_run.py" --repo-root "$REPO_ROOT" --output "$eval_json" || true

  # --- Review changes (unless --no-review) ---
  local review_json="${LOG_DIR}/review_${num}.json"
  if [[ "$NO_REVIEW" == false ]]; then
    echo "  Reviewing changes..."
    local review_args=(
      --repo-root "$REPO_ROOT" --issue "$num" --title "$title"
      --output "$review_json" --codegen-dir "$CODEGEN_DIR"
    )
    $AUTO_FIX && review_args+=(--auto-fix)
    python3 "${SCRIPT_DIR}/review_changes.py" "${review_args[@]}" || true
  fi

  log_run "$num" "$title" "$branch" "$status" "$start_time" "$end_time" "$LOG_DIR" "$eval_json" "$review_json"

  if [[ $exit_code -ne 0 ]]; then
    echo "[#${num}/${total}] FAILED (exit code $exit_code) -- see $logfile"
    return 1
  else
    echo "[#${num}/${total}] DONE: #${num} (branch: $branch)"
    return 0
  fi
}

# --- Sequential run ---
run_sequential() {
  local total=${#ISSUE_LINES[@]}
  local idx=0

  for entry in "${ISSUE_LINES[@]}"; do
    IFS=$'\t' read -r num title labels assignees <<< "$entry"
    idx=$((idx + 1))

    local prompt
    prompt="$(make_prompt "$num" "$title")"

    echo "[${idx}/${total}] #${num}: ${title}"

    if $DRY_RUN; then
      local v
      v="$(next_branch_version "$num")"
      echo "  Branch: ${GIT_USER}/issue-${num}-codegen-v${v}"
      echo "  Prompt: $prompt"
      echo "  (dry run -- skipping)"
      echo ""
      continue
    fi

    local start_time
    start_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    create_issue_branch "$num"
    local branch="$CURRENT_BRANCH"
    echo "  Branch: $branch"

    create_issue_worktree "$num"
    local wt_dir="$CURRENT_WORKTREE"
    echo "  Worktree: $wt_dir"

    cd "${wt_dir}/codegen-bh"
    claude -p "$prompt" --dangerously-skip-permissions --effort max --verbose --model "$MODEL" --output-format json 2>&1 1>"${LOG_DIR}/issue_${num}.json" | tee "${LOG_DIR}/issue_${num}.log"

    exit_code=${PIPESTATUS[0]}
    local end_time
    end_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    local status="completed"
    if [[ $exit_code -ne 0 ]]; then
      status="crashed"
    fi

    # --- Evaluate results ---
    local eval_json="${LOG_DIR}/eval_${num}.json"
    echo "  Evaluating results..."
    python3 "${SCRIPT_DIR}/evaluate_run.py" --repo-root "$wt_dir" --output "$eval_json" || true

    # --- Review changes (unless --no-review) ---
    local review_json="${LOG_DIR}/review_${num}.json"
    if [[ "$NO_REVIEW" == false ]]; then
      echo "  Reviewing changes..."
      local review_args=(
        --repo-root "$wt_dir" --issue "$num" --title "$title"
        --output "$review_json" --codegen-dir "${wt_dir}/codegen-bh"
      )
      $AUTO_FIX && review_args+=(--auto-fix)
      python3 "${SCRIPT_DIR}/review_changes.py" "${review_args[@]}" || true
    fi

    log_run "$num" "$title" "$branch" "$status" "$start_time" "$end_time" "$LOG_DIR" "$eval_json" "$review_json" "$wt_dir"

    # Clean up worktree before continuing to the next issue
    cleanup_issue_worktree "$num"

    if [[ $exit_code -ne 0 ]]; then
      echo "  FAILED (exit code $exit_code) -- stopping."
      exit 1
    fi

    echo "  Done."
    echo ""
  done
}

# --- Parallel run ---
# NOTE: Parallel mode uses git worktrees so each issue gets its own checkout.
# Each worktree is created under /tmp/codegen_bh_worktree_<issue>/ and cleaned
# up after the run completes (successful or not).
run_parallel() {
  local pids=()
  local nums=()
  local worktrees=()
  local active=0
  local total=${#ISSUE_LINES[@]}

  for entry in "${ISSUE_LINES[@]}"; do
    IFS=$'\t' read -r num title labels assignees <<< "$entry"
    if $DRY_RUN; then
      local v
      v="$(next_branch_version "$num")"
      echo "[#${num}/${total}] Blackhole issue #${num}: ${title}"
      echo "  Branch: ${GIT_USER}/issue-${num}-codegen-v${v}"
      echo "  (dry run -- skipping)"
      continue
    fi

    # Throttle if max jobs reached
    if [[ $MAX_JOBS -gt 0 ]]; then
      while [[ $active -ge $MAX_JOBS ]]; do
        wait -n 2>/dev/null || true
        active=0
        for pid in "${pids[@]}"; do
          if kill -0 "$pid" 2>/dev/null; then
            active=$((active + 1))
          fi
        done
      done
    fi

    # Create versioned branch
    local v
    v="$(next_branch_version "$num")"
    local branch="${GIT_USER}/issue-${num}-codegen-v${v}"
    cd "$REPO_ROOT"
    git branch "$branch" origin/main

    # Create worktree for this issue
    local wt_dir="/tmp/codegen_bh_worktree_${num}"

    # Clean up stale worktree if it exists
    if [[ -d "$wt_dir" ]]; then
      git worktree remove --force "$wt_dir" 2>/dev/null || rm -rf "$wt_dir"
    fi

    git worktree add "$wt_dir" "$branch"
    worktrees+=("$wt_dir")

    # Copy orchestration infrastructure into worktree (not on main)
    [[ -d "${REPO_ROOT}/codegen-bh" ]] && cp -r "${REPO_ROOT}/codegen-bh" "${wt_dir}/codegen-bh"
    [[ -d "${REPO_ROOT}/.claude" ]] && cp -r "${REPO_ROOT}/.claude" "${wt_dir}/.claude"
    # Hide copied dirs from git
    printf '\n# batch-runner infrastructure (do not commit)\ncodegen-bh/\n.claude/\n' >> "${wt_dir}/.gitignore"
    git -C "$wt_dir" update-index --assume-unchanged .gitignore

    # Run in the worktree
    (
      local prompt
      prompt="$(make_prompt "$num" "$title")"
      local logfile="${LOG_DIR}/issue_${num}.log"
      local jsonfile="${LOG_DIR}/issue_${num}.json"
      local start_time
      start_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

      echo "[#${num}/${total}] START: ${title} (model: $MODEL)"
      echo "  Branch: $branch"
      echo "  Worktree: $wt_dir"

      cd "${wt_dir}/codegen-bh"
      claude -p "$prompt" --dangerously-skip-permissions --effort max --verbose --model "$MODEL" --output-format json > "$jsonfile" 2>"$logfile"
      local exit_code=$?
      local end_time
      end_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

      local status="completed"
      [[ $exit_code -ne 0 ]] && status="crashed"

      # --- Evaluate results ---
      local eval_json="${LOG_DIR}/eval_${num}.json"
      echo "  Evaluating results..."
      python3 "${SCRIPT_DIR}/evaluate_run.py" --repo-root "$wt_dir" --output "$eval_json" || true

      # --- Review changes (unless --no-review) ---
      local review_json="${LOG_DIR}/review_${num}.json"
      if [[ "$NO_REVIEW" == false ]]; then
        echo "  Reviewing changes..."
        local review_args=(
          --repo-root "$wt_dir" --issue "$num" --title "$title"
          --output "$review_json" --codegen-dir "${wt_dir}/codegen-bh"
        )
        $AUTO_FIX && review_args+=(--auto-fix)
        python3 "${SCRIPT_DIR}/review_changes.py" "${review_args[@]}" || true
      fi

      log_run "$num" "$title" "$branch" "$status" "$start_time" "$end_time" "$LOG_DIR" "$eval_json" "$review_json" "$wt_dir"

      if [[ $exit_code -ne 0 ]]; then
        echo "[#${num}/${total}] FAILED (exit code $exit_code) -- see $logfile"
        exit 1
      else
        echo "[#${num}/${total}] DONE: #${num} (branch: $branch)"
        exit 0
      fi
    ) &
    pids+=($!)
    nums+=("$num")
    active=$((active + 1))
  done

  if $DRY_RUN; then return; fi

  echo ""
  echo "=== Waiting for ${#pids[@]} parallel job(s) to complete ==="
  echo ""

  local failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      failed=$((failed + 1))
    fi
  done

  # Clean up worktrees
  cd "$REPO_ROOT"
  for wt in "${worktrees[@]}"; do
    git worktree remove --force "$wt" 2>/dev/null || true
  done

  echo ""
  if [[ $failed -gt 0 ]]; then
    echo "=== ${failed} issue(s) FAILED -- check logs in ${LOG_DIR}/ ==="
    return 1
  fi
}

# --- Main ---
mkdir -p "$LOG_DIR"
mkdir -p "$RUNS_BASE"

mode="sequentially"
if $PARALLEL; then mode="in parallel"; fi
echo "=== Will process ${#ISSUE_LINES[@]} Blackhole P2 issue(s) ${mode} ==="
echo "=== Model: $MODEL ==="
if $PARALLEL && [[ $MAX_JOBS -gt 0 ]]; then
  echo "=== Max concurrent jobs: ${MAX_JOBS} ==="
fi
echo ""

if $PARALLEL; then
  run_parallel
else
  run_sequential
fi

echo "=== All ${#ISSUE_LINES[@]} issue(s) complete ==="
echo "=== Logs: ${LOG_DIR}/ ==="
