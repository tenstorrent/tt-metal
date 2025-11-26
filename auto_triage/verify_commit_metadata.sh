#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHOSEN_FILE="${ROOT}/output/chosen_commit.json"
ALTS_FILE="${ROOT}/output/alternatives.json"
OWNER="tenstorrent"
REPO="tt-metal"

fail() {
  local reason="$1"
  echo "detected hallucination: $reason"
  echo "  context_file: $2"
  echo "  entry_index: ${3:-n/a}"
  echo "  commit: ${4:-<none>}"
  exit 1
}

check_entry() {
  local entry_json="$1"
  local context_file="$2"
  local index="$3"

  local commit
  commit=$(echo "$entry_json" | jq -r '.commit // .commit_sha // empty')
  if [ -z "$commit" ]; then
    fail "missing commit SHA" "$context_file" "$index"
  fi

  local expected_url="https://github.com/${OWNER}/${REPO}/commit/${commit}"
  local declared_url
  declared_url=$(echo "$entry_json" | jq -r '.commit_url // empty')
  if [ -n "$declared_url" ] && [ "$declared_url" != "$expected_url" ]; then
    fail "commit_url mismatch" "$context_file" "$index" "$commit"
  fi

  mapfile -t decl_authors < <(echo "$entry_json" | jq -r '.authors[]?.login // empty') || true
  mapfile -t decl_approvers < <(echo "$entry_json" | jq -r '.approvers[]?.login // empty') || true

  local actual_author
  actual_author=$(gh api "repos/${OWNER}/${REPO}/commits/${commit}" --jq '.author.login // empty' 2>/dev/null || echo "")
  if [ -n "$actual_author" ]; then
    if ! printf '%s
' "${decl_authors[@]}" | grep -Fxq "$actual_author"; then
      fail "commit author missing from authors[]" "$context_file" "$index" "$commit"
    fi
  fi

  local sorted_decl_authors
  sorted_decl_authors=$(printf '%s
' "${decl_authors[@]}" | sort -u)
  if [ -n "$actual_author" ]; then
    if [ -z "$sorted_decl_authors" ]; then
      fail "authors[] empty but GitHub returned author" "$context_file" "$index" "$commit"
    fi
  else
    if [ -z "$sorted_decl_authors" ]; then
      fail "authors[] empty and author unresolved" "$context_file" "$index" "$commit"
    fi
  fi

  local pr_number
  pr_number=$(gh api -H "Accept: application/vnd.github.groot-preview+json" "repos/${OWNER}/${REPO}/commits/${commit}/pulls" --jq '.[0].number // empty' 2>/dev/null || echo "")
  local -a actual_approvers=()
  if [ -n "$pr_number" ]; then
    while IFS= read -r reviewer; do
      [ -z "$reviewer" ] && continue
      actual_approvers+=("$reviewer")
    done < <(gh api "repos/${OWNER}/${REPO}/pulls/${pr_number}/reviews" --jq '[.[] | select(.state=="APPROVED") | .user.login] | unique[]' 2>/dev/null || echo "")
  fi

  local sorted_decl_approvers
  sorted_decl_approvers=$(printf '%s
' "${decl_approvers[@]}" | sort -u)
  local sorted_actual_approvers
  sorted_actual_approvers=$(printf '%s
' "${actual_approvers[@]}" | sort -u)

  if [ -n "$sorted_actual_approvers" ]; then
    if [ -z "$sorted_decl_approvers" ]; then
      fail "approvers[] empty but GitHub shows approvals" "$context_file" "$index" "$commit"
    fi
    while IFS= read -r decl; do
      [ -z "$decl" ] && continue
      if ! printf '%s
' "${sorted_actual_approvers}" | grep -Fxq "$decl"; then
        fail "approver '$decl' not found in GitHub approvals" "$context_file" "$index" "$commit"
      fi
    done <<< "$sorted_decl_approvers"
  else
    if [ -n "$sorted_decl_approvers" ]; then
      fail "approvers listed but GitHub shows none" "$context_file" "$index" "$commit"
    fi
  fi
}

if [ -f "$CHOSEN_FILE" ]; then
  entry=$(cat "$CHOSEN_FILE" | jq -c '.')
  if [ -n "$entry" ] && [ "$entry" != "{}" ]; then
    check_entry "$entry" "chosen_commit.json" 0
  fi
fi

if [ -f "$ALTS_FILE" ]; then
  mapfile -t entries < <(jq -c '.[]' "$ALTS_FILE" 2>/dev/null || echo "")
  idx=0
  for entry in "${entries[@]}"; do
    [ -z "$entry" ] && continue
    check_entry "$entry" "alternatives.json" "$idx"
    idx=$((idx + 1))
  done
fi

exit 0
