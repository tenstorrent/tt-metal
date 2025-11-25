#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="${ROOT}/output/chosen_commit.json"
OWNER="tenstorrent"
REPO="tt-metal"

fail() {
  local reason="$1"
  echo "detected hallucination: $reason"
  echo "  commit: $COMMIT"
  echo "  declared commit url: ${DECL_URL:-<none>}"
  echo "  expected commit url: $EXPECTED_URL"
  echo "  declared authors (login): ${DECL_AUTHORS[*]:-<none>}"
  echo "  actual author (login): ${actual_author:-<unknown>}"
  echo "  declared approvers: ${DECL_APPROVERS[*]:-<none>}"
  echo "  actual approvers: ${ACTUAL_APPROVERS[*]:-<none>}"
  exit 1
}

if [ ! -f "$OUTPUT_FILE" ]; then
  exit 0
fi

COMMIT=$(jq -r '.commit // .commit_sha // empty' "$OUTPUT_FILE")
if [ -z "$COMMIT" ]; then
  exit 0
fi

EXPECTED_URL="https://github.com/${OWNER}/${REPO}/commit/${COMMIT}"
DECL_URL=$(jq -r '.commit_url // empty' "$OUTPUT_FILE")
if [ -n "$DECL_URL" ] && [ "$DECL_URL" != "$EXPECTED_URL" ]; then
  fail "commit_url mismatch"
fi

mapfile -t DECL_AUTHORS < <(jq -r '.authors[]?.login // empty' "$OUTPUT_FILE")
mapfile -t DECL_APPROVERS < <(jq -r '.approvers[]?.login // empty' "$OUTPUT_FILE")

# Helper to produce sorted unique string
sorted_unique() {
  if [ "$#" -eq 0 ]; then
    return
  fi
  printf '%s\n' "$@" | sort -u
}

actual_author=$(gh api "repos/${OWNER}/${REPO}/commits/${COMMIT}" --jq '.author.login // empty' 2>/dev/null || echo "")
if [ -n "$actual_author" ]; then
  if ! printf '%s\n' "${DECL_AUTHORS[@]}" | grep -Fxq "$actual_author"; then
    fail "commit author missing from authors[]"
  fi
fi

# fetch approvals
PR_NUM=$(gh api -H "Accept: application/vnd.github.groot-preview+json" "repos/${OWNER}/${REPO}/commits/${COMMIT}/pulls" --jq '.[0].number // empty' 2>/dev/null || echo "")
ACTUAL_APPROVERS=()
if [ -n "$PR_NUM" ]; then
  while IFS= read -r reviewer; do
    [ -z "$reviewer" ] && continue
    ACTUAL_APPROVERS+=("$reviewer")
  done < <(gh api "repos/${OWNER}/${REPO}/pulls/${PR_NUM}/reviews" --jq '[.[] | select(.state=="APPROVED") | .user.login] | unique[]' 2>/dev/null || echo "")
fi

sorted_decl_authors=$(sorted_unique "${DECL_AUTHORS[@]}")
if [ -n "$actual_author" ]; then
  if [ -z "$sorted_decl_authors" ]; then
    fail "authors[] empty but GitHub returned a primary author"
  fi
else
  # if we couldn't resolve an actual author, require LLM to note at least one author
  if [ -z "$sorted_decl_authors" ]; then
    fail "authors[] empty and actual author could not be resolved"
  fi
fi

sorted_actual_approvers=$(sorted_unique "${ACTUAL_APPROVERS[@]}")
sorted_decl_approvers=$(sorted_unique "${DECL_APPROVERS[@]}")

if [ -n "$sorted_actual_approvers" ]; then
  if [ "$sorted_actual_approvers" != "$sorted_decl_approvers" ]; then
    fail "approver list mismatch"
  fi
else
  if [ -n "$sorted_decl_approvers" ]; then
    fail "approved reviewers missing in GitHub data"
  fi
fi

exit 0
