#!/bin/bash
# usage: ci_triage.sh <run_id>
# For each non-successful job of a run: conclusion, pytest/gtest summary line, and failing test ids.
# Exit non-zero = GitHub could not be read (the listing is incomplete).
# Repo: $GH_REPO, default tenstorrent/tt-metal.
set -euo pipefail
id=$1; repo=${GH_REPO:-tenstorrent/tt-metal}
gh run view "$id" --repo "$repo" --json name,headBranch,status,conclusion,url \
    -q '"\(.name) @ \(.headBranch): \(.status)/\(.conclusion // "-")  \(.url)"'
gh api "repos/$repo/actions/runs/$id/jobs?per_page=100" --paginate \
    -q '.jobs[] | select(.conclusion != "success" and .conclusion != "skipped") | "\(.id)\t\(.conclusion // .status)\t\(.name)"' |
while IFS=$'\t' read -r j c n; do
    echo "== [$c] $n (job $j)"
    { [ "$c" = in_progress ] || [ "$c" = queued ]; } && continue
    log=$(gh api "repos/$repo/actions/jobs/$j/logs" 2>/dev/null) || { echo "   (log unavailable)"; continue; }
    echo "$log" | grep -E "[0-9]+ (passed|failed).* in [0-9.]+s|\[  (PASSED|FAILED)  \] [0-9]+ test" |
        sed 's/^[0-9TZ:.-]* //' | tail -2 | sed 's/^/   /' || true
    echo "$log" | grep -E "^[0-9TZ:.-]* (FAILED |ERROR |⨯ |\[  FAILED  \] )" |
        sed 's/^[0-9TZ:.-]* //' | sort -u | head -12 | sed 's/^/   FAIL /' || true
    echo "$log" | grep -iE "##\[error\]" | sed 's/^[0-9TZ:.-]* //' | sort -u | head -4 | sed 's/^/   ERR  /' || true
done
