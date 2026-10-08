#!/bin/bash
# usage: ci_triage.sh <run_id>
# A tally of the run's jobs by conclusion, then for each non-successful, non-skipped job: conclusion,
# pytest/gtest summary line, failing test ids (FAIL) and ##[error] lines (ERR).
# Exit non-zero = GitHub could not be read (the listing is incomplete).
# Repo: $GH_REPO, default tenstorrent/tt-metal.
set -euo pipefail
id=$1; repo=${GH_REPO:-tenstorrent/tt-metal}
gh run view "$id" --repo "$repo" --json name,headBranch,status,conclusion,url \
    -q '"\(.name) @ \(.headBranch): \(.status)/\(.conclusion // "-")  \(.url)"'
jobs=$(gh api "repos/$repo/actions/runs/$id/jobs?per_page=100" --paginate -q '.jobs[] | "\(.id)\t\(.conclusion // .status)\t\(.name)"')
# tally first: a run whose test jobs are all "skipped" was dispatched without its opt-in inputs
printf '%s\n' "$jobs" | awk -F'\t' '$1!=""{n++; c[$2]++} END{printf "jobs: %d total", n; for(k in c) printf ", %d %s", c[k], k; print ""}'
printf '%s\n' "$jobs" | awk -F'\t' '$1!="" && $2!="success" && $2!="skipped"' |
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
