#!/bin/bash
# usage: main_job_status.sh "<workflow name>" "<job name substring>" [branch=main] [n_runs=5]
# The matching job(s) in the last n completed runs of <workflow> on <branch>, newest first: conclusion,
# and for every job that did not succeed, its failing test ids / error lines (same grep as ci_triage.sh).
# "unrelated, main also broken" needs the SAME failing tests here as in the PR run — a job that fails
# on main on different tests is not evidence, it is a different failure. A run that prints "no matching
# job" ran a different matrix — it is NOT evidence that main is green. Repo: $GH_REPO, default
# tenstorrent/tt-metal.
set -euo pipefail
wf=$1; job=$2; br=${3:-main}; n=${4:-5}; repo=${GH_REPO:-tenstorrent/tt-metal}
wid=$(gh api "repos/$repo/actions/workflows?per_page=100" --paginate -q ".workflows[] | select(.name == \"$wf\") | .id")
wid=${wid%%$'\n'*}
[ -z "$wid" ] && { echo "no workflow named '$wf'" >&2; exit 2; }
gh api "repos/$repo/actions/workflows/$wid/runs?branch=$br&status=completed&per_page=$n" \
    -q '.workflow_runs[] | "\(.id)\t\(.created_at)\t\(.head_sha[0:11])"' |
while IFS=$'\t' read -r rid t sha; do
    jobs=$(gh api "repos/$repo/actions/runs/$rid/jobs?per_page=100" --paginate \
        -q ".jobs[] | select(.name | contains(\"$job\")) | \"\(.conclusion)\t\(.id)\t\(.name)\"")
    [ -z "$jobs" ] && { echo "$t $sha run=$rid  no matching job"; continue; }
    echo "$jobs" | while IFS=$'\t' read -r c j name; do
        echo "$t $sha run=$rid  $c  $name"
        case $c in success|skipped) continue;; esac
        log=$(gh api "repos/$repo/actions/jobs/$j/logs" 2>/dev/null) || { echo "     (log unavailable)"; continue; }
        echo "$log" | grep -E "^[0-9TZ:.-]* (FAILED |ERROR |⨯ |\[  FAILED  \] )" |
            sed 's/^[0-9TZ:.-]* //' | sort -u | head -12 | sed 's/^/     FAIL /' || true
    done
done
