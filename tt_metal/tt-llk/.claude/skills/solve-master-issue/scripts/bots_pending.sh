#!/bin/bash
# usage: bots_pending.sh <pr> [dispatched_run_id ...]
# What still has to finish before the PR's head SHA counts as "bots quiet". Exit 0 AND empty output =
# quiet. A non-zero exit means GitHub could not be read: not quiet, retry after the poll interval.
# Observed, not guessed:
#   RUN      a workflow run on the head SHA (auto-triggered reviewers, static checks, PR gate — any
#            event, incl. pull_request_target) that is still queued / in progress
#   DISPATCH an "LLK PR Review" run for this PR (found by name — its run name embeds "PR #<n>" — among
#            every run of that workflow since the PR was opened), or a run id you passed, not completed
#   FAILED   the latest "LLK PR Review" run for this PR did not succeed (infra failure, cancelled):
#            re-dispatch it once, then ack its URL (same ack file as bot_threads.sh) if it fails again
#   COPILOT  Copilot is a requested reviewer and has not submitted its review yet
#   SETTLE   no workflow run has registered for the head SHA yet and this script first saw that SHA
#            (with no runs) under 10 min ago — GitHub takes a minute or two to queue runs after a push.
#            Run this right after every push so that clock starts at the push; the commit date is
#            useless for it (commits are made long before they are pushed). After 10 min with no
#            runs, nothing is coming. State: $MASTER_ISSUE_DIR/seen-<pr>.txt (survives resume).
#   THREAD / REVIEW   unanswered bot feedback (bot_threads.sh)
# Repo: $GH_REPO, default tenstorrent/tt-metal.
set -euo pipefail
pr=$1; shift
repo=${GH_REPO:-tenstorrent/tt-metal}
here=$(dirname "$(readlink -f "$0")")
state=${MASTER_ISSUE_DIR:-$HOME/.claude/master-issues}
ack=$state/acked-$pr.txt; seen=$state/seen-$pr.txt
mkdir -p "$state"; touch "$ack" "$seen"
prinfo=$(gh api "repos/$repo/pulls/$pr" -q '"\(.head.sha) \(.created_at)"')
sha=${prinfo%% *}; opened=${prinfo#* }

runs=$(gh api "repos/$repo/actions/runs?head_sha=$sha&per_page=100" --paginate \
        -q '.workflow_runs[]|"\(.status)\t\(.name)\t\(.id)\t\(.created_at)"')
echo "$runs" | awk -F'\t' '$1!="" && $1!="completed" {print "RUN      " $2 " (" $1 ", run " $3 ")"}'

wid=$(gh api "repos/$repo/actions/workflows?per_page=100" --paginate -q '.workflows[]|select(.name=="LLK PR Review")|.id')
wid=${wid%%$'\n'*}
if [ -n "$wid" ]; then
    # every review run for this PR was created after the PR was opened: a bounded, complete window
    mine=$(gh api "repos/$repo/actions/workflows/$wid/runs?created=>=$opened&per_page=100" --paginate \
        -q ".workflow_runs[]|select(.name|test(\"PR #$pr\\\\b\"))|\"\(.status)\t\(.conclusion)\t\(.id)\t\(.html_url)\t\(.name)\"")
    echo "$mine" | awk -F'\t' '$1!="" && $1!="completed" {print "DISPATCH " $1 " " $5 " (run " $3 ")"}'
    latest=$(echo "$mine" | head -1)
    case $latest in completed$'\t'success*|completed$'\t'skipped*|"") ;;
        completed*)
            url=$(echo "$latest" | cut -f4)
            grep -qxF "$url" "$ack" || echo "FAILED   LLK PR Review $(echo "$latest" | cut -f2) $url — re-dispatch once, then ack";;
    esac
fi

for rid in "$@"; do
    st=$(gh api "repos/$repo/actions/runs/$rid" -q '"\(.status) \(.name)"')
    case $st in completed*) ;; *) echo "DISPATCH $st (run $rid)";; esac
done

gh api "repos/$repo/pulls/$pr/requested_reviewers" -q '.users[]|select(.type=="Bot")|.login' |
    sed 's/^/COPILOT  review requested, not yet submitted: /'

if [ -z "$runs" ]; then
    first=$(awk -v s="$sha" '$1==s{print $2}' "$seen")
    if [ -z "$first" ]; then first=$(date +%s); echo "$sha $first" >> "$seen"; fi
    age=$(( $(date +%s) - first ))
    if [ "$age" -lt 600 ]; then
        echo "SETTLE   no workflow run registered for $sha yet (first seen ${age}s ago)"
    fi
fi

"$here/bot_threads.sh" "$pr"
