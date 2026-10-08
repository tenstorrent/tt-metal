#!/bin/bash
# usage: bots_pending.sh <pr> [dispatched_run_id ...]
# What still has to finish before the PR's head SHA counts as "bots quiet". Empty output = quiet.
# Observed, not guessed:
#   RUN      a workflow run on the head SHA (auto-triggered reviewers, static checks, PR gate — any
#            event, incl. pull_request_target) that is still queued / in progress
#   DISPATCH an "LLK PR Review" run for this PR (found by name — its run name embeds "PR #<n>"),
#            or a run id you passed, that has not completed
#   FAILED   the latest "LLK PR Review" run for this PR did not succeed (infra failure, cancelled):
#            re-dispatch it once, then ack its URL (same ack file as bot_threads.sh) if it fails again
#   COPILOT  Copilot is a requested reviewer and has not submitted its review yet
#   SETTLE   no workflow run has registered for the head SHA yet and the head is < 10 min old —
#            GitHub takes a minute or two to queue runs after a push; after 10 min with no runs,
#            nothing is coming
#   THREAD / REVIEW   unanswered bot feedback (bot_threads.sh)
# Repo: $GH_REPO, default tenstorrent/tt-metal.
pr=$1; shift
repo=${GH_REPO:-tenstorrent/tt-metal}
here=$(dirname "$(readlink -f "$0")")
sha=$(gh api "repos/$repo/pulls/$pr" -q .head.sha)

runs=$(gh api "repos/$repo/actions/runs?head_sha=$sha&per_page=100" --paginate \
        -q '.workflow_runs[]|"\(.status)\t\(.name)\t\(.id)\t\(.created_at)"')
echo "$runs" | awk -F'\t' '$1!="" && $1!="completed" {print "RUN      " $2 " (" $1 ", run " $3 ")"}'

ack=${MASTER_ISSUE_DIR:-$HOME/.claude/master-issues}/acked-$pr.txt
gh api "repos/$repo/actions/workflows?per_page=100" --paginate -q '.workflows[]|select(.name=="LLK PR Review")|.id' | head -1 |
while read -r wid; do
    mine=$(gh api "repos/$repo/actions/workflows/$wid/runs?per_page=50" \
        -q ".workflow_runs[]|select(.name|test(\"PR #$pr\\\\b\"))|\"\(.status)\t\(.conclusion)\t\(.id)\t\(.html_url)\t\(.name)\"")
    echo "$mine" | awk -F'\t' '$1!="" && $1!="completed" {print "DISPATCH " $1 " " $5 " (run " $3 ")"}'
    latest=$(echo "$mine" | head -1)
    case $latest in completed$'\t'success*|"") ;; *)
        url=$(echo "$latest" | cut -f4)
        grep -qxF "$url" "$ack" 2>/dev/null || echo "FAILED   LLK PR Review $(echo "$latest" | cut -f2) $url — re-dispatch once, then ack";;
    esac
done

for rid in "$@"; do
    st=$(gh api "repos/$repo/actions/runs/$rid" -q '"\(.status) \(.name)"')
    case $st in completed*) ;; *) echo "DISPATCH $st (run $rid)";; esac
done

gh api "repos/$repo/pulls/$pr/requested_reviewers" -q '.users[]|select(.type=="Bot")|.login' |
    sed 's/^/COPILOT  review requested, not yet submitted: /'

if [ -z "$runs" ]; then
    # head commit's committer time is the closest thing to push time without a push event
    age=$(( $(date +%s) - $(date -d "$(gh api "repos/$repo/commits/$sha" -q .commit.committer.date)" +%s) ))
    [ "$age" -lt 600 ] && echo "SETTLE   no workflow run registered for $sha yet (head is ${age}s old)"
fi

"$here/bot_threads.sh" "$pr"
