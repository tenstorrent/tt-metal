#!/bin/bash
# usage: bots_pending.sh <pr> [dispatched_run_id ...]
# What still has to finish before the PR's head SHA counts as "bots quiet". Empty output = quiet.
# Observed, not guessed:
#   RUN      a workflow run on the head SHA (auto-triggered reviewers, static checks, PR gate — any
#            event, incl. pull_request_target) that is still queued / in progress
#   DISPATCH an "LLK PR Review" run for this PR (found by name — its run name embeds "PR #<n>"),
#            or a run id you passed, that has not completed
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

gh api "repos/$repo/actions/workflows?per_page=100" --paginate -q '.workflows[]|select(.name=="LLK PR Review")|.id' | head -1 |
while read -r wid; do
    gh api "repos/$repo/actions/workflows/$wid/runs?per_page=30" \
        -q ".workflow_runs[]|select(.status!=\"completed\")|select(.name|test(\"PR #$pr\\\\b\"))|\"DISPATCH \(.status) \(.name) (run \(.id))\""
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
