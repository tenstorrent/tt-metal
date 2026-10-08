#!/bin/bash
# usage: bot_threads.sh <pr> [ackfile]
# Bot feedback on a PR that still needs handling. Exit 0 AND empty output = nothing left to answer.
# A non-zero exit means GitHub could not be read: the output is incomplete, never "nothing left".
#   THREAD <thread-id> <root-comment-rest-id> path:line [bot] text
#          unresolved inline thread that a bot OPENED and whose latest comment is from a bot
#          (so a bot follow-up to your reply re-lists it; a thread a human opened never appears)
#   REVIEW <url> [bot] text
#          bot review body / issue comment NOT listed in <ackfile> (one URL per line)
# Ack a REVIEW you handled (or that needs no reply, e.g. "No issues found") by appending its URL to
# the ackfile, default ${MASTER_ISSUE_DIR:-$HOME/.claude/master-issues}/acked-<pr>.txt.
# Reply to a THREAD:   gh api repos/<repo>/pulls/<pr>/comments/<root-rest-id>/replies -f body='...'
#                      (the replies endpoint takes the thread's top-level comment id)
# Resolve a THREAD:    gh api graphql -f query='mutation{resolveReviewThread(input:{threadId:"<thread-id>"}){thread{isResolved}}}'
# Bot = GitHub account type Bot (copilot-pull-request-reviewer, github-actions[bot], cycode-security, ...).
# Repo: $GH_REPO, default tenstorrent/tt-metal.
set -euo pipefail
pr=$1; repo=${GH_REPO:-tenstorrent/tt-metal}; owner=${repo%/*}; name=${repo#*/}
ackdir=${MASTER_ISSUE_DIR:-$HOME/.claude/master-issues}
ack=${2:-$ackdir/acked-$pr.txt}; mkdir -p "$(dirname "$ack")" && touch "$ack"

gh api graphql --paginate -F owner="$owner" -F name="$name" -F pr="$pr" -f query='
query($owner:String!,$name:String!,$pr:Int!,$endCursor:String){repository(owner:$owner,name:$name){
 pullRequest(number:$pr){reviewThreads(first:100,after:$endCursor){pageInfo{hasNextPage endCursor}
  nodes{id isResolved isOutdated path line
        root:comments(first:1){nodes{databaseId author{__typename login}}}
        latest:comments(last:1){nodes{author{__typename login} body}}}}}}}' \
 -q '.data.repository.pullRequest.reviewThreads.nodes[]
     | select(.isResolved|not)
     | .root.nodes[0] as $r | .latest.nodes[0] as $c
     | select(($r.author.__typename // "") == "Bot" and ($c.author.__typename // "") == "Bot")
     | "THREAD \(.id) \($r.databaseId) \(.path):\(.line // "outdated") [\($c.author.login)] \($c.body | gsub("\\s+";" ") | .[0:200])"'

{ gh api "repos/$repo/issues/$pr/comments" --paginate \
     -q '.[]|select(.user.type=="Bot")|"\(.created_at)\t\(.html_url)\t[\(.user.login)]\t\(.body|gsub("\\s+";" ")|.[0:200])"'
  gh api "repos/$repo/pulls/$pr/reviews" --paginate \
     -q '.[]|select(.user.type=="Bot")|select(.body!="")|"\(.submitted_at)\t\(.html_url)\t[\(.user.login)]\t\(.body|gsub("\\s+";" ")|.[0:200])"'
} | sort | while IFS=$'\t' read -r _ url who body; do
    grep -qxF "$url" "$ack" || printf 'REVIEW %s %s %s\n' "$url" "$who" "$body"
done
