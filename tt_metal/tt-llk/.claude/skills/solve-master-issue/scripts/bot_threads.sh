#!/bin/bash
# usage: bot_threads.sh <pr> [ackfile]
# Bot feedback on a PR that still needs handling. Empty output = nothing left to answer.
#   THREAD <thread-id> <last-comment-rest-id> path:line [bot] text
#          unresolved inline thread whose latest comment is from a bot
#   REVIEW <url> [bot] text
#          bot review body / issue comment NOT listed in <ackfile> (one URL per line)
# Ack a REVIEW you handled (or that needs no reply, e.g. "No issues found") by appending its URL to
# the ackfile, default ${MASTER_ISSUE_DIR:-$HOME/.claude/master-issues}/acked-<pr>.txt.
# Reply to a THREAD:   gh api repos/<repo>/pulls/<pr>/comments/<rest-id>/replies -f body='...'
# Resolve a THREAD:    gh api graphql -f query='mutation{resolveReviewThread(input:{threadId:"<thread-id>"}){thread{isResolved}}}'
# Bot = GitHub account type Bot (copilot-pull-request-reviewer, github-actions[bot], cycode-security, ...).
# Repo: $GH_REPO, default tenstorrent/tt-metal.
pr=$1; repo=${GH_REPO:-tenstorrent/tt-metal}; owner=${repo%/*}; name=${repo#*/}
ackdir=${MASTER_ISSUE_DIR:-$HOME/.claude/master-issues}
ack=${2:-$ackdir/acked-$pr.txt}; mkdir -p "$(dirname "$ack")" && touch "$ack"

gh api graphql --paginate -F owner="$owner" -F name="$name" -F pr="$pr" -f query='
query($owner:String!,$name:String!,$pr:Int!,$endCursor:String){repository(owner:$owner,name:$name){
 pullRequest(number:$pr){reviewThreads(first:100,after:$endCursor){pageInfo{hasNextPage endCursor}
  nodes{id isResolved isOutdated path line comments(last:50){nodes{databaseId author{__typename login} body}}}}}}}' \
 -q '.data.repository.pullRequest.reviewThreads.nodes[]
     | select(.isResolved|not)
     | .comments.nodes[-1] as $c
     | select(($c.author.__typename // "") == "Bot")
     | "THREAD \(.id) \($c.databaseId) \(.path):\(.line // "outdated") [\($c.author.login)] \($c.body | gsub("\\s+";" ") | .[0:200])"'

{ gh api "repos/$repo/issues/$pr/comments" --paginate \
     -q '.[]|select(.user.type=="Bot")|"\(.created_at)\t\(.html_url)\t[\(.user.login)]\t\(.body|gsub("\\s+";" ")|.[0:200])"'
  gh api "repos/$repo/pulls/$pr/reviews" --paginate \
     -q '.[]|select(.user.type=="Bot")|select(.body!="")|"\(.submitted_at)\t\(.html_url)\t[\(.user.login)]\t\(.body|gsub("\\s+";" ")|.[0:200])"'
} | sort | while IFS=$'\t' read -r _ url who body; do
    grep -qxF "$url" "$ack" || printf 'REVIEW %s %s %s\n' "$url" "$who" "$body"
done
