#!/bin/bash
# usage: dispatch.sh [--match <regex>] "<workflow name>" <branch> [-f key=value ...]
# `gh workflow run` returns no run id. This snapshots the current user's workflow_dispatch runs of
# <workflow> on <branch>, dispatches, then polls (up to 90 s) for exactly one run that was not in the
# snapshot — and, with --match, whose run name matches <regex> — and prints
#   <run_id> <html_url>
# so the run can be watched (`gh run watch -i 120 --exit-status <id>`) and recorded in the ledger.
# Use --match whenever the workflow+branch is shared with other dispatches, so a run that is not
# yours can never be returned to you: "LLK PR Review" runs for every PR share actor and ref main,
# and its run name embeds the PR, so pass --match 'PR #<n>\b'. Non-matching new runs are ignored.
# Exit 1: no (matching) new run appeared — check `gh run list -w "<workflow>"` by hand before
#         dispatching again.
# Exit 3: more than one matching new run appeared (something else dispatched the same thing
#         meanwhile); the candidates are printed, nothing is chosen for you.
# Repo: $GH_REPO, default tenstorrent/tt-metal.
set -euo pipefail
match=
[ "${1:-}" = --match ] && { match=$2; shift 2; }
wf=$1; ref=$2; shift 2
repo=${GH_REPO:-tenstorrent/tt-metal}
me=$(gh api user -q .login)
wid=$(gh api "repos/$repo/actions/workflows?per_page=100" --paginate -q ".workflows[]|select(.name==\"$wf\")|.id")
wid=${wid%%$'\n'*}
[ -z "$wid" ] && { echo "no workflow named '$wf'" >&2; exit 2; }
q="repos/$repo/actions/workflows/$wid/runs?event=workflow_dispatch&actor=$me&branch=$ref&per_page=20"
before=$(gh api "$q" -q '.workflow_runs[].id')
gh workflow run "$wf" --repo "$repo" --ref "$ref" "$@"
for _ in $(seq 1 18); do
    sleep 5
    new=$(gh api "$q" -q '.workflow_runs[]|"\(.id) \(.html_url) \(.name)"' |
          awk -v b="$before" -v m="$match" 'BEGIN{n=split(b,a,"\n"); for(i=1;i<=n;i++) seen[a[i]]=1}
               !seen[$1] && (m=="" || $0 ~ m) {print $1, $2}')
    n=$(printf '%s\n' "$new" | grep -c . || true)
    [ "$n" -eq 1 ] && { echo "$new"; exit 0; }
    [ "$n" -gt 1 ] && { printf 'ambiguous: %s new runs of "%s" @ %s:\n%s\n' "$n" "$wf" "$ref" "$new" >&2; exit 3; }
done
echo "dispatched but no new run found for '$wf' @ $ref" >&2; exit 1
