#!/bin/bash
# usage: dispatch.sh "<workflow name>" <branch> [-f key=value ...]
# `gh workflow run` returns no run id. This snapshots the current user's workflow_dispatch runs of
# <workflow> on <branch>, dispatches, then polls (up to 90 s) for exactly one run that was not in the
# snapshot and prints
#   <run_id> <html_url>
# so the run can be watched (`gh run watch -i 120 --exit-status <id>`) and recorded in the ledger.
# Exit 1: no new run appeared — check `gh run list -w "<workflow>"` by hand before dispatching again.
# Exit 3: more than one new run appeared (something else dispatched the same workflow on the same
#         branch meanwhile); the candidates are printed, nothing is chosen for you. For "LLK PR Review"
#         this is harmless — bots_pending.sh finds that run by the "PR #<n>" in its name.
# Repo: $GH_REPO, default tenstorrent/tt-metal.
set -euo pipefail
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
    new=$(gh api "$q" -q '.workflow_runs[]|"\(.id) \(.html_url)"' |
          awk -v b="$before" 'BEGIN{n=split(b,a,"\n"); for(i=1;i<=n;i++) seen[a[i]]=1} !seen[$1]')
    n=$(printf '%s\n' "$new" | grep -c . || true)
    [ "$n" -eq 1 ] && { echo "$new"; exit 0; }
    [ "$n" -gt 1 ] && { printf 'ambiguous: %s new runs of "%s" @ %s:\n%s\n' "$n" "$wf" "$ref" "$new" >&2; exit 3; }
done
echo "dispatched but no new run found for '$wf' @ $ref" >&2; exit 1
