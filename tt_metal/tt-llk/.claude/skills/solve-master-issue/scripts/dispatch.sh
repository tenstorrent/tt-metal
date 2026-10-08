#!/bin/bash
# usage: dispatch.sh "<workflow name>" <ref> [-f key=value ...]
# `gh workflow run` returns no run id. This dispatches, then polls (up to 90 s) for the new
# workflow_dispatch run by the current user on that workflow created after the dispatch, and prints
#   <run_id> <html_url>
# so the run can be watched (`gh run watch -i 120 --exit-status <id>`) and recorded in the ledger.
# Exit 1 if no run appeared — check `gh run list -w "<workflow>"` by hand before dispatching again.
# Repo: $GH_REPO, default tenstorrent/tt-metal.
set -e
wf=$1; ref=$2; shift 2
repo=${GH_REPO:-tenstorrent/tt-metal}
me=$(gh api user -q .login)
wid=$(gh api "repos/$repo/actions/workflows?per_page=100" --paginate -q ".workflows[]|select(.name==\"$wf\")|.id" | head -1)
[ -z "$wid" ] && { echo "no workflow named '$wf'" >&2; exit 2; }
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
gh workflow run "$wf" --repo "$repo" --ref "$ref" "$@"
for _ in $(seq 1 18); do
    sleep 5
    r=$(gh api "repos/$repo/actions/workflows/$wid/runs?event=workflow_dispatch&actor=$me&created=>=$since&per_page=5" \
          -q '.workflow_runs[]|"\(.id) \(.html_url)"' | tail -1)
    [ -n "$r" ] && { echo "$r"; exit 0; }
done
echo "dispatched but no run found for '$wf' @ $ref since $since" >&2; exit 1
