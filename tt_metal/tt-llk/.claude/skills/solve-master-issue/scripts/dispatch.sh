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
# Exit 4: Nightly requested for a ttllk-only PR — not dispatched (policy, see below); use Sanity + LLK e2e.
# Repo: $GH_REPO, default tenstorrent/tt-metal.
set -euo pipefail
match=
[ "${1:-}" = --match ] && { match=$2; shift 2; }
wf=$1; ref=$2; shift 2
repo=${GH_REPO:-tenstorrent/tt-metal}
me=$(gh api user -q .login)
# Policy (2026-10-17): the Nightly L2 is not dispatched for a ttllk-only PR (every changed file under
# tt_metal/tt-llk/); its CI set is Sanity + LLK e2e, and the PR is green when those pass. Exit 4 tells the
# caller so; CI_SCOPE_FORCE_NIGHTLY=1 overrides for a documented exception.
if [ "$wf" = "Nightly tt-metal L2 tests" ] && [ -z "${CI_SCOPE_FORCE_NIGHTLY:-}" ]; then
    prn=$(gh pr list --repo "$repo" --head "$ref" --state open --json number -q '.[0].number' 2>/dev/null || true)
    if [ -n "$prn" ] && "$(dirname "$0")/ci_scope.sh" --pr "$prn" > /dev/null 2>&1; then
        cat >&2 <<MSG
not dispatched: PR #$prn ($ref) is ttllk-only (all changes under tt_metal/tt-llk/). Policy 2026-10-17: its CI set is
Sanity (run-llk-sanity-tests=true) + LLK e2e Tests only; do NOT dispatch or re-dispatch the Nightly for it. Budget it
with ci_budget.sh --need-nightly 1. Record in the ledger and the PR body: "Nightly L2: not run (tt-llk-only change;
Sanity LLK smoke + LLK e2e cover it)". The PR is green when Sanity + LLK e2e pass on the final SHA.
MSG
        exit 4
    fi
fi
wid=$(gh api "repos/$repo/actions/workflows?per_page=100" --paginate -q ".workflows[]|select(.name==\"$wf\")|.id")
wid=${wid%%$'\n'*}
[ -z "$wid" ] && { echo "no workflow named '$wf'" >&2; exit 2; }
q="repos/$repo/actions/workflows/$wid/runs?event=workflow_dispatch&actor=$me&branch=$ref&per_page=20"
before=$(gh api "$q" -q '.workflow_runs[].id')
gh workflow run "$wf" --repo "$repo" --ref "$ref" "$@"
for _ in $(seq 1 18); do
    sleep 5
    # the match goes through grep -E, not awk -v (which turns \b into a backspace); no match is not an error
    new=$(gh api "$q" -q '.workflow_runs[]|"\(.id) \(.html_url) \(.name)"' |
          awk -v b="$before" 'BEGIN{n=split(b,a,"\n"); for(i=1;i<=n;i++) seen[a[i]]=1} !seen[$1]' |
          { if [ -n "$match" ]; then grep -E -- "$match" || [ $? -eq 1 ]; else cat; fi; } |
          awk '{print $1, $2}')
    n=$(printf '%s\n' "$new" | grep -c . || true)
    [ "$n" -eq 1 ] && { echo "$new"; exit 0; }
    [ "$n" -gt 1 ] && { printf 'ambiguous: %s new runs of "%s" @ %s:\n%s\n' "$n" "$wf" "$ref" "$new" >&2; exit 3; }
done
echo "dispatched but no new run found for '$wf' @ $ref" >&2; exit 1
