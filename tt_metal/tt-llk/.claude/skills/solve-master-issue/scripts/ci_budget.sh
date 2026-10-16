#!/bin/bash
# usage: ci_budget.sh [--need-nightly N] [sanity=5] [nightly=5]
# Is there room to dispatch one more PR's CI without crowding other developers off the runners?
# Counts the current user's workflow_dispatch runs that are not completed (queued, waiting, pending,
# requested, in progress) — asked of GitHub, not read from the ledger, so runs dispatched by another
# session of the same account count too:
#   sanity  bucket = "Sanity tests"
#   nightly bucket = "Nightly tt-metal L2 tests" + "LLK e2e Tests"   (the long lanes share one budget)
# Prints  `sanity <used>/<cap>  nightly <used>/<cap>`  and exits
#   0  room for one more PR: sanity has a free slot AND nightly has --need-nightly free slots (default 1;
#      pass 2 when the PR also needs LLK e2e) — dispatch both lanes of that PR now
#   1  no room: put the PR in the ledger's CI queue and advance other work; re-check when a run completes
#   2  GitHub could not be read: not an answer, retry after the poll interval
# Caps: arguments, else $CI_BUDGET_SANITY / $CI_BUDGET_NIGHTLY, else 5 and 5. Repo: $GH_REPO.
set -uo pipefail
need=1
[ "${1:-}" = --need-nightly ] && { need=$2; shift 2; }
cap_s=${CI_BUDGET_SANITY:-5}; cap_n=${CI_BUDGET_NIGHTLY:-5}
for kv in "$@"; do case $kv in sanity=*) cap_s=${kv#*=};; nightly=*) cap_n=${kv#*=};; esac; done
repo=${GH_REPO:-tenstorrent/tt-metal}
me=$(gh api user -q .login) || exit 2
count() {  # count <workflow name>...: non-completed workflow_dispatch runs of mine across those workflows
    local n=0 wf wid c
    for wf in "$@"; do
        wid=$(gh api "repos/$repo/actions/workflows?per_page=100" --paginate -q ".workflows[]|select(.name==\"$wf\")|.id") || return 2
        wid=${wid%%$'\n'*}
        [ -z "$wid" ] && { echo "no workflow named '$wf'" >&2; return 2; }
        for st in queued in_progress waiting pending requested; do
            c=$(gh api "repos/$repo/actions/workflows/$wid/runs?event=workflow_dispatch&actor=$me&status=$st&per_page=100" \
                -q '.total_count') || return 2
            n=$((n + c))
        done
    done
    echo "$n"
}
used_s=$(count "Sanity tests") || exit 2
used_n=$(count "Nightly tt-metal L2 tests" "LLK e2e Tests") || exit 2
echo "sanity $used_s/$cap_s  nightly $used_n/$cap_n"
[ "$used_s" -lt "$cap_s" ] && [ $((used_n + need)) -le "$cap_n" ]
