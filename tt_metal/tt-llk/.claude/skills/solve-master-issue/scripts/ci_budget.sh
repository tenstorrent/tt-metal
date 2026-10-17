#!/bin/bash
# usage: ci_budget.sh [--need-nightly N] [--verify] [sanity=5] [nightly=5]
# Is there room for one more PR's CI? Counts MY non-completed workflow_dispatch runs on GitHub (never the
# ledger): "Sanity tests" against the sanity cap; "Nightly tt-metal L2 tests" + "LLK e2e Tests" against the
# nightly cap (--need-nightly 2 when a PR needs both). Prints `sanity <used>/<cap>  nightly <used>/<cap>`.
# Exit 0 = dispatch, 1 = queue the PR, 2 = GitHub could not be read (treat as "not now").
# Caps: args, $CI_BUDGET_SANITY / $CI_BUDGET_NIGHTLY, default 5 each.
# Counting by GitHub actor makes the cap global: runs dispatched from any box under this login count.
# --verify: run it right AFTER a dispatch. The gate is check-then-dispatch, so two workers on two boxes can
#   both see 4/5 and both dispatch; --verify re-counts with the new runs included and exits 1 when either
#   lane is now OVER its cap — the caller cancels the run(s) it just dispatched (`gh run cancel <id>`) and
#   puts the PR back at the head of its ci-queue. (used <= cap passes under --verify; used < cap before.)
set -uo pipefail
need=1; verify=0
while :; do case "${1:-}" in
    --need-nightly) need=$2; shift 2;;
    --verify) verify=1; shift;;
    *) break;;
esac; done
cap_s=${CI_BUDGET_SANITY:-5}; cap_n=${CI_BUDGET_NIGHTLY:-5}
for kv in "$@"; do for part in ${kv//,/ }; do case $part in sanity=*) cap_s=${part#*=};; nightly=*) cap_n=${part#*=};; esac; done; done   # "sanity=5 nightly=5" or "sanity=5,nightly=5"
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
if [ "$verify" -eq 1 ]; then
    [ "$used_s" -le "$cap_s" ] && [ "$used_n" -le "$cap_n" ] || { echo "OVER budget after dispatch: cancel the run(s) you just dispatched and re-queue the PR" >&2; exit 1; }
    exit 0
fi
[ "$used_s" -lt "$cap_s" ] && [ $((used_n + need)) -le "$cap_n" ]
