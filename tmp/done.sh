#!/usr/bin/env bash
# Exit 0 once both real e2e jobs from tmp/submit_attempt5.log have finished (or the submitter died).
cd "$(dirname "$0")/.."
ids=$(grep -oE 'Job [0-9]+ queued' tmp/submit_attempt5.log | awk '{print $2}')
n=$(echo "$ids" | grep -c .)
if [ "$n" -lt 2 ]; then pgrep -f "prewarm_and_submit.sh -e tmp/e2e_env.yaml" >/dev/null && exit 1; exit 0; fi
for j in $ids; do tt-device-mcp status -j $j | grep -qiE 'Status: +(running|queued|pending)' && exit 1; done
exit 0
