#!/usr/bin/env bash
# Exit 0 once both real e2e jobs queued by tmp/drive6.sh have finished (or the driver died without submitting).
cd "$(dirname "$0")/.."
ids=$(grep -oE 'Job [0-9]+ queued' tmp/drive6.jobs 2>/dev/null | awk '{print $2}')
if [ "$(echo "$ids" | grep -c .)" -lt 2 ]; then pgrep -f "tmp/drive6.sh" >/dev/null && exit 1; exit 0; fi
for j in $ids; do tt-device-mcp status -j $j | grep -qiE 'Status: +(running|queued|pending)' && exit 1; done
exit 0
