#!/usr/bin/env bash
# Exit 0 once the e2e job queued by tmp/drive6.sh has finished (or the driver died without submitting).
cd "$(dirname "$0")/.."
id=$(grep -oE 'Job [0-9]+ queued' tmp/drive6.jobs 2>/dev/null | awk '{print $2}' | head -1)
if [ -z "$id" ]; then pgrep -f "bash tmp/drive6.sh" >/dev/null && exit 1; exit 0; fi
tt-device-mcp status -j $id | grep -qiE 'Status: +(running|queued|pending)' && exit 1
exit 0
