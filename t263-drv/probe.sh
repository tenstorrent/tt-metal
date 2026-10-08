#!/bin/bash
# Exit 0 once blx01's broker is healthy, its last incident is >= 15 min old and no smarton job runs or queues.
ssh -o ConnectTimeout=20 -o BatchMode=yes g15blx01 'bash -s' <<'R' || exit 1
s=$(python3 -c 'import json; print(json.load(open("/var/lib/tt-device-broker/health/fsm.json"))["state"])' 2>/dev/null)
[ "$s" = healthy ] || exit 1
[ "$(systemctl is-active tt-device-broker)" = active ] || exit 1
last=$(ls /var/lib/tt-device-broker/health/incidents | sort | tail -1)
lt=$(date -u -d "$(echo $last | sed -E 's/^(....)(..)(..)T(..)(..)(..)Z.*/\1-\2-\3 \4:\5:\6/')" +%s 2>/dev/null || echo 0)
[ $(($(date -u +%s) - lt)) -ge 900 ] || exit 1
timeout 40 tt-device-mcp status 1 2>&1 | sed -n '/^RUNNING/,/^RECENT/p' | grep -q smarton && exit 1
exit 0
R
