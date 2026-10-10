#!/bin/bash
# Exit 0 once blx03's broker is active and no longer holds the device; 1 otherwise; 255 if ssh fails.
out=$(timeout 40 ssh -o ConnectTimeout=10 -o BatchMode=yes g14blx03 \
  'systemctl is-active -q tt-device-broker && tt-device-mcp status 2>&1 | sed -n "/^RUNNING/,/^QUEUED/p"')
rc=$?
[ $rc -eq 255 ] && exit 255
[ $rc -ne 0 ] && exit 1
echo "$out" | grep -qE 'device HELD|recovery|gate/' && exit 1
exit 0
