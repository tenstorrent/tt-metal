#!/bin/bash
# Exits 0 once blx03's broker is active and no device hold is running; 1 while held; 255 if ssh fails.
ssh -o ConnectTimeout=10 -o BatchMode=yes g14blx03 \
  'systemctl is-active -q tt-device-broker || exit 1; tt-device-mcp status 3 2>&1 | grep -qE "(running|started) .*HELD" && exit 1; exit 0'
