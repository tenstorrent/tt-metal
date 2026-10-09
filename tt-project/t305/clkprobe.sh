#!/bin/bash
# Exit 0 when blx01's newest device-opening broker job ran at full AICLK (no "AICLK failed to settle"),
# or after the deadline; 1 while the 900 MHz clamp persists; 255 if ssh fails.
[ "$(date -u +%s)" -gt "$(date -u -d '2026-10-10 02:00' +%s)" ] && exit 0
ssh -o ConnectTimeout=15 -o BatchMode=yes blx01 '
  cd /var/log/tt-device-broker || exit 1
  for f in $(ls -t 2026-*.log | head -12); do
    grep -q "Physical groupings" "$f" || continue
    grep -q "AICLK failed to settle" "$f" && exit 1 || exit 0
  done
  exit 1'
