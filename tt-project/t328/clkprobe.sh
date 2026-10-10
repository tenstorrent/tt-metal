#!/bin/bash
# t328 retry_when: exit 0 once blx03's newest finished device-opening broker job ran at full AICLK
# (its log has no "AICLK failed to settle"); 1 otherwise (ssh failure counts as not yet).
# Logs touched in the last 120 s are skipped: a job still opening may not have logged the clamp yet.
timeout 40 ssh -o ConnectTimeout=10 -o BatchMode=yes g14blx03 '
  cd /var/log/tt-device-broker || exit 1
  now=$(date +%s)
  for f in $(ls -t 2026-*.log | head -15); do
    [ $((now - $(stat -c %Y "$f"))) -lt 120 ] && continue
    grep -q "Opening user mode device driver" "$f" || continue
    grep -q "AICLK failed to settle" "$f" && exit 1 || exit 0
  done; exit 1' || exit 1
