#!/bin/bash
# Exit 0 once job 349 is done AND one of: (a) blx01's newest device-opening broker job ran at full AICLK,
# (b) blx03 answers ssh and tt-project/state/ready/g14blx03.READY exists, (c) 2026-10-11 02:00 UTC passed.
# 1 otherwise; 255 if blx01 ssh fails.
s=$(ssh -o ConnectTimeout=15 -o BatchMode=yes blx01 'tt-device-mcp status -j 349 2>&1
  cd /var/log/tt-device-broker || exit 1
  for f in $(ls -t 2026-*.log | head -12); do
    grep -q "Opening local chip ids" "$f" || continue
    grep -q "AICLK failed to settle" "$f" && echo CLAMPED || echo FULLCLK; break
  done') || exit 255
echo "$s" | grep -qiE "^Status:.*(running|queued|pending)" && exit 1
echo "$s" | grep -q FULLCLK && exit 0
[ -e tt-project/state/ready/g14blx03.READY ] && timeout 20 ssh -o ConnectTimeout=10 -o BatchMode=yes g14blx03 true && exit 0
[ "$(date -u +%s)" -ge "$(date -u -d '2026-10-11 02:00' +%s)" ] && exit 0
exit 1
