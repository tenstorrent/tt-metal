#!/bin/bash
# t295 wake probe: exit 0 once blx03 or blx01 has been up >=20 min and its broker shows no hold,
# reset or health check running. 1 otherwise (unreachable counts as not yet).
for h in g14blx03 blx01; do
  out=$(timeout 25 ssh -o ConnectTimeout=8 -o BatchMode=yes $h \
    'echo UP=$(( $(date +%s) - $(date -d "$(uptime -s)" +%s) )); tt-device-mcp status 2>&1 | sed -n "/^RUNNING/,/^QUEUED/p"' 2>/dev/null) || continue
  up=$(sed -n 's/^UP=//p' <<<"$out")
  [ -n "$up" ] && [ "$up" -ge 1200 ] || continue
  grep -qiE 'hold|health|reset|fabric-check|reboot' <<<"$out" && continue
  exit 0
done
exit 1
