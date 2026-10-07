#!/bin/bash
# t220 retry_when: exit 0 once blx03 (preferred) or blx01 has no broker hold/recovery running and
# no bridge-reset/hold/health-gate failure in its last 30 recent jobs within 30 min. 1 = not yet.
ok() {  # $1 = host
  ssh -o ConnectTimeout=10 "$1" 'tt-device-mcp status 30' 2>/dev/null | python3 -c '
import sys, re, datetime, zoneinfo
t = sys.stdin.read()
if not t.strip(): sys.exit(1)
run = t.split("QUEUED")[0]
if re.search(r"HELD|recovering|upgrade", run): sys.exit(1)
tz = zoneinfo.ZoneInfo("America/Los_Angeles"); now = datetime.datetime.now(tz)
for m in re.finditer(r"^(\d\d) (\d\d):(\d\d):(\d\d)\s+\S+\s+\S+\s+(bridge-reset|hold|health-gate)\s.*?(failed|started)", t, re.M):
    d, H, M, S = map(int, m.groups()[:4])
    ts = now.replace(day=d, hour=H, minute=M, second=S)
    if ts > now: ts -= datetime.timedelta(days=28)
    if (now - ts).total_seconds() < 1800: sys.exit(1)
'
}
ok g14blx03 && exit 0
ok blx01 && exit 0
exit 1
