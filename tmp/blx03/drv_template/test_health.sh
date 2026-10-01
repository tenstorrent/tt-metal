#!/bin/bash
# Offline check of health()/gate_fail_since() against sample broker log lines. No device access.
D=$(mktemp -d); SL=$D/broker.log; log() { echo "LOG: $*" >&2; }
systemctl() { return 0; }; tt-device-mcp() { return 0; }
eval "$(sed -n '/^OKRE=/,/^}/p' "$(dirname "$0")/driver.sh")"
fail=0
chk() { [ "$1" = "$2" ] && echo "PASS $3" || { echo "FAIL $3 (got $1 want $2)"; fail=1; }; }
L() { echo "$@" > $SL; }
for line in "2026-10-01 10:00:00 | INFO | HEALTH-GATE pre-job: OK" \
            "2026-10-01 10:00:00 | INFO | HEALTH-GATE: device healthy; no reset needed" \
            "2026-10-01 10:00:00 | INFO | heartbeat: HEALTHY 32 chips"; do
  L "$line"; health; chk $? 0 "health ok: $line"
  chk "$(gate_fail_since '2026-10-01 09:00:00')" "" "no alarm: $line"
done
for line in "2026-10-01 10:00:00 | ERROR | chip 24 dropped" \
            "2026-10-01 10:00:00 | WARN | ESCALATE: power cycle" \
            "2026-10-01 10:00:00 | INFO | RECOVER chip 24: device healthy; no reset needed" \
            "2026-10-01 10:00:00 | WARN | HEALTH-GATE pre-job: FAIL 31/32 chips"; do
  L "$line"; health; chk $? 1 "health bad: $line"
  [ -n "$(gate_fail_since '2026-10-01 09:00:00')" ]; chk $? 0 "alarm: $line"
done
L "2026-10-01 08:00:00 | ERROR | old"; chk "$(gate_fail_since '2026-10-01 09:00:00')" "" "old error ignored"
printf '%s\n' "2026-10-01 10:00:00 | WARN | ESCALATE x" "2026-10-01 10:05:00 | INFO | heartbeat: HEALTHY" > $SL; health; chk $? 0 "healthy after recovery"
rm -rf $D; exit $fail
