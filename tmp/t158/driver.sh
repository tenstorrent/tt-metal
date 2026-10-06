#!/bin/bash
# t158 driver on g15blx02: wait for build + cache staging, then ONE broker job (warm 4x8 1080p e2e,
# seed 0), then the post-job drop check. Health logic copied from tmp/t141/driver.sh.
# Marker: $V/DRIVER.done holds "<exit code> <reason>". WATCH_JOB=<id> re-attaches without submitting.
# Broker timeout is capped at 600 s by a local hook: prewarm kernels before starting this.
W=/home/smarton/fasth3/tt-metal/tt-project/worktrees/t158
RUN=/home/smarton/fasth3/tt-metal/tt-project/state/runs/645
V=/home/smarton/fasth3/tt-metal/tt-project/data/g15/driver; mkdir -p $V
SL=/var/log/tt-device-broker/server.log; D=$V/driver.log; BOOT0=$(uptime -s)
log() { echo "$(date -u '+%F %T') $*" >> $D; }
done_() { log "T158_DRIVER_DONE $1 $2"; echo "$1 $2" > $V/DRIVER.done; exit 0; }
now() { date -u '+%F %T'; }
OKRE=': OK|device healthy; no reset needed|heartbeat: HEALTHY'
BADRE='[|] ERROR [|]|ESCALATE|RECOVER'
errors_since() { awk -v s="$1" 'substr($0,1,19) > s && /\| ERROR \|/' $SL; }
gate_fail_since() { awk -v s="$1" -v ok="$OKRE" -v bad="$BADRE" 'substr($0,1,19) > s && ($0 ~ bad || (/HEALTH-GATE/ && $0 !~ ok))' $SL; }
health() {
  systemctl is-active -q tt-device-broker || { log "health: broker inactive"; return 1; }
  st=$(tt-device-mcp status 1 2>&1) || { log "health: status failed"; return 1; }
  echo "$st" | sed -n '/^RUNNING/,/^QUEUED/p' | grep -qiE '🔧|health-gate|fabric-check|recover|upgrade|bridge-reset|power-cycle' && { log "health: broker gate/recovery running"; return 1; }
  last=$(grep -E "HEALTH-GATE|$OKRE|ESCALATE|RECOVER|[|] ERROR [|]" $SL | tail -1)
  if echo "$last" | grep -qE "$OKRE" && ! echo "$last" | grep -qE "$BADRE"; then return 0; fi
  log "health: last event not healthy: ${last:0:200}"; return 1
}
log "start boot=$BOOT0"
if [ -z "$WATCH_JOB" ]; then
  for i in $(seq 360); do [ -e $RUN/t158-build2.rc ] && [ -e $RUN/t158-stage.rc ] && break; sleep 60; done
  [ "$(cat $RUN/t158-build2.rc 2>/dev/null)" = 0 ] || done_ 10 "build rc=$(cat $RUN/t158-build2.rc 2>/dev/null)"
  [ "$(cat $RUN/t158-stage.rc 2>/dev/null)" = 0 ] || done_ 11 "stage rc=$(cat $RUN/t158-stage.rc 2>/dev/null)"
  log "build and stage ok; $(du -sh /home/smarton/fasth3 | cut -f1) under ~/fasth3"
  ok=0
  for i in $(seq 240); do
    health || { sleep 60; continue; }
    active=$(tt-device-mcp status 1 2>&1 | sed -n "/^RUNNING/,/^RECENT/p" | grep -w smarton)
    [ -n "$active" ] && { log "busy: $active"; sleep 60; continue; }
    ok=1; break
  done
  [ $ok = 1 ] || done_ 8 "broker never healthy or free"
  T0=$(now); echo "$T0" > $V/t0
  out=$(cd $W && tt-device-mcp run-bg "bash tmp/t158/run_e2e.sh" -w "$W" -e tmp/t158/env.yaml -t 600 2>&1); src=$?
  log "submit rc=$src: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
  [ $src = 0 ] || done_ 7 "submit failed"
  JOB=$(echo "$out" | grep -oE '[0-9]+' | tail -1); echo $JOB > $V/job_id; log "JOB=$JOB"
else
  JOB=$WATCH_JOB; T0=$(cat $V/t0)
fi
while :; do
  st=$(tt-device-mcp status -j $JOB 2>&1)
  echo "$st" | grep -qiE "^Status: *(running|queued|pending)" || break
  [ "$(uptime -s)" = "$BOOT0" ] || done_ 9 "host reboot during job $JOB"
  sleep 30
done
log "job $JOB: $(echo "$st" | grep -iE '^(Status|Exit)' | tr '\n' ' ')"
sleep 90
awk -v s="$T0" 'substr($0,1,19) >= s' $SL > $V/broker_slice.log 2>/dev/null
journalctl -k --since "$T0 UTC" --no-pager 2>/dev/null | grep -iE 'pcie|aer|tenstorrent|fatal|mce|link' > $V/journal_slice.log
e=$(errors_since "$T0"; gate_fail_since "$T0")
[ -n "$e" ] && { echo "$e" > $V/drop_evidence.log; done_ 9 "drop/error during or after job $JOB"; }
echo "$st" | grep -qiE "^Status: *completed" && echo "$st" | grep -qiE "^Exit: *0" && done_ 0 "job $JOB completed"
done_ 6 "job $JOB not completed cleanly"
