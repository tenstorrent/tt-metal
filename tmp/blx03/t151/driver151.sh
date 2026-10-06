#!/bin/bash
# #151 detached driver on blx03: wait for build151.sh -> broker health, no other project job -> ONE broker
# job (run151.sh). Marker "T151_DRIVER_DONE <stage> <rc>" in $D; outcome lines in $V/outcomes.txt.
# Drop rules (user 2026-10-05): a drop/reboot/fabric error during our job -> wait for health, rerun;
# stop after 2 drops in a row. A hang after a guarded blocking launched is the result (guard missed), not a drop.
V=/var/tmp/fasth3/t151; D=$V/driver.log; SL=/var/log/tt-device-broker/server.log; O=$V/outcomes.txt
R=/home/smarton/fasth3/tt-metal
mkdir -p $V; echo $$ > $V/driver.pid
log() { echo "$(date -u '+%F %T') $*" >> $D; }
done_() { log "T151_DRIVER_DONE $1 $2"; exit 0; }
now() { date -u '+%F %T'; }
OKRE=': OK|device healthy; no reset needed|heartbeat: HEALTHY|reset complete \+ health verified'
BADRE='[|] ERROR [|]|ESCALATE|RECOVER|off the bus|TRAY_DOWN'
bad_since() { awk -v s="$1" -v ok="$OKRE" -v bad="$BADRE" 'substr($0,1,19) > s && ($0 ~ bad || (/HEALTH-GATE/ && $0 !~ ok))' $SL; }
health() {  # broker up and answering, nothing HELD, no other project job, last health event healthy
  systemctl is-active -q tt-device-broker || { log "health: broker inactive"; return 1; }
  pgrep -f /opt/tt-device-broker/autoupdate.sh > /dev/null && { log "health: broker upgrade"; return 1; }
  st=$(tt-device-mcp status 1 2>&1) || { log "health: status failed"; return 1; }
  echo "$st" | grep -q "device HELD" && { log "health: device held"; return 1; }
  echo "$st" | sed '/^RECENT/,$d' | grep -q "smarton" && { log "health: other project job queued/running"; return 1; }
  last=$(grep -E "HEALTH-GATE|$OKRE|ESCALATE|RECOVER|[|] ERROR [|]" $SL | tail -1)
  echo "$last" | grep -q "reset complete + health verified" && return 0
  if echo "$last" | grep -qE "$OKRE" && ! echo "$last" | grep -qE "$BADRE"; then return 0; fi
  log "health: last event not healthy: ${last:0:200}"; return 1
}
wait_health() { for i in $(seq ${1:-480}); do health && return 0; sleep 60; done; return 1; }
log "start boot=$(uptime -s)"
for i in $(seq 120); do grep -q BUILD151_DONE $V/build.log 2>/dev/null && break; sleep 30; done
brc=$(sed -n 's/^BUILD151_DONE rc=//p' $V/build.log | tail -1); log "build rc=${brc:-none}"
[ "$brc" = 0 ] || done_ build ${brc:-timeout}
drops=0
while :; do
  wait_health || done_ health 8
  for i in $(seq 240); do
    out=$(cd $R && tmp/blx03/submit.sh 700 bash ~/fasth3/t151drv/run151.sh 2>&1); src=$?
    [ $src = 75 ] && { sleep 60; continue; }; break
  done
  T0=$(now); TR=; BOOT0=$(uptime -s)
  log "submit rc=$src: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
  [ $src = 0 ] || done_ submit $src
  JOB=$(echo "$out" | tail -1); log "JOB=$JOB"; echo "$JOB" >> $V/job_ids
  while :; do
    st=$(tt-device-mcp status -j $JOB 2>&1)
    echo "$st" | grep -qiE "^Status: *(running|queued|pending)" || break
    [ -z "$TR" ] && echo "$st" | grep -qiE "^Status: *running" && { TR=$(now); log "running since $TR"; }
    sleep 20
  done
  JST=$(echo "$st" | sed -n 's/^Status: *//p'); JEXIT=$(echo "$st" | sed -n 's/^Exit: *//p')
  log "job $JOB status=$JST exit=$JEXIT"
  sleep 90   # the broker's post-job gate
  BAD=$(bad_since "${TR:-$T0}" | head -8)
  [ "$(uptime -s)" = "$BOOT0" ] || BAD="reboot; $BAD"
  L=$V/run151.log
  # Last blocking launched with no FAIL/timing line after it = the job stopped inside that blocking.
  lastl=$(grep -E '^  (launching|FAIL) |us$' $L 2>/dev/null | tail -1)
  echo "$(now) job=$JOB status=$JST exit=$JEXIT exitline=$(grep '^T151_EXIT=' $L | tail -1) last=[$lastl]" >> $O
  if grep -q '^T151_EXIT=' $L; then
    [ -n "$BAD" ] && echo "  broker events during job: ${BAD:0:600}" >> $O
    cp $L $V/run151_job$JOB.log; done_ ran 0
  fi
  if echo "$lastl" | grep -qE 'launching \(64,(128|32),5,4,4\)'; then
    echo "  HANG inside a guarded blocking: guard missed. events: ${BAD:0:600}" >> $O
    cp $L $V/run151_job$JOB.log; done_ hang 0
  fi
  cp $L $V/run151_job$JOB.log 2>/dev/null
  drops=$((drops+1)); log "DROP $drops job $JOB: ${BAD:0:600}"
  echo "  DROP#$drops events: ${BAD:0:600}" >> $O
  [ $drops -ge 2 ] && done_ two_drops 9
done
