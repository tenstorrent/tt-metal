#!/bin/bash
# t115/#122 detached driver on blx03: health -> one broker job per blocking (run115.sh), one at a time.
# Final marker "T115_DRIVER_DONE <stage> <rc>" in $D; per-combo outcome lines in $V/outcomes.txt.
# Drop rules (user 2026-10-05): a drop/reboot/fabric error does not stop the bisect. Wait for broker health,
# rerun the combo; skip it after 2 drops in a row. A hung combo is a result (no rerun); stop (rc 6) only if
# the broker is not healthy again within 30 min after it. rc 8 = broker never healthy.
S=/var/tmp/fasth3/t115/src; V=/var/tmp/fasth3/t115; D=$V/driver.log; SL=/var/log/tt-device-broker/server.log
R=/home/smarton/fasth3/tt-metal; O=$V/outcomes.txt
# DRY_RUN=1 (any host, no device, no broker): print the job commands and exit.
DRY_RUN=${DRY_RUN:-0}
[ "$DRY_RUN" = 1 ] || { mkdir -p $V/results; echo $$ > $V/driver.pid; }
log() { echo "$(date -u '+%F %T') $*" >> $D; }
done_() { log "T115_DRIVER_DONE $1 $2"; exit 0; }
now() { date -u '+%F %T'; }
OKRE=': OK|device healthy; no reset needed|heartbeat: HEALTHY|reset complete \+ health verified'
BADRE='[|] ERROR [|]|ESCALATE|RECOVER|off the bus'
bad_since() { awk -v s="$1" -v ok="$OKRE" -v bad="$BADRE" 'substr($0,1,19) > s && ($0 ~ bad || (/HEALTH-GATE/ && $0 !~ ok))' $SL; }
health() {  # broker up and answering, nothing HELD, last health/recovery event is healthy
  systemctl is-active -q tt-device-broker || { log "health: broker inactive"; return 1; }
  st=$(tt-device-mcp status 1 2>&1) || { log "health: status failed"; return 1; }
  echo "$st" | grep -q "device HELD" && { log "health: device held"; return 1; }
  last=$(grep -E "HEALTH-GATE|$OKRE|ESCALATE|RECOVER|[|] ERROR [|]" $SL | tail -1)
  echo "$last" | grep -q "reset complete + health verified" && return 0
  if echo "$last" | grep -qE "$OKRE" && ! echo "$last" | grep -qE "$BADRE"; then return 0; fi
  log "health: last event not healthy: ${last:0:200}"; return 1
}
wait_health() { for i in $(seq ${1:-480}); do health && return 0; sleep 60; done; return 1; }
# run_job <name> <timeout> <cmd..>: sets JOB JST JEXIT BAD; returns 0 ran, 7 submit failed, 8 never healthy
run_job() {
  local name=$1 t=$2; shift 2
  wait_health || { log "$name: broker never healthy"; return 8; }
  for i in $(seq 240); do
    out=$(cd $R && tmp/blx03/submit.sh $t "$@" 2>&1); src=$?
    [ $src = 75 ] && { sleep 60; continue; }; break
  done
  T0=$(now); TR=; BOOT0=$(uptime -s)
  log "$name submit rc=$src: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
  [ $src = 0 ] || return 7
  JOB=$(echo "$out" | tail -1); log "$name JOB=$JOB"
  while :; do
    st=$(tt-device-mcp status -j $JOB 2>&1)
    echo "$st" | grep -qiE "^Status: *(running|queued|pending)" || break
    # Errors count against our job from when it runs; a drop while it is queued is another tenant's.
    [ -z "$TR" ] && echo "$st" | grep -qiE "^Status: *running" && { TR=$(now); log "$name running since $TR"; }
    sleep 20
  done
  JST=$(echo "$st" | sed -n 's/^Status: *//p'); JEXIT=$(echo "$st" | sed -n 's/^Exit: *//p')
  log "$name job $JOB status=$JST exit=$JEXIT"
  sleep 90   # let the broker's post-job gate land
  BAD=$(bad_since "${TR:-$T0}" | head -5)
  [ "$(uptime -s)" = "$BOOT0" ] || BAD="reboot; $BAD"
  return 0
}
# Job 484 combos 141-150 (exact_s2_res, C_in=C_out=512, halo mode), in sweep order, minus 143 (64,128,6,8,8)
# and 144 (64,128,6,16,4): no L1 prefetch shard (direct reader, halo dropped); the harness drops them anyway.
COMBOS=${COMBOS:-"64,64,3,8,8 64,64,3,16,4 64,32,3,4,4 64,32,3,8,2 64,128,5,4,4 64,128,5,8,2 64,128,7,4,4 64,128,7,8,2"}
if [ "$DRY_RUN" = 1 ]; then
  for C in $COMBOS; do echo "job: tmp/blx03/submit.sh 700 bash $S/tmp/blx03/t115/run115.sh $C"; done
  exit 0
fi
log "start boot=$(uptime -s)"
for C in $COMBOS; do
  TAG=${C//,/_}
  [ -f $V/results/${TAG}_done ] && { log "$C already done"; continue; }
  drops=0
  while :; do
    rm -f $V/run115_$TAG.log
    run_job $C 700 bash $S/tmp/blx03/t115/run115.sh $C; rc=$?
    [ $rc = 8 ] && done_ $C 8
    [ $rc = 7 ] && done_ $C 7
    L=$V/run115_$TAG.log
    if grep -q '^T115_EXIT=0' $L 2>/dev/null; then res=PASS
    elif grep -q '^T115_EXIT=' $L 2>/dev/null; then res="FAIL($(sed -n 's/^T115_EXIT=//p' $L | tail -1))"
    else res=HANG_OR_KILLED; fi
    [ "$JEXIT" = 124 ] || [ "$JEXIT" = 130 ] || [ "$JST" = hung ] || [ "$JST" = killed ] && res=HANG_OR_KILLED
    log "$C result=$res job=$JOB status=$JST exit=$JEXIT bad=${BAD:0:600}"
    if [ "$res" = HANG_OR_KILLED ]; then
      echo "$C HANG job=$JOB status=$JST exit=$JEXIT bad=${BAD:0:300}" >> $O
      touch $V/results/${TAG}_done
      wait_health 30 || done_ $C 6
      break
    fi
    if [ -n "$BAD" ]; then
      drops=$((drops+1)); log "$C DROP $drops during/after job $JOB: ${BAD:0:600}"
      echo "$C DROP#$drops job=$JOB result=$res at=$(now) bad=${BAD:0:300}" >> $O
      [ $drops -ge 2 ] && { echo "$C SKIPPED after 2 drops in a row" >> $O; touch $V/results/${TAG}_done; break; }
      continue
    fi
    echo "$C $res job=$JOB" >> $O; touch $V/results/${TAG}_done; break
  done
done
done_ all 0
