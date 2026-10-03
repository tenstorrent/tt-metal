#!/bin/bash
# t115 detached driver on blx03: health -> one broker job per layer (run114.sh), one at a time.
# Final marker "T115_DRIVER_DONE <stage> <rc>" in $D. stage <blocking> rc 9 = drop/ERROR/reboot during OUR job.
S=/var/tmp/fasth3/t115/src; V=/var/tmp/fasth3/t115; D=$V/driver.log; SL=/var/log/tt-device-broker/server.log
R=/home/smarton/fasth3/tt-metal
BOOT0=$(uptime -s)
# DRY_RUN=1 (any host, no device, no broker): print the job commands and exit.
DRY_RUN=${DRY_RUN:-0}
[ "$DRY_RUN" = 1 ] || mkdir -p $V/results
log() { echo "$(date -u '+%F %T') $*" >> $D; }
done_() { log "T115_DRIVER_DONE $1 $2"; exit 0; }
now() { date -u '+%F %T'; }
errors_since() { awk -v s="$1" 'substr($0,1,19) > s && /\| ERROR \|/' $SL; }
# A line is healthy if it carries an OK marker and no ERROR/ESCALATE/RECOVER; bad if it has one of those
# or is a HEALTH-GATE without an OK marker. Heartbeat/no-reset lines are healthy events, not alarms.
# A broker recovery that ends in "reset complete + health verified" is logged at ERROR level but is healthy.
OKRE=': OK|device healthy; no reset needed|heartbeat: HEALTHY|reset complete \+ health verified'
BADRE='[|] ERROR [|]|ESCALATE|RECOVER'
gate_fail_since() { awk -v s="$1" -v ok="$OKRE" -v bad="$BADRE" 'substr($0,1,19) > s && ($0 ~ bad || (/HEALTH-GATE/ && $0 !~ ok))' $SL; }
health() {  # pre-submit: broker up and answering, last health/recovery event is healthy
  systemctl is-active -q tt-device-broker || { log "health: broker inactive"; return 1; }
  tt-device-mcp status 1 > /dev/null 2>&1 || { log "health: status failed"; return 1; }
  last=$(grep -E "HEALTH-GATE|$OKRE|ESCALATE|RECOVER|[|] ERROR [|]" $SL | tail -1)
  echo "$last" | grep -q "reset complete + health verified" && return 0
  if echo "$last" | grep -qE "$OKRE" && ! echo "$last" | grep -qE "$BADRE"; then return 0; fi
  log "health: last event not healthy: ${last:0:200}"; return 1
}
wait_health() { for i in $(seq 120); do health && return 0; sleep 60; done; return 1; }
run_job() {  # $1 name, $2 timeout, $3.. cmd; sets JOB, JRC; returns 9 on drop during our job
  local name=$1 t=$2; shift 2
  wait_health || { log "$name: broker never healthy"; return 8; }
  for i in $(seq 60); do
    out=$(cd $R && tmp/blx03/submit.sh $t "$@" 2>&1); src=$?
    [ $src = 75 ] && { sleep 60; continue; }; break
  done
  T0=; T0S=$(now)
  log "$name submit rc=$src: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
  [ $src = 0 ] || return 7
  JOB=$(echo "$out" | tail -1); log "$name JOB=$JOB"
  while :; do
    st=$(tt-device-mcp status -j $JOB 2>&1)
    echo "$st" | grep -qiE "^Status: *(running|queued|pending)" || break
    # Errors count against our job only from when it runs; a drop while it is queued is another tenant's.
    [ -z "$T0" ] && echo "$st" | grep -qiE "^Status: *running" && { T0=$(now); log "$name running since $T0"; }
    [ "$(uptime -s)" = "$BOOT0" ] || return 9
    sleep 20
  done
  T1=$(now); T0=${T0:-$T0S}   # never seen running: count from submit (conservative)
  log "$name status: $(echo "$st" | grep -iE '^(Status|Exit)' | tr '\n' ' ')"
  sleep 90   # let the broker's post-job gate land
  e=$(errors_since "$T0"; gate_fail_since "$T0")
  [ -n "$e" ] && { log "$name DROP/ERROR during or after our job: ${e:0:600}"; return 9; }
  [ "$(uptime -s)" = "$BOOT0" ] || { log "$name reboot"; return 9; }
  T0=; return 0
}
# Job 484 combos 141-150 (exact_s2_res, C_in=C_out=512, halo mode), in sweep order. 143 (64,128,6,8,8) and
# 144 (64,128,6,16,4) are left out: they get no L1 prefetch shard (direct reader, halo dropped), and the
# harness now drops them before launch. Stop at the first drop (rc 9), at the first hung job (rc 6), or when
# the broker never turns healthy (rc 8).
COMBOS=${COMBOS:-"64,64,3,8,8 64,64,3,16,4 64,32,3,4,4 64,32,3,8,2 64,128,5,4,4 64,128,5,8,2 64,128,7,4,4 64,128,7,8,2"}
if [ "$DRY_RUN" = 1 ]; then
  for C in $COMBOS; do echo "job: tmp/blx03/submit.sh 700 bash $S/tmp/blx03/t115/run115.sh $C"; done
  exit 0
fi
log "start boot=$BOOT0"
for C in $COMBOS; do
  TAG=${C//,/_}
  [ -f $V/results/${TAG}_done ] && { log "$C already done"; continue; }
  run_job $C 700 bash $S/tmp/blx03/t115/run115.sh $C; rc=$?
  log "$C rc=$rc"
  [ $rc = 9 ] && done_ $C 9
  [ $rc = 8 ] && done_ $C 8
  echo "$st" | grep -qiE "^Status: *hung" && done_ $C 6
  [ $rc = 0 ] && touch $V/results/${TAG}_done
done
done_ all 0
