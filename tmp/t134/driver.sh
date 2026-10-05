#!/bin/bash
# t134 detached driver on blx03: setup check -> health -> job REF (run134.sh ref, t48 build) -> health ->
# job NEW (run134.sh new, t134 build: OFF and LOFI). Final marker "T134_DRIVER_DONE <stage> <rc>" in $D.
# Stops (no further submits) on any broker ERROR, failed health gate or reboot during OUR job (rc 9):
# that is the charter's stop-all-device-work case.
W=/home/smarton/fasth3/t134; V=/var/tmp/fasth3/t134; D=$V/driver.log; SL=/var/log/tt-device-broker/server.log
R=/home/smarton/fasth3/tt-metal; REF=/home/smarton/fasth3/t48; BASE=c4409b1fa24
BOOT0=$(uptime -s)
mkdir -p $V
log() { echo "$(date -u '+%F %T') $*" >> $D; }
done_() {
  grep -h "T134_\|PASSED\|FAILED" $V/run134_*.log 2>/dev/null | grep -v T134_START > $V/summary.txt
  rm -f $V/ref_*.pt; rm -rf $V/run
  log "T134_DRIVER_DONE $1 $2"; exit 0
}
now() { date -u '+%F %T'; }
errors_since() { awk -v s="$1" 'substr($0,1,19) > s && /\| ERROR \|/' $SL; }
# A line is healthy if it carries an OK marker and no ERROR/ESCALATE/RECOVER; bad if it has one of those
# or is a HEALTH-GATE without an OK marker. Heartbeat/no-reset lines are healthy events, not alarms.
OKRE=': OK|device healthy; no reset needed|heartbeat: HEALTHY'
BADRE='[|] ERROR [|]|ESCALATE|RECOVER'
gate_fail_since() { awk -v s="$1" -v ok="$OKRE" -v bad="$BADRE" 'substr($0,1,19) > s && ($0 ~ bad || (/HEALTH-GATE/ && $0 !~ ok))' $SL; }
health() {  # pre-submit: broker up and answering, last health/recovery event is healthy
  systemctl is-active -q tt-device-broker || { log "health: broker inactive"; return 1; }
  tt-device-mcp status 1 > /dev/null 2>&1 || { log "health: status failed"; return 1; }
  last=$(grep -E "HEALTH-GATE|$OKRE|ESCALATE|RECOVER|[|] ERROR [|]" $SL | tail -1)
  if echo "$last" | grep -qE "$OKRE" && ! echo "$last" | grep -qE "$BADRE"; then return 0; fi
  log "health: last event not healthy: ${last:0:200}"; return 1
}
wait_health() { for i in $(seq 120); do health && return 0; sleep 60; done; return 1; }
run_job() {  # $1 name, $2 timeout, $3.. cmd; sets JOB; returns 9 on drop/reboot during our job
  local name=$1 t=$2; shift 2
  wait_health || { log "$name: broker never healthy"; return 8; }
  for i in $(seq 60); do
    out=$(cd $R && tmp/blx03/submit.sh $t "$@" 2>&1); src=$?
    [ $src = 75 ] && { sleep 60; continue; }; break
  done
  T0=$(now)
  log "$name submit rc=$src: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
  [ $src = 0 ] || return 7
  JOB=$(echo "$out" | tail -1); log "$name JOB=$JOB"
  while :; do
    st=$(tt-device-mcp status -j $JOB 2>&1)
    echo "$st" | grep -qiE "^Status: *(running|queued|pending)" || break
    [ "$(uptime -s)" = "$BOOT0" ] || return 9
    sleep 20
  done
  T1=$(now); T0=${T0:-$T1}
  log "$name status: $(echo "$st" | grep -iE '^(Status|Exit)' | tr '\n' ' ')"
  sleep 90   # let the broker's post-job gate land
  e=$(errors_since "$T0"; gate_fail_since "$T0")
  [ -n "$e" ] && { log "$name DROP/ERROR during or after our job: ${e:0:600}"; return 9; }
  [ "$(uptime -s)" = "$BOOT0" ] || { log "$name reboot"; return 9; }
  T0=; return 0
}
log "start boot=$BOOT0"
for i in $(seq 180); do grep -q "SETUP134_DONE" ~/fasth3/t134-setup.log 2>/dev/null && break; sleep 60; done
src=$(grep -o "SETUP134_DONE rc=[0-9]*" ~/fasth3/t134-setup.log 2>/dev/null | tail -1 | cut -d= -f2)
log "setup rc=${src:-none}"; [ "$src" = 0 ] || done_ setup ${src:-1}
# REF must be this branch's base in code; a t48 tree that moved on makes REF vs OFF compare more than the
# cherry-picks. The run goes ahead either way (OFF vs LOFI stays valid); the drift is logged for the reader.
RREV=$(git -C $REF rev-parse HEAD); log "trees: t134=$(git -C $W rev-parse --short HEAD) t48=${RREV:0:11} base=$BASE"
drift=$(git -C $W diff --stat $RREV $BASE -- ttnn tt_metal models/tt_dit/models models/tt_dit/parallel \
  models/tt_dit/utils models/tt_dit/layers models/tt_dit/tests/models/ltx/test_transformer_ltx.py 2>&1 | tail -1)
[ -n "$drift" ] && log "T134_REF_DRIFT t48 differs from $BASE: $drift"
run_job ref 1200 bash $W/tmp/t134/run134.sh ref; rc=$?
[ $rc = 9 ] && done_ ref_drop 9
[ $rc = 0 ] || log "ref job rc=$rc; continuing to the new arms (OFF vs LOFI stays valid without REF)"
grep -q "^T134_EXIT_ref=0" $V/run134_ref.log 2>/dev/null || log "ref run did not exit 0"
run_job new 1800 bash $W/tmp/t134/run134.sh new; rc=$?
[ $rc = 9 ] && done_ new_drop 9
[ $rc = 0 ] || done_ new_job $rc
grep -q "^T134_EXIT_new=0" $V/run134_new.log || done_ new_exit 1
done_ new 0
