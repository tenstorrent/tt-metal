#!/bin/bash
# t119 detached driver on blx03: health -> one broker job per step (run119.sh), one at a time.
# Final marker "T119_DRIVER_DONE <step> <rc>" in $D. rc 9 = drop/ERROR/reboot during OUR job: stop everything.
S=/var/tmp/fasth3/t119/src; V=/var/tmp/fasth3/t119; D=$V/driver.log; SL=/var/log/tt-device-broker/server.log
R=/home/smarton/fasth3/tt-metal
BOOT0=$(uptime -s)
mkdir -p $V/results
log() { echo "$(date -u '+%F %T') $*" >> $D; }
done_() { log "T119_DRIVER_DONE $1 $2"; exit 0; }
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
  JRC=$(echo "$st" | sed -nE 's/^Exit: *(-?[0-9]+).*/\1/Ip' | head -1); log "$name job exit=$JRC"
  T0=; [ "$JRC" = 0 ] && return 0; return 1
}
log "start boot=$BOOT0"
# No build step: ~/fasth3/t48 on blx03 holds 64571a953b2's C++; c4409b1fa24 and t119 change Python only.
# s0ups (128,128,1,2,4) is the t48 value: run first and last to bracket drift. A failed arm (TT_FATAL,
# reference gate) does not stop the rest; a missing reference skips the arms that need it.
STEPS=${STEPS:-"ref544 s0ups:128,128,1,2,4 s0ups:128,128,3,2,2 s0ups:64,128,3,2,4 s0ups:128,64,3,2,4 s0ups:128,128,1,2,4:rep ref1080 s4res:64,128,10,4,8 s4res:64,128,11,4,8 s4res:128,64,12,4,8"}
for ST in $STEPS; do
  N=${ST//[:,]/_}; RUN=${ST%:rep}
  [ -f $V/results/${N}_done ] && { log "$ST already done"; continue; }
  case $ST in s0ups*) [ -f $V/results/ref544_done ] || { log "$ST skipped: no ref544"; continue; } ;;
              s4res*) [ -f $V/results/ref1080_done ] || { log "$ST skipped: no ref1080"; continue; } ;; esac
  run_job $N 1700 bash $S/tmp/blx03/t119/run119.sh $RUN; rc=$?
  log "$ST rc=$rc"
  [ $rc = 9 ] && done_ $N 9
  [ $rc = 8 ] && done_ $N 8
  touch $V/results/${N}_rc$rc
  [ $rc = 0 ] && touch $V/results/${N}_done
done
done_ all 0
