#!/bin/bash
# t133 detached driver on blx03: wait for blx03_setup133.sh -> health -> ONE broker job (run133.sh, A/B of #56023)
# -> CPU compare (cmp133.py). Final marker "T133_DRIVER_DONE <stage> <rc>" in $D. Stops (no further submits) on any
# broker ERROR or reboot during OUR job.
# Launch from g15blx02 (after the setup script was started):
#   ssh g14blx03 mkdir -p fasth3/t133drv && scp tmp/t133/driver.sh g14blx03:fasth3/t133drv/driver.sh
#   tt-project/harness/templates/blx03-launch.sh t133 /home/smarton/fasth3/t133drv/driver.sh
W=/home/smarton/fasth3/t133b; V=/var/tmp/fasth3/t133; D=$V/driver.log; SL=/var/log/tt-device-broker/server.log
R=/home/smarton/fasth3/tt-metal
BOOT0=$(uptime -s)
mkdir -p $V
log() { echo "$(date -u '+%F %T') $*" >> $D; }
done_() { log "T133_DRIVER_DONE $1 $2"; exit 0; }
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
run_job() {  # $1 name, $2 timeout, $3.. cmd; sets JOB, JRC; returns 9 on drop during our job
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
until grep -q "SETUP133_DONE" ~/fasth3/t133-setup.log 2>/dev/null; do sleep 60; done
src=$(grep -o "SETUP133_DONE rc=[0-9]*" ~/fasth3/t133-setup.log | tail -1 | cut -d= -f2)
log "setup rc=$src"; [ "$src" = 0 ] || done_ setup $src
run_job ab 2400 bash $W/tmp/t133/run133.sh; rc=$?
[ $rc = 0 ] || done_ ab $rc
( source $R/python_env/bin/activate && cd $W && python tmp/t133/cmp133.py $V ) >> $D 2>&1
grep -q "^T133_EXIT=0" $V/run133.log || done_ ab_exit 1
done_ ab 0
