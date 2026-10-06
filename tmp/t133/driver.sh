#!/bin/bash
# t133 detached driver on blx03: wait for blx03_setup133.sh -> two short 4x8 broker jobs, one A/B pair each
# (j1: A B, j2: B A) -> CPU compare (cmp133.py). Final marker "T133_DRIVER_DONE <stage> <rc>" in $D.
# Drops: never reset; wait for the broker's health check, then rerun the job (or keep waiting on it if the broker
# re-queued it). A job that drops twice in a row is skipped. Every drop is logged as "DROP". A job whose
# run133_<j>.log already ends with T133_EXIT=0 is not rerun, so a relaunch after a reboot resumes.
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
active() { tt-device-mcp status -j $1 2>&1 | grep -qiE "^Status: *(running|queued|pending)"; }
health() {  # pre-submit: broker up and answering, nothing held, last health/recovery event is healthy
  systemctl is-active -q tt-device-broker || { log "health: broker inactive"; return 1; }
  st=$(tt-device-mcp status 1 2>&1) || { log "health: status failed"; return 1; }
  echo "$st" | sed -n '/^RUNNING/,/^QUEUED/p' | grep -qiE "HELD|degraded|recovery" && { log "health: broker gate/recovery running"; return 1; }
  last=$(grep -E "HEALTH-GATE|$OKRE|ESCALATE|RECOVER|[|] ERROR [|]" $SL | tail -1)
  if echo "$last" | grep -qE "$OKRE" && ! echo "$last" | grep -qE "$BADRE"; then return 0; fi
  log "health: last event not healthy: ${last:0:200}"; return 1
}
wait_health() { for i in $(seq 720); do health && return 0; sleep 60; done; return 1; }
wait_job() {  # $1 job id
  while active $1; do
    [ "$(uptime -s)" = "$BOOT0" ] || return 9
    sleep 20
  done
}
run_job() {  # $1 job tag, $2.. arms; returns 0 ok, 9 drop/error during or right after our job, else failure
  local j=$1; shift
  if [ -n "$JOB" ] && active $JOB; then
    log "$j: broker re-queued job $JOB; waiting on it"
  else
    wait_health || { log "$j: broker never healthy"; return 8; }
    for i in $(seq 720); do
      out=$(cd $R && tmp/blx03/submit.sh 2400 bash $W/tmp/t133/run133.sh $j "$@" 2>&1)
      src=$?
      [ $src = 75 ] && { sleep 60; continue; }
      break
    done
    log "$j submit rc=$src: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
    [ $src = 0 ] || return 7
    JOB=$(echo "$out" | tail -1); log "$j JOB=$JOB"
  fi
  T0=$(now)
  wait_job $JOB || { log "$j DROP: blx03 rebooted during job $JOB"; return 9; }
  st=$(tt-device-mcp status -j $JOB 2>&1)
  log "$j status: $(echo "$st" | grep -iE '^(Status|Exit)' | tr '\n' ' ')"
  sleep 90  # let the broker's post-job gate land
  e=$(errors_since "$T0"; gate_fail_since "$T0")
  ok=0; grep -q "^T133_EXIT=0" $V/run133_$j.log 2>/dev/null && ok=1
  if [ -n "$e" ]; then
    log "$j DROP (job $JOB, results_ok=$ok): ${e:0:800}"
    [ $ok = 1 ] && return 0
    return 9
  fi
  [ $ok = 1 ] && return 0
  log "$j failed: $(grep -E 'T133_ARM_EXIT|T133_EXIT' $V/run133_$j.log 2>/dev/null | tr '\n' ' ')"
  return 1
}

log "start boot=$BOOT0"
until grep -q "SETUP133_DONE" ~/fasth3/t133-setup.log 2>/dev/null; do sleep 60; done
src=$(grep -o "SETUP133_DONE rc=[0-9]*" ~/fasth3/t133-setup.log | tail -1 | cut -d= -f2)
log "setup rc=$src"; [ "$src" = 0 ] || done_ setup $src
skipped=
for spec in "j1 A B" "j2 B A"; do
  set -- $spec; j=$1
  grep -q "^T133_EXIT=0" $V/run133_$j.log 2>/dev/null && { log "$j already done"; continue; }
  JOB=; drops=0
  while :; do
    run_job $spec; rc=$?
    [ $rc = 0 ] && break
    [ $rc = 9 ] || done_ $j $rc
    drops=$((drops + 1))
    [ $drops -ge 2 ] && { log "$j skipped after 2 drops in a row"; skipped="$skipped $j"; break; }
  done
done
( source $R/python_env/bin/activate && cd $W && python tmp/t133/cmp133.py $V ) >> $D 2>&1
[ -z "$skipped" ] || done_ "ab_skipped:${skipped# }" 1
done_ ab 0
