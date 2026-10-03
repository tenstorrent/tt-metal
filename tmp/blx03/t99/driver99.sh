#!/bin/bash
# t99 detached driver on blx03: worktree + build of $T99_REV -> health -> one broker job (run99.sh).
# Final marker "T99_DRIVER_DONE <stage> <rc>" in $D. stage ab rc 9 = drop/ERROR/reboot during OUR job.
S=/var/tmp/fasth3/t99/src; V=/var/tmp/fasth3/t99; D=$V/driver.log; SL=/var/log/tt-device-broker/server.log
R=/home/smarton/fasth3/tt-metal
BOOT0=$(uptime -s)
mkdir -p $V
log() { echo "$(date -u '+%F %T') $*" >> $D; }
done_() { log "T99_DRIVER_DONE $1 $2"; exit 0; }
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
log "start boot=$BOOT0"
# Own worktree + release build of the t99 C++ (fused RMSNorm + residual sum); ~/fasth3/t48 is other tasks'.
REV=${T99_REV:?}; B=/home/smarton/fasth3/t99
(set -e
 git -C $R fetch -q origin ttp/t99-t93-4-fold-resnet-residual-add-into-next
 [ -d $B ] || git -C $R worktree add --detach $B $REV
 git -C $B checkout -q --detach $REV
 for n in tracy umd tt-cluster-descriptors; do git -C $B submodule update --init --depth 1 -- tt_metal/third_party/$n; done
 cd $B && bash build_metal.sh --release --cpm-source-cache $R/.cpmcache
 test -f $B/ttnn/ttnn/_ttnn.so) > $V/build.log 2>&1; brc=$?
log "build rc=$brc $(git -C $B log -1 --format=%h 2>/dev/null)"
[ $brc = 0 ] || done_ build $brc
run_job ab 2400 bash $S/tmp/blx03/t99/run99.sh; rc=$?
done_ ab $rc
