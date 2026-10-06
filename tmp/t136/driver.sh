#!/bin/bash
# t136 detached driver on blx03 (from t138's): check the t134 build -> health -> ONE broker job (run_ab.sh:
# 4x8 e2e, arms OFF then LOFI) -> post-job gate -> evidence slice. Final marker "T136_DRIVER_DONE <stage>
# <rc>" in $D. rc 9 = drop/ERROR/reboot during or right after our job. Per the 2026-10-05 drop rule it then
# waits for a healthy broker and resubmits once (no duplicate if the broker re-queued it); two drops in a row
# end with rc 9. A blx03 reboot kills this driver too: the next agent run picks up from driver.log.
W=/home/smarton/fasth3/t134; V=/var/tmp/fasth3/t136; D=$V/driver.log; SL=/var/log/tt-device-broker/server.log
R=/home/smarton/fasth3/tt-metal; BUILT=e146b42f0e
BOOT0=$(uptime -s)
mkdir -p $V
log() { echo "$(date -u '+%F %T') $*" >> $D; }
done_() { log "T136_DRIVER_DONE $1 $2"; exit 0; }
now() { date -u '+%F %T'; }
errors_since() { awk -v s="$1" 'substr($0,1,19) > s && /\| ERROR \|/' $SL; }
OKRE=': OK|device healthy; no reset needed|heartbeat: HEALTHY'
BADRE='[|] ERROR [|]|ESCALATE|RECOVER'
gate_fail_since() { awk -v s="$1" -v ok="$OKRE" -v bad="$BADRE" 'substr($0,1,19) > s && ($0 ~ bad || (/HEALTH-GATE/ && $0 !~ ok))' $SL; }
health() {
  systemctl is-active -q tt-device-broker || { log "health: broker inactive"; return 1; }
  st=$(tt-device-mcp status 1 2>&1) || { log "health: status failed"; return 1; }
  echo "$st" | sed -n '/^RUNNING/,/^QUEUED/p' | grep -qiE 'health-gate|fabric-check|recover|upgrade' && { log "health: broker gate/recovery running"; return 1; }
  last=$(grep -E "HEALTH-GATE|$OKRE|ESCALATE|RECOVER|[|] ERROR [|]" $SL | tail -1)
  if echo "$last" | grep -qE "$OKRE" && ! echo "$last" | grep -qE "$BADRE"; then return 0; fi
  log "health: last event not healthy: ${last:0:200}"; return 1
}
wait_health() { for i in $(seq 120); do health && return 0; sleep 60; done; return 1; }
evidence() {
  awk -v s="$1" 'substr($0,1,19) >= s' $SL > $V/broker_slice_$2.log 2>/dev/null
  journalctl -k --since "$1 UTC" --no-pager 2>/dev/null | grep -iE 'pcie|aer|tenstorrent|fatal|mce|link' > $V/journal_slice_$2.log
}
active() { tt-device-mcp status -j $1 2>&1 | grep -qiE "^Status: *(running|queued|pending)"; }
watch_job() {  # $1 attempt; polls $JOB to the end, then the post-job gate; T0 = our submit time
  while active $JOB; do
    [ "$(uptime -s)" = "$BOOT0" ] || return 9
    sleep 20
  done
  st=$(tt-device-mcp status -j $JOB 2>&1)
  log "job $JOB status: $(echo "$st" | grep -iE '^(Status|Exit)' | tr '\n' ' ')"
  sleep 90   # let the broker's post-job gate land
  evidence "$T0" $1
  e=$(errors_since "$T0"; gate_fail_since "$T0")
  [ -n "$e" ] && { log "job $JOB DROP/ERROR during or after our job: ${e:0:600}"; return 9; }
  [ "$(uptime -s)" = "$BOOT0" ] || { log "reboot"; return 9; }
  echo "$st" | grep -qiE '^Status: *completed' || return 6
  return 0
}
submit() {
  wait_health || { log "broker never healthy"; return 8; }
  for i in $(seq 120); do
    out=$(cd $R && tmp/blx03/submit.sh 3600 bash $V/run_ab.sh 2>&1); src=$?
    [ $src = 75 ] && { sleep 60; continue; }; break
  done
  T0=$(now)
  log "submit rc=$src: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
  [ $src = 0 ] || return 7
  JOB=$(echo "$out" | tail -1); log "JOB=$JOB"; echo $JOB >> $V/job_ids
}
log "start boot=$BOOT0"
[ -x $R/tmp/blx03/submit.sh ] && [ -f $R/tmp/blx03/env.yaml ] && [ -f $V/run_ab.sh ] || done_ setup_submit 1
cur=$(git -C $W rev-parse --short=10 HEAD)
log "tree $W HEAD $cur status: $(git -C $W status --porcelain -uno | head -5 | tr '\n' ' ')"
[ "$cur" = "$BUILT" ] || done_ tree_moved 1
grep -q "SETUP134_DONE rc=0" ~/fasth3/t134-setup.log || done_ build 1
if [ -n "$WATCH_JOB" ]; then JOB=$WATCH_JOB; T0=$WATCH_T0; log "watch-only JOB=$JOB since $T0"; watch_job w; rc=$?
else submit || done_ submit $?; watch_job 1; rc=$?
fi
if [ $rc = 9 ]; then
  log "drop 1; waiting for a healthy broker before one rerun"
  wait_health || done_ e2e_drop 9
  if active $JOB; then log "broker re-queued $JOB itself; watching it, no duplicate"; T0=$(now)
  else submit || done_ submit $?
  fi
  watch_job 2; rc=$?
  [ $rc = 9 ] && { log "second drop in a row: config skipped"; done_ e2e_drop2 9; }
fi
done_ e2e $rc
