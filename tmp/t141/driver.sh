#!/bin/bash
# t141 detached driver on blx03 (from the t138 driver): health -> put the t48 tree on TIP (c4409b1fa24 + the
# test-only LTX_E2E_SEEDS knob) -> ONE broker job (run_e2e.sh: 4x8 e2e, warmup + gen#0 + 5 warm seed gens)
# -> post-job gate -> evidence slice -> tree back to BASE. Waits while another project job holds the device.
# Final marker "T141_DRIVER_DONE <stage> <rc>" in $D. rc 9 = drop/ERROR/reboot during or right after OUR job:
# stop all device work. Submits nothing else.
W=/home/smarton/fasth3/t48; V=/var/tmp/fasth3/t141; D=$V/driver.log; SL=/var/log/tt-device-broker/server.log
R=/home/smarton/fasth3/tt-metal; TIP=9f2b28b7663; BASE=c4409b1fa24; BUILT=64571a953b2
BOOT0=$(uptime -s)
mkdir -p $V
log() { echo "$(date -u '+%F %T') $*" >> $D; }
done_() { log "T141_DRIVER_DONE $1 $2"; exit 0; }
now() { date -u '+%F %T'; }
errors_since() { awk -v s="$1" 'substr($0,1,19) > s && /\| ERROR \|/' $SL; }
OKRE=': OK|device healthy; no reset needed|heartbeat: HEALTHY'
BADRE='[|] ERROR [|]|ESCALATE|RECOVER'
gate_fail_since() { awk -v s="$1" -v ok="$OKRE" -v bad="$BADRE" 'substr($0,1,19) > s && ($0 ~ bad || (/HEALTH-GATE/ && $0 !~ ok))' $SL; }
health() {
  systemctl is-active -q tt-device-broker || { log "health: broker inactive"; return 1; }
  st=$(tt-device-mcp status 1 2>&1) || { log "health: status failed"; return 1; }
  echo "$st" | sed -n '/^RUNNING/,/^QUEUED/p' | grep -qiE '🔧|health-gate|fabric-check|recover|upgrade|bridge-reset|power-cycle' && { log "health: broker gate/recovery running"; return 1; }
  last=$(grep -E "HEALTH-GATE|$OKRE|ESCALATE|RECOVER|[|] ERROR [|]" $SL | tail -1)
  if echo "$last" | grep -qE "$OKRE" && ! echo "$last" | grep -qE "$BADRE"; then return 0; fi
  log "health: last event not healthy: ${last:0:200}"; return 1
}
wait_health() { for i in $(seq 120); do health && return 0; sleep 60; done; return 1; }
evidence() {  # broker log + kernel journal since our submit, kept for the hand-off
  awk -v s="$1" 'substr($0,1,19) >= s' $SL > $V/broker_slice.log 2>/dev/null
  journalctl -k --since "$1 UTC" --no-pager 2>/dev/null | grep -iE 'pcie|aer|tenstorrent|fatal|mce|link' > $V/journal_slice.log
}
run_job() {
  local name=$1 t=$2; shift 2
  src=8
  for i in $(seq 240); do  # health right before every attempt: a busy wait can end in a drop
    health || { src=8; sleep 60; continue; }
    out=$(cd $R && tmp/blx03/submit.sh $t "$@" 2>&1); src=$?
    [ $src = 75 ] && { sleep 60; continue; }; break
  done
  [ $src = 8 ] && { log "$name: broker never healthy"; return 8; }
  T0=$(now)
  log "$name submit rc=$src: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
  [ $src = 0 ] || return 7
  JOB=$(echo "$out" | tail -1); log "$name JOB=$JOB"; echo $JOB > $V/job_id
  watch_job "$name"
}
watch_job() {  # polls $JOB to the end, then the post-job gate; T0 = our submit time
  local name=$1
  while :; do
    st=$(tt-device-mcp status -j $JOB 2>&1)
    echo "$st" | grep -qiE "^Status: *(running|queued|pending)" || break
    [ "$(uptime -s)" = "$BOOT0" ] || return 9
    sleep 20
  done
  log "$name status: $(echo "$st" | grep -iE '^(Status|Exit)' | tr '\n' ' ')"
  sleep 90   # let the broker's post-job gate land
  evidence "$T0"
  e=$(errors_since "$T0"; gate_fail_since "$T0")
  [ -n "$e" ] && { log "$name DROP/ERROR during or after our job: ${e:0:600}"; return 9; }
  [ "$(uptime -s)" = "$BOOT0" ] || { log "$name reboot"; return 9; }
  return 0
}
log "start boot=$BOOT0"
[ -x $R/tmp/blx03/submit.sh ] && [ -f $R/tmp/blx03/env.yaml ] || done_ setup_submit 1
cur=$(git -C $W rev-parse --short=11 HEAD)
if [ "$cur" != "$TIP" ]; then
  git -C $W cat-file -e $TIP^{commit} 2>/dev/null || git -C $W fetch -q origin ttp/t141-e2e-5seed >> $D 2>&1
  if git -C $W cat-file -e $TIP^{commit} 2>/dev/null && [ -z "$(git -C $W status --porcelain -uno)" ] \
     && [ -z "$(git -C $W diff --name-only $cur $TIP | grep -vE '^models/tt_dit/tests/')" ]; then
    git -C $W checkout -q --detach $TIP >> $D 2>&1 && log "tree $cur -> $TIP (test-only delta)"
  else
    log "tree stays at $cur (tip $TIP not reachable, tree dirty, or non-test delta)"; done_ tree 1
  fi
fi
log "tree HEAD $(git -C $W rev-parse --short=11 HEAD) status: $(git -C $W status --porcelain -uno | head -5 | tr '\n' ' ')"
if [ -n "$WATCH_JOB" ]; then  # re-attach to an already submitted job: never submit another
  JOB=$WATCH_JOB; T0=$WATCH_T0; log "watch-only JOB=$JOB since $T0"; watch_job e2e; rc=$?
else
  run_job e2e 2400 env W=$W bash $V/run_e2e.sh; rc=$?
fi
if [ "$(git -C $W rev-parse --short=11 HEAD)" = "$TIP" ] && [ -z "$(git -C $W status --porcelain -uno)" ]; then
  git -C $W checkout -q --detach $BASE >> $D 2>&1 && log "tree back to $BASE"
fi
done_ e2e $rc
