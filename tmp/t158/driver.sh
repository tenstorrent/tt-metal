#!/bin/bash
# t158 driver on g15blx02: wait for build + cache staging, then ONE broker job (warm 4x8 1080p e2e,
# seed 0), then the post-job drop check. Health logic copied from tmp/t141/driver.sh.
# Marker: $V/DRIVER.done holds "<exit code> <reason>". WATCH_JOB=<id> re-attaches without submitting.
# Broker timeout is capped at 600 s by a local hook, and job 099 compiled 2569 kernels in-window (530 s),
# so a cold cache first gets a capture-only job, then an off-device kernel_prewarm compile, then the real job.
W=/home/smarton/fasth3/tt-metal/tt-project/worktrees/t158
RUN=/home/smarton/fasth3/tt-metal/tt-project/state/runs/645
V=/home/smarton/fasth3/tt-metal/tt-project/data/g15/driver; mkdir -p $V
SL=/var/log/tt-device-broker/server.log; D=$V/driver.log; BOOT0=$(uptime -s)
log() { echo "$(date -u '+%F %T') $*" >> $D; }
done_() { log "T158_DRIVER_DONE $1 $2"; echo "$1 $2" > $V/DRIVER.done; exit 0; }
now() { date -u '+%F %T'; }
OKRE=': OK|device healthy; no reset needed|heartbeat: HEALTHY'
BADRE='[|] ERROR [|]|ESCALATE|RECOVER'
errors_since() { awk -v s="$1" 'substr($0,1,19) > s && /\| ERROR \|/' $SL; }
gate_fail_since() { awk -v s="$1" -v ok="$OKRE" -v bad="$BADRE" 'substr($0,1,19) > s && ($0 ~ bad || (/HEALTH-GATE/ && $0 !~ ok))' $SL; }
health() {
  systemctl is-active -q tt-device-broker || { log "health: broker inactive"; return 1; }
  st=$(tt-device-mcp status 1 2>&1) || { log "health: status failed"; return 1; }
  echo "$st" | sed -n '/^RUNNING/,/^QUEUED/p' | grep -qiE '🔧|health-gate|fabric-check|recover|upgrade|bridge-reset|power-cycle' && { log "health: broker gate/recovery running"; return 1; }
  last=$(grep -E "HEALTH-GATE|$OKRE|ESCALATE|RECOVER|[|] ERROR [|]" $SL | tail -1)
  if echo "$last" | grep -qE "$OKRE" && ! echo "$last" | grep -qE "$BADRE"; then return 0; fi
  log "health: last event not healthy: ${last:0:200}"; return 1
}
wait_ready() {
  for i in $(seq 240); do
    health || { sleep 60; continue; }
    active=$(tt-device-mcp status 1 2>&1 | sed -n "/^RUNNING/,/^RECENT/p" | grep -w smarton)
    [ -n "$active" ] && { log "busy: $active"; sleep 60; continue; }
    return 0
  done
  done_ 8 "broker never healthy or free"
}
# submit_watch <tag> <command>: one broker job, watched to the end; sets JOB and ST.
submit_watch() {
  T0=$(now); echo "$T0" > $V/t0_$1
  out=$(cd $W && tt-device-mcp run-bg "$2" -w "$W" -e tmp/t158/env.yaml -t 600 2>&1); src=$?
  log "$1 submit rc=$src: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"
  [ $src = 0 ] || done_ 7 "$1 submit failed"
  JOB=$(echo "$out" | grep -oE '[0-9]+' | tail -1); echo $JOB > $V/job_id_$1; log "$1 JOB=$JOB"
  watch_job
}
watch_job() {
  while :; do
    ST=$(tt-device-mcp status -j $JOB 2>&1)
    echo "$ST" | grep -qiE "^Status: *(running|queued|pending)" || break
    [ "$(uptime -s)" = "$BOOT0" ] || done_ 9 "host reboot during job $JOB"
    sleep 30
  done
  log "job $JOB: $(echo "$ST" | grep -iE '^(Status|Exit)' | tr '\n' ' ')"
  sleep 90
  awk -v s="$T0" 'substr($0,1,19) >= s' $SL > $V/broker_slice_$JOB.log 2>/dev/null
  journalctl -k --since "$T0 UTC" --no-pager 2>/dev/null | grep -iE 'pcie|aer|tenstorrent|fatal|mce|link' > $V/journal_slice_$JOB.log
  e=$(errors_since "$T0"; gate_fail_since "$T0")
  [ -n "$e" ] && { echo "$e" > $V/drop_evidence_$JOB.log; done_ 9 "drop/error during or after job $JOB"; }
  return 0
}
log "start boot=$BOOT0"
DATA=/home/smarton/fasth3/tt-metal/tt-project/data/g15; MAN=$DATA/tt-metal-cache/kernel_prewarm.manifest
if [ -n "$WATCH_JOB" ]; then
  JOB=$WATCH_JOB; T0=$(cat $V/t0_${WATCH_TAG:-run}); watch_job
  [ "${WATCH_TAG:-run}" = run ] || done_ 5 "re-attached to $WATCH_TAG job $JOB; restart the driver without WATCH_JOB"
else
  [ "$(cat $RUN/t158-build2.rc 2>/dev/null)" = 0 ] || done_ 10 "build rc=$(cat $RUN/t158-build2.rc 2>/dev/null)"
  [ "$(cat $RUN/t158-stage.rc 2>/dev/null)" = 0 ] || done_ 11 "stage rc=$(cat $RUN/t158-stage.rc 2>/dev/null)"
  if [ ! -s $MAN ]; then
    wait_ready
    # Capture-only traverses on garbage tensors, so its exit code is not checked; the manifest is the artifact.
    submit_watch capture "export TT_METAL_KERNEL_PREWARM=1 TT_METAL_KERNEL_CAPTURE_ONLY=1 OUT=$DATA/capture; bash tmp/t158/run_e2e.sh"
    [ -s $MAN ] || done_ 12 "capture job $JOB wrote no manifest"
  fi
  log "offline compile start ($(wc -l < $MAN) manifest lines)"
  (cd $W && env TT_METAL_KERNEL_PREWARM=1 TT_METAL_CACHE=$DATA/tt-metal-cache TT_METAL_HOME=$W build_Release/tools/kernel_prewarm) >> $V/prewarm.log 2>&1 \
    || done_ 13 "kernel_prewarm rc=$?"
  fp=$(du -s --block-size=1G /home/smarton/fasth3 | cut -f1); log "offline compile done; ~/fasth3 ${fp} GB"
  [ "$fp" -lt 100 ] || done_ 14 "~/fasth3 at ${fp} GB, over the 100 GB cap"
  wait_ready
  submit_watch run "bash tmp/t158/run_e2e.sh"
fi
echo "$ST" | grep -qiE "^Status: *completed" && echo "$ST" | grep -qiE "^Exit: *0" && done_ 0 "job $JOB completed"
done_ 6 "job $JOB not completed cleanly"
