#!/bin/bash
# t164 driver on g15blx02: one broker job per line of $CONFIGS, in order, each after the health gate and
# with no other smarton job running or queued. A failed job with broker errors around it is a drop and is
# rerun; two drops in a row skip that config. A failed baseline without a drop stops the driver.
# Marker: $V/DRIVER.done = "<code> <reason>". Results: $DATA/t164/<label>/ (run.log, mp4, jpg).
# Usage: CONFIGS=<file> TAG=<name> bash driver.sh
T=/home/smarton/fasth3/tt-metal/tt-project/worktrees/t164/tmp/t164
S=/home/smarton/fasth3/tt-metal/tt-project/worktrees/t164
W=/home/smarton/fasth3/tt-metal/tt-project/worktrees/t158
DATA=/home/smarton/fasth3/tt-metal/tt-project/data/g15
CONFIGS=${CONFIGS:-$T/configs.txt}; TAG=${TAG:-pack}
V=$DATA/t164/driver_$TAG; mkdir -p $V
SL=/var/log/tt-device-broker/server.log; D=$V/driver.log; BOOT0=$(uptime -s)
# Job 406 (same build, caches and warmup cuts, gen#0 + 5 warm gens) held the device 152 s; the baseline
# limit is that +50%.
# Knob configs add in-window JIT compiles for their changed kernels (unmeasured), so they get 400 s.
TO_BASE=${TO_BASE:-240}; TO_KNOB=${TO_KNOB:-400}
log() { echo "$(date -u '+%F %T') $*" >> $D; }
done_() { log "T164_DRIVER_DONE $1 $2"; echo "$1 $2" > $V/DRIVER.done; exit 0; }
now() { date -u '+%F %T'; }
OKRE=': OK|device healthy; no reset needed|heartbeat: HEALTHY'
BADRE='[|] ERROR [|]|ESCALATE|RECOVER'
errors_since() { awk -v s="$1" 'substr($0,1,19) > s && /\| ERROR \|/' $SL; }
gate_fail_since() { awk -v s="$1" -v ok="$OKRE" -v bad="$BADRE" 'substr($0,1,19) > s && ($0 ~ bad || (/HEALTH-GATE/ && $0 !~ ok))' $SL; }
health() {
  systemctl is-active -q tt-device-broker || { log "health: broker inactive"; return 1; }
  st=$(tt-device-mcp status 1 2>&1) || { log "health: status failed"; return 1; }
  echo "$st" | grep -qi 'upgrade' && { log "health: broker upgrade running"; return 1; }
  echo "$st" | sed -n '/^RUNNING/,/^QUEUED/p' | grep -qiE '🔧|health-gate|fabric-check|recover|bridge-reset|power-cycle' && { log "health: broker gate/recovery running"; return 1; }
  last=$(grep -E "HEALTH-GATE|$OKRE|ESCALATE|RECOVER|[|] ERROR [|]" $SL | tail -1)
  # The broker logs the end of its own reset-and-verify at ERROR level; that line means the gate passed.
  echo "$last" | grep -qE 'reset complete \+ health verified' && return 0
  if echo "$last" | grep -qE "$OKRE" && ! echo "$last" | grep -qE "$BADRE"; then return 0; fi
  log "health: last event not healthy: ${last:0:200}"; return 1
}
wait_ready() {
  for i in $(seq 360); do
    health || { sleep 60; continue; }
    active=$(tt-device-mcp status 1 2>&1 | sed -n "/^RUNNING/,/^RECENT/p" | grep -w smarton)
    [ -n "$active" ] && { log "busy: $(echo $active | cut -c1-160)"; sleep 60; continue; }
    return 0
  done
  done_ 8 "broker not healthy or free for 6 h"
}
disk_ok() {
  fp=$(du -s --block-size=1M /home/smarton/fasth3 | cut -f1); log "~/fasth3 $((fp / 1024)) GiB ($fp MiB)"
  [ "$fp" -lt $((99 * 1024)) ]
}
watch_job() {
  while :; do
    ST=$(tt-device-mcp status -j $JOB 2>&1)
    echo "$ST" | grep -qiE "^Status: *(running|queued|pending)" || break
    [ "$(uptime -s)" = "$BOOT0" ] || done_ 9 "host reboot during job $JOB"
    sleep 30
  done
  log "job $JOB: $(echo "$ST" | grep -iE '^(Status|Exit|Runtime)' | tr -s ' ' | tr '\n' ' ')"
  sleep 30
  awk -v s="$T0" 'substr($0,1,19) >= s' $SL | grep -v DEBUG > $V/broker_slice_$JOB.log 2>/dev/null
  journalctl -k --since "$T0 UTC" --no-pager 2>/dev/null | grep -iE 'pcie|aer|tenstorrent|fatal|mce|link' > $V/journal_slice_$JOB.log
  EVID=$(errors_since "$T0"; gate_fail_since "$T0")
  [ -n "$EVID" ] && echo "$EVID" > $V/drop_evidence_$JOB.log
}
# run_one <label> <flags...>: returns 0 ok, 1 failed, 2 skipped after two drops in a row.
run_one() {
  label=$1; shift; drops=0
  # PRIOR_DROP=<label>: that label's last job already dropped (an earlier driver run), so one more drop skips it.
  [ "$label" = "${PRIOR_DROP:-}" ] && drops=1
  while :; do
    wait_ready
    disk_ok || done_ 14 "~/fasth3 at or over 99 GiB before $label"
    to=$TO_KNOB; [ "$label" = baseline ] || [ "$label" = baseline5 ] && to=$TO_BASE
    T0=$(now)
    out=$(cd $S && tt-device-mcp run-bg "env PYTEST_S=$((to - 30)) bash $T/run_cfg.sh $label $*" -w $S -e $T/env.yaml -t $to 2>&1); src=$?
    log "$label submit rc=$src -t $to: $(echo "$out" | tr '\n' ' ' | cut -c1-200)"
    [ $src = 0 ] || done_ 7 "$label submit failed"
    JOB=$(echo "$out" | grep -oE '[0-9]+' | tail -1); echo "$label $JOB $T0" >> $V/jobs.txt
    watch_job
    ok=0
    echo "$ST" | grep -qiE "^Status: *completed" && echo "$ST" | grep -qiE "^Exit: *0" \
      && grep -q "T164_EXIT\[$label\]=0" $DATA/t164/$label/run.log 2>/dev/null && ok=1
    if [ $ok = 1 ]; then
      [ -n "$EVID" ] && log "$label job $JOB ok, but broker errors after it (drop logged, result kept)"
      log "$label OK job $JOB"; return 0
    fi
    if [ -n "$EVID" ]; then
      drops=$((drops + 1)); log "$label DROP #$drops job $JOB (evidence drop_evidence_$JOB.log)"
      [ $drops -ge 2 ] && { log "$label SKIPPED: two drops in a row"; return 2; }
      continue
    fi
    log "$label FAILED job $JOB without broker errors"; return 1
  done
}
log "start tag=$TAG configs=$CONFIGS boot=$BOOT0 pid=$$"
while read -r label flags <&3; do
  [ -z "$label" ] || [ "${label:0:1}" = "#" ] && continue
  if grep -q "T164_EXIT\[$label\]=0" $DATA/t164/$label/run.log 2>/dev/null; then log "$label already done, skipped"; continue; fi
  run_one $label $flags; rc=$?
  echo "$label rc=$rc" >> $V/results.txt
  [ $rc != 0 ] && [ "${label#baseline}" != "$label" ] && done_ 6 "$label did not complete (rc $rc); the pack needs it"
done 3< $CONFIGS
disk_ok; log "device jobs done; CPU post-processing"
(cd $S && source /home/smarton/fasth3/tt-metal/python_env/bin/activate && \
  PYTHONPATH=$S:$W/ttnn:$W/tools timeout 3000 python $T/post.py $DATA/t164 $CONFIGS) >> $V/post.log 2>&1
prc=$?
disk_ok
done_ 0 "all configs run (post rc $prc)"
