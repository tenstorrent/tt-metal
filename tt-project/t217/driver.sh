#!/bin/bash
# t217 driver on blx01: waits for the #211 and #209 drivers to end, then two broker jobs one after another:
#   fill: first all_bf8_lofi run (host bf8 cast + DiT cache write + new JIT kernels), seed 0 only.
#   time: warm run, seeds 0-4 in one process (the measured job).
# Ready/drop logic as t211's driver. Marker: $T/driver.marker "T217_DRIVER_DONE stage=.. rc=.. jobs=.."
F=/var/tmp/fasth3; T=$F/t217; L=$T/driver.log; M=$T/driver.marker; WS=$F/t48
STAGE=start; JOBS=""; first_rc=0
trap 'echo "T217_DRIVER_DONE stage=$STAGE rc=$first_rc jobs=${JOBS# }" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
bst() { tt-device-mcp status "$@" 2>&1; }
ready_once() {
  local s run queued
  s=$(bst) || return 1
  echo "$s" | grep -qi 'upgrade' && return 1
  echo "$s" | grep -q '^RUNNING' || return 1
  run=$(echo "$s" | sed -n '/^RUNNING/,/^QUEUED/p')
  queued=$(echo "$s" | sed -n '/^QUEUED/,/^RECENT/p')
  echo "$run" | grep -qE 'HELD|🔧' && return 1
  echo "$run$queued" | grep -q 'smarton' && return 1
  return 0
}
wait_ready() {
  local t0=$(date +%s)
  while :; do
    if ready_once; then sleep 60; ready_once && return 0; fi
    (( $(date +%s) - t0 > 21600 )) && return 1
    sleep 60
  done
}
one_job() {
  local label=$1 tmo=$2 ps=$3; shift 3
  local out
  out=$(cd $WS && tt-device-mcp run-bg "env PYTEST_S=$ps bash $T/run_cfg.sh $label $*" -w $WS -e $F/t159/env.yaml -t $tmo 2>&1)
  JOB=$(echo "$out" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p' | head -1)
  [ -n "$JOB" ] || { log "$label: submit failed: $(echo "$out" | tr '\n' ' ' | cut -c1-300)"; ST=submitfail; EX=-; return; }
  JOBS="$JOBS $label:$JOB"
  log "$label: submitted job $JOB (-t $tmo, pytest $ps, $*)"
  while :; do
    ST=$(bst -j $JOB | sed -n 's/^Status: *//p')
    echo "$ST" | grep -qiE 'running|queued' || break
    sleep 20
  done
  EX=$(bst -j $JOB | sed -n 's/^Exit: *//p')
  bst -j $JOB > $T/job$JOB.status
  log "$label: job $JOB ended status=$ST exit=${EX:--}"
}
# arm <label> <tmo> <pytest_s> <maxfail> [VAR=val ...]; a fill arm also passes once the bf8 DiT cache is complete
arm() {
  local label=$1 tmo=$2 ps=$3; shift 3
  local drops=0 fails=0 maxfail=$1; shift
  while :; do
    STAGE="ready-$label"; log "$label: waiting for blx01 ready"
    wait_ready || { log "$label: blx01 not ready after 6 h"; first_rc=75; return 1; }
    STAGE="job-$label"; one_job $label $tmo $ps "$@"
    [ "$ST" = completed ] && [ "$EX" = 0 ] && return 0
    if [ "$label" = fill ] && grep -q "Writing cache to '.*q-all_bf8_lofi" $T/res/fill/run.log 2>/dev/null \
       && grep -q "E2E_WALL_S\|Video export\|denois" $T/res/fill/run.log && ! grep -qE 'Traceback|TT_FATAL' $T/res/fill/run.log; then
      log "fill: job ended $ST $EX but bf8 cache exists, going on"; return 0
    fi
    if echo "$ST" | grep -qiE 'killed|abandoned|interrupted|power|reboot'; then
      drops=$((drops + 1))
      log "DROP $label: utc=$(date -u '+%F %T') job=$JOB status=$ST (drop $drops); $(grep -m1 -i '^Cause' $T/job$JOB.status)"
      [ $drops -ge 2 ] && { log "$label: dropped twice on blx01, skipped"; first_rc=86; return 1; }
      continue
    fi
    fails=$((fails + 1))
    [ $fails -le $maxfail ] && { log "$label: failed ($ST $EX), retry $fails of $maxfail"; continue; }
    first_rc=1; return 1
  done
}
log "driver start pid $$"
STAGE=wait-others
t0=$(date +%s)
while pgrep -u smarton -f "$F/t211/driver.sh" > /dev/null || pgrep -u smarton -f "$F/t209/driver.sh" > /dev/null; do
  (( $(date +%s) - t0 > 28800 )) && { log "t209/t211 drivers still running after 8 h"; first_rc=75; exit; }
  sleep 60
done
log "t211: $(cat $F/t211/driver.marker 2>/dev/null); t209: $(cat $F/t209/driver.marker 2>/dev/null)"
arm fill 600 570 1 LTX_E2E_SEEDS=0 || exit
arm time 600 570 1 || exit
STAGE=done
log "driver done"
