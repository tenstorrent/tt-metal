#!/bin/bash
# t211 driver on blx01: waits for #209's driver to end, then runs two broker jobs one after another
# (conv = 2.3 VAE, diffvae = 2.5 DiffVAE; same config otherwise), then post-processes on CPU.
# Before each submit: blx01 healthy and project-free (no broker-owned job, not HELD, no upgrade, no smarton job
# running or queued; two passes 60 s apart). A drop is logged and rerun; a second drop of the same arm skips it.
# A plain failure of the unmeasured diffvae arm (cold DiffVAE weight cache / JIT) is retried once, warmer.
# Marker: $T/driver.marker "T211_DRIVER_DONE stage=.. rc=.. jobs=.."
F=/var/tmp/fasth3; T=$F/t211; L=$T/driver.log; M=$T/driver.marker; WS=$F/t48
STAGE=start; JOBS=""; first_rc=0
trap 'echo "T211_DRIVER_DONE stage=$STAGE rc=$first_rc jobs=${JOBS# }" > $M' EXIT
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
# one_job <label> <tmo> <pytest_s> [VAR=val ...] -> sets ST EX JOB
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
arm() {
  local label=$1 tmo=$2 ps=$3; shift 3
  local drops=0 fails=0 maxfail=$1; shift
  while :; do
    STAGE="ready-$label"; log "$label: waiting for blx01 ready"
    wait_ready || { log "$label: blx01 not ready after 6 h"; first_rc=75; return 1; }
    STAGE="job-$label"; one_job $label $tmo $ps "$@"
    [ "$ST" = completed ] && [ "$EX" = 0 ] && return 0
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
STAGE=wait-t209
t0=$(date +%s)
while [ ! -e $F/t209/driver.marker ] && pgrep -u smarton -f "$F/t209/driver.sh" > /dev/null; do
  (( $(date +%s) - t0 > 28800 )) && { log "t209 driver still running after 8 h"; first_rc=75; exit; }
  sleep 60
done
log "t209 driver done: $(cat $F/t209/driver.marker 2>/dev/null)"
arm conv 220 200 0 LTX25_DIFFVAE=0 || exit
arm diffvae 600 570 1 LTX25_DIFFVAE=1 || exit
STAGE=post
bash $T/post.sh >> $L 2>&1 || first_rc=$?
log "post rc=$first_rc"
