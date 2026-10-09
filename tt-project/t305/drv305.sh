#!/bin/bash
# t305 driver on blx01: repeat of t301. Main arm is already broker job $1 (submitted by hand); wait for it, then run the
# PR arm, one broker job each (-t 570), submitted only when the broker has no hold/health/reset/upgrade/fabric-check job
# and no other smarton job. A non-completed arm without its own exit line counts as a drop and is rerun once (two drops
# in a row: skipped). Logs copied to t305/run_<arm>_job<id>.log. Marker: t305/drv305.done (first line = outcome).
set -o pipefail
F=/var/tmp/fasth3; D=$F/t301; O=$F/t305; M=$O/drv305.done; L=$M.log
MAINJOB=${1:?main job id}
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $L; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 120); do s=$(st $1); case $s in running|queued|pending|"") sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {
  for i in $(seq 1 720); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|fabric-check|smarton'; then sleep 30; continue; fi
    sub=$(tt-device-mcp run-bg "bash $D/run301.sh $1" -w $D -e $D/env.yaml -t 570 2>&1)
    echo "$sub" >> $L; echo "$sub" | grep -oE 'Job [0-9]+' | head -1 | grep -oE '[0-9]+'; return
  done
}
finish() {  # arm job -> status line; copies the log and records leftovers
  local A=$1 J=$2 s; s=$(waitjob $J); log "arm=$A job=$J status=$s"
  cp $D/out_$A/run.log $O/run_${A}_job$J.log 2>/dev/null
  sleep 10; log "arm=$A job=$J leftover: $(ps -u $(id -u) -o pid=,pgid=,args= | grep -E 'pytest|run301' | grep -v grep | tr '\n' ';')"
  echo $s
}
dropped() { [ "$2" != completed ] && ! grep -q '^T301_EXIT=' $D/out_$1/run.log 2>/dev/null; }
declare -A RES
for A in main pr; do
  for att in 1 2; do
    if [ $A = main ] && [ $att = 1 ]; then J=$MAINJOB; else J=$(submit $A); fi
    [ -n "$J" ] || { RES[$A]="notsubmitted"; break; }
    log "arm=$A attempt=$att job=$J"
    s=$(finish $A $J); RES[$A]="job=$J status=$s"
    dropped $A $s || break
    log "arm=$A job=$J DROP? status=$s (no exit line); waiting for the broker, then rerun"
    sleep 240
  done
done
echo "DONE main:${RES[main]} pr:${RES[pr]}" > $M
