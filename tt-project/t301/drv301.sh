#!/bin/bash
# t301 driver on blx01: wait for the PR build (setup301.sh) and t293's main build, then run the main arm and the PR arm
# back to back, one broker job each (-t 600), each only when the broker has no hold/health/reset/upgrade job and no other
# smarton job. A non-completed arm that did not reach its own exit line counts as a drop and is rerun once (two drops in
# a row: skipped). Then host-side PCC/PSNR main vs PR per gen. Marker: drv301.done (first line = outcome).
set -o pipefail
F=/var/tmp/fasth3; D=$F/t301; M=$D/drv301.done; L=$M.log
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $L; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 120); do s=$(st $1); case $s in running|queued|pending|"") sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {
  for i in $(seq 1 720); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|fabric-check|smarton'; then sleep 30; continue; fi
    sub=$(tt-device-mcp run-bg "bash $D/run301.sh $1" -w $D -t 600 2>&1)
    echo "$sub" >> $L; echo "$sub" | grep -oE 'Job [0-9]+' | head -1 | grep -oE '[0-9]+'; return
  done
}
for i in $(seq 1 240); do [ -e $D/setup.rc ] && [ -e $F/t293/setup.rc ] && break; sleep 60; done
log "setup.rc=$(cat $D/setup.rc 2>/dev/null) t293 setup.rc=$(cat $F/t293/setup.rc 2>/dev/null)"
[ "$(cat $D/setup.rc 2>/dev/null)" = 0 ] || { echo "PR_BUILD_FAILED rc=$(cat $D/setup.rc 2>/dev/null)" > $M; exit 0; }
[ -e $F/t293/main/ttnn/ttnn/_ttnn.so ] || { echo "MAIN_BUILD_MISSING t293 rc=$(cat $F/t293/setup.rc 2>/dev/null)" > $M; exit 0; }
declare -A RES
for A in main pr; do
  for att in 1 2; do
    J=$(submit $A); [ -n "$J" ] || { RES[$A]="notsubmitted"; break; }
    log "arm=$A attempt=$att job=$J submitted"
    s=$(waitjob $J); log "arm=$A attempt=$att job=$J status=$s"
    cp $D/out_$A/run.log $D/run_${A}_job$J.log 2>/dev/null
    RES[$A]="job=$J status=$s"
    [ "$s" = completed ] && break
    if [ "$s" = failed ] && grep -q '^T301_EXIT=' $D/out_$A/run.log 2>/dev/null; then break; fi
    log "arm=$A job=$J DROP? status=$s (no exit line); waiting for the broker, then rerun"
    sleep 240
  done
done
P=$F/t48/python_env/bin/python
for g in 0 1 2; do f=ltx_av_fast_1920x1088_$g.mp4
  [ -e $D/out_main/$f ] && [ -e $D/out_pr/$f ] && $P $D/cmp301.py $D/out_main/$f $D/out_pr/$f $D gen$g >> $D/cmp.txt 2>&1
done
log "cmp rc=$?"
echo "DONE main:${RES[main]} pr:${RES[pr]}" > $M
