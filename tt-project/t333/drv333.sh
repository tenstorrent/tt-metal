#!/bin/bash
# t333 driver on blx01 (no device work itself): waits for setup333.sh (weights copy + build), then runs the standard
# LTX-2.5 conv-VAE e2e as separate broker jobs, one at a time, -t 600 each:
#   c6a = 145 frames, cold JIT (fills t333/jit; its timing counts only if it finishes), c6 = 145 frames warm (headline),
#   c10 = 241 frames (10 s). Submitted only when the broker has no upgrade/hold/health/reset/fabric-check job and no
# other smarton job. A non-completed job without its T333_EXIT line counts as a drop and is rerun once (two in a row:
# skipped). Logs: t333/run_<tag>_job<id>.log. Marker: t333/drv333.done (first line = outcome).
set -o pipefail
F=/var/tmp/fasth3; D=$F/t333; M=$D/drv333.done; L=$D/drv333.log
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $L; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 120); do s=$(st $1); case $s in running|queued|pending|"") sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {  # tag frames -> job id
  for i in $(seq 1 720); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|fabric-check|smarton'; then sleep 30; continue; fi
    lg=$(ls -t /var/log/tt-device-broker/* 2>/dev/null | head -1)
    log "submit $1: broker log $lg AICLK-clamp lines $(grep -c 'AICLK failed to settle' $lg 2>/dev/null)"
    sub=$(tt-device-mcp run-bg "bash $D/run333.sh $1 $2" -w $D -e $D/env333.yaml -t 600 2>&1)
    echo "$sub" >> $L; echo "$sub" | grep -oE 'Job [0-9]+' | head -1 | grep -oE '[0-9]+'; return
  done
}
for i in $(seq 1 480); do grep -q T333_DRIVER_DONE $D/driver.log 2>/dev/null && break; sleep 30; done
grep -q 'T333_DRIVER_DONE setup 0 ' $D/driver.log || { echo "SETUP_FAILED $(tail -1 $D/driver.log 2>/dev/null)" > $M; exit 1; }
log "setup ok: $(tail -1 $D/driver.log)"
OUT=""
for spec in c6a:145 c6:145 c10:241; do
  tag=${spec%%:*}; nf=${spec##*:}
  for att in 1 2; do
    J=$(submit $tag $nf); [ -n "$J" ] || { OUT="$OUT $tag:notsubmitted"; break; }
    log "tag=$tag attempt=$att job=$J"
    s=$(waitjob $J); log "tag=$tag job=$J status=$s"
    cp $D/out_$tag/run.log $D/run_${tag}_job$J.log 2>/dev/null
    sleep 10; log "tag=$tag job=$J leftover: $(ps -u $(id -u) -o pid=,pgid=,args= | grep -E 'pytest|run333' | grep -v grep | tr '\n' ';')"
    OUT="$OUT $tag:job=$J:$s"
    { [ "$s" = completed ] || grep -q '^T333_EXIT=' $D/out_$tag/run.log 2>/dev/null; } && break
    log "tag=$tag job=$J DROP? status=$s (no exit line); waiting for the broker, then rerun"; sleep 240
  done
done
echo "DONE$OUT" > $M
