#!/bin/bash
# t375 driver b on blx01 (no device work itself): waits until the broker shows no upgrade/hold/health/reset/
# fabric-check job and no other smarton job, then submits run375b.sh (-t 600) and waits for it. A non-completed
# job without its T375_EXIT line counts as a drop and is rerun once. Marker: t375/drv375b.done.
set -o pipefail
F=/var/tmp/fasth3; D=$F/t375; M=$D/drv375b.done; L=$D/drv375b.log
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $L; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
log() { echo "$(date -u '+%F %T') $*" >> $L; }
export HOME=$F/home XDG_CACHE_HOME=$F/home/.cache TMPDIR=$F/tmp
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 240); do s=$(st $1); case $s in running|queued|pending|"") sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {
  for i in $(seq 1 1440); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|fabric-check|smarton'; then sleep 30; continue; fi
    sub=$(tt-device-mcp run-bg "bash $D/run375b.sh" -w $D -e $D/env375.yaml -t 600 2>&1)
    echo "$sub" >> $L; echo "$sub" | grep -oE 'Job [0-9]+' | head -1 | grep -oE '[0-9]+'; return
  done
}
OUT=""
for att in 1 2; do
  J=$(submit); [ -n "$J" ] || { OUT="$OUT notsubmitted"; break; }
  log "attempt=$att job=$J"
  s=$(waitjob $J); log "job=$J status=$s"
  cp $D/out_b/run.log $D/run_b_job$J.log 2>/dev/null
  OUT="$OUT b:job=$J:$s"
  { [ "$s" = completed ] || grep -q '^T375_EXIT=' $D/out_b/run.log 2>/dev/null; } && break
  log "job=$J DROP? status=$s (no exit line); waiting, then rerun once"; sleep 240
done
echo "DONE$OUT" > $M
