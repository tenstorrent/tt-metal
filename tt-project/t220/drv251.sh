#!/bin/bash
# t251 driver on blx01: wait for fill job $1; if it completed, submit the 5-seed timed job once the box is
# clear (no hold/upgrade, no other smarton job running or queued), then wait for it. Marker: drv251.done.
set -o pipefail
T=/var/tmp/fasth3/t220; M=$T/drv251.done; FILL=$1
trap 'echo "exit=$? $(date -u +%T)" >> $M.log; [ -e $M ] || echo "DRIVER_EXIT rc=$?" > $M' EXIT
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 60); do s=$(st $1); case $s in running|queued|pending) sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
s=$(waitjob $FILL); echo "fill $FILL status=$s $(date -u +%T)" >> $M.log
[ "$s" = completed ] || { echo "FILL_NOT_OK job=$FILL status=$s" > $M; exit 0; }
for i in $(seq 1 40); do
  out=$(tt-device-mcp status 2>&1)
  run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
  if echo "$out" | grep -qi 'upgrade' || echo "$run$q" | grep -qiE 'hold|health|smarton'; then sleep 30; continue; fi
  sub=$(tt-device-mcp run-bg "env T220_TAG=time T220_SEEDS=1,2,3,4 T220_PYTEST_S=570 bash $T/run220.sh" -w $T/src -e $T/env.yaml -t 600 2>&1)
  echo "$sub" >> $M.log; J=$(echo "$sub" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p'); break
done
[ -n "$J" ] || { echo "TIME_NOT_SUBMITTED" > $M; exit 0; }
s=$(waitjob $J); echo "time $J status=$s $(date -u +%T)" >> $M.log
echo "TIME_DONE job=$J status=$s" > $M
