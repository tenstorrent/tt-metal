#!/bin/bash
# t251 driver run 2 on blx01: fill job 901 hit the 570 s pytest timeout while JIT compiling cold (warmup had just
# finished; the JIT cache now holds those kernels). Submit fill2 (seed 0) once the box is clear, wait; if it
# completed, submit the 5-seed timed job (seeds 0-4 in one process), wait. Marker: drv251b.done.
set -o pipefail
T=/var/tmp/fasth3/t220; M=$T/drv251b.done
trap 'echo "exit=$? $(date -u +%T)" >> $M.log; [ -e $M ] || echo "DRIVER_EXIT rc=$?" > $M' EXIT
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 120); do s=$(st $1); case $s in running|queued|pending) sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
# Submit $1 (tag) with seeds $2 once no hold/upgrade/health job and no other smarton job runs or is queued.
submit() {
  for i in $(seq 1 240); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi 'upgrade' || echo "$run$q" | grep -qiE 'hold|health|smarton'; then sleep 30; continue; fi
    sub=$(tt-device-mcp run-bg "env T220_TAG=$1 T220_SEEDS=$2 T220_PYTEST_S=570 bash $T/run220.sh" -w $T/src -e $T/env.yaml -t 600 2>&1)
    echo "$sub" >> $M.log; echo "$sub" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p'; return
  done
}
J=$(submit fill2 ""); [ -n "$J" ] || { echo "FILL_NOT_SUBMITTED" > $M; exit 0; }
echo "fill2 job=$J $(date -u +%T)" >> $M.log
s=$(waitjob $J); echo "fill2 $J status=$s $(date -u +%T)" >> $M.log
[ "$s" = completed ] || { echo "FILL_NOT_OK job=$J status=$s" > $M; exit 0; }
J=$(submit time 1,2,3,4); [ -n "$J" ] || { echo "TIME_NOT_SUBMITTED" > $M; exit 0; }
echo "time job=$J $(date -u +%T)" >> $M.log
s=$(waitjob $J); echo "time $J status=$s $(date -u +%T)" >> $M.log
echo "TIME_DONE job=$J status=$s" > $M
