#!/bin/bash
# t221 rerun driver on blx01 after job 914 was killed by the 03:15 UTC tray-4 drop. Caches are filled (job 913),
# so only the 5-seed timed job (out_time2/) and the host-side bf16-vs-bf8 compare. Waits until the broker has no
# hold/upgrade/health job and no other smarton job. Marker: drv221b.done.
set -o pipefail
T=/var/tmp/fasth3/t221; M=$T/drv221b.done
trap 'echo "exit=$? $(date -u +%T)" >> $M.log; [ -e $M ] || echo "DRIVER_EXIT rc=$?" > $M' EXIT
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 120); do s=$(st $1); case $s in running|queued|pending) sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {
  for i in $(seq 1 720); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|smarton'; then sleep 30; continue; fi
    sub=$(tt-device-mcp run-bg "env T220_TAG=$1 T220_SEEDS=$2 T220_PYTEST_S=570 bash $T/run221.sh" -w /var/tmp/fasth3/t220/src -e $T/env.yaml -t 600 2>&1)
    echo "$sub" >> $M.log; echo "$sub" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p'; return
  done
}
J=$(submit time2 1,2,3,4); [ -n "$J" ] || { echo "TIME_NOT_SUBMITTED" > $M; exit 0; }
echo "time2 job=$J $(date -u +%T)" >> $M.log
s=$(waitjob $J); echo "time2 $J status=$s $(date -u +%T)" >> $M.log
if [ "$s" = completed ]; then
  /var/tmp/fasth3/t48/python_env/bin/python $T/cmp221.py /var/tmp/fasth3/t220/out_time/ltx_av_fast_1920x1088_1.mp4 \
    $T/out_time2/ltx_av_fast_1920x1088_1.mp4 $T/out_time2 > $T/out_time2/cmp.txt 2>&1; echo "cmp rc=$?" >> $M.log
fi
echo "TIME_DONE job=$J status=$s" > $M
