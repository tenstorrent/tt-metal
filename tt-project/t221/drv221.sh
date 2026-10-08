#!/bin/bash
# t221 driver on blx01 (copy of t220 drv251b): bf16 LTX-2.3. Fill job(s) (seed 0; builds the bf16 DiT cache + JIT,
# up to 3 tries since caches persist per module), then the 5-seed timed job, then host-side bf16-vs-bf8 compare.
# Each job waits until the broker has no hold/upgrade/health job and no other smarton job. Marker: drv221.done.
set -o pipefail
T=/var/tmp/fasth3/t221; M=$T/drv221.done
trap 'echo "exit=$? $(date -u +%T)" >> $M.log; [ -e $M ] || echo "DRIVER_EXIT rc=$?" > $M' EXIT
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 120); do s=$(st $1); case $s in running|queued|pending) sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {
  for i in $(seq 1 360); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|smarton'; then sleep 30; continue; fi
    sub=$(tt-device-mcp run-bg "env T220_TAG=$1 T220_SEEDS=$2 T220_PYTEST_S=570 bash $T/run221.sh" -w /var/tmp/fasth3/t220/src -e $T/env.yaml -t 600 2>&1)
    echo "$sub" >> $M.log; echo "$sub" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p'; return
  done
}
ok=0
for n in 1 2 3; do
  J=$(submit fill$n ""); [ -n "$J" ] || { echo "FILL_NOT_SUBMITTED try=$n" > $M; exit 0; }
  echo "fill$n job=$J $(date -u +%T)" >> $M.log
  s=$(waitjob $J); echo "fill$n $J status=$s $(date -u +%T)" >> $M.log
  [ "$s" = completed ] && { ok=1; break; }
done
[ $ok = 1 ] || { echo "FILL_NOT_OK last_job=$J status=$s" > $M; exit 0; }
J=$(submit time 1,2,3,4); [ -n "$J" ] || { echo "TIME_NOT_SUBMITTED" > $M; exit 0; }
echo "time job=$J $(date -u +%T)" >> $M.log
s=$(waitjob $J); echo "time $J status=$s $(date -u +%T)" >> $M.log
if [ "$s" = completed ]; then
  /var/tmp/fasth3/t48/python_env/bin/python $T/cmp221.py /var/tmp/fasth3/t220/out_time/ltx_av_fast_1920x1088_1.mp4 \
    $T/out_time/ltx_av_fast_1920x1088_1.mp4 $T/out_time > $T/out_time/cmp.txt 2>&1; echo "cmp rc=$?" >> $M.log
fi
echo "TIME_DONE job=$J status=$s" > $M
