#!/bin/bash
# t283 driver on blx01 (ltx-rt HEAD Python, t48 build): bf16 job, then bf8 job, one at a time through the blx01 broker. Marker: drv283c.done.
set -o pipefail
T=/var/tmp/fasth3/t283; M=$T/drv283c.done
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $M.log; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 120); do s=$(st $1); case $s in running|queued|pending) sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {
  for i in $(seq 1 720); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|smarton'; then sleep 30; continue; fi
    sub=$(tt-device-mcp run-bg "bash $T/run283c.sh $1" -w /var/tmp/fasth3/t48 -e $T/envrt.yaml -t 600 2>&1)
    echo "$sub" >> $M.log; echo "$sub" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p'; return
  done
}
res=""
for P in ${T283_PRECS:-bf16 bf8}; do
  J=$(submit $P); [ -n "$J" ] || { res="$res $P:NOT_SUBMITTED"; continue; }
  echo "$P job=$J submitted $(date -u +%T)" >> $M.log
  s=$(waitjob $J); echo "$P job=$J status=$s $(date -u +%T)" >> $M.log
  # A cold weight cache can push the first run past the 570 s pytest timeout after the cache is published;
  # rerun once (warm) only in that case.
  if [ "$s" != completed ] && grep -q "Timeout" $T/outrt_$P/run.log 2>/dev/null; then
    J2=$(submit $P); echo "$P retry job=$J2 submitted $(date -u +%T)" >> $M.log
    [ -n "$J2" ] && { s=$(waitjob $J2); echo "$P job=$J2 status=$s $(date -u +%T)" >> $M.log; J="$J,$J2"; }
  fi
  res="$res $P:$J:$s"
done
echo "DONE$res" > $M
