#!/bin/bash
# t283 driver on blx01: bf8 job, then bf16 job, one at a time through the blx01 broker. Marker: drv283.done.
set -o pipefail
T=/var/tmp/fasth3/t283; M=$T/drv283.done
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $M.log; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
waitjob() { for i in $(seq 1 120); do s=$(st $1); case $s in running|queued|pending) sleep 30;; *) echo $s; return;; esac; done; echo stillrunning; }
submit() {
  for i in $(seq 1 720); do
    out=$(tt-device-mcp status 2>&1)
    run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
    if echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|smarton'; then sleep 30; continue; fi
    sub=$(tt-device-mcp run-bg "bash $T/run283.sh $1" -w /var/tmp/fasth3/t220/src -e $T/env.yaml -t 600 2>&1)
    echo "$sub" >> $M.log; echo "$sub" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p'; return
  done
}
res=""
for P in ${T283_PRECS:-bf8 bf16}; do
  J=$(submit $P); [ -n "$J" ] || { res="$res $P:NOT_SUBMITTED"; continue; }
  echo "$P job=$J submitted $(date -u +%T)" >> $M.log
  s=$(waitjob $J); echo "$P job=$J status=$s $(date -u +%T)" >> $M.log
  res="$res $P:$J:$s"
done
echo "DONE$res" > $M
