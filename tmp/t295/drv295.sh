#!/bin/bash
# t295 driver on blx01: waits until the box has been up >=15 min and the broker runs/queues no hold, health
# check, reset or other smarton job, submits job295.sh once (-t 520: ~2x170 s measured-ish +50%), waits for it.
# Marker: $T/drv295.done (first line DONE job=<id> status=<s> or a failure reason).
set -o pipefail
# Broker env: run-bg needs PYTHON_ENV_DIR (run 1178 failed NOT_SUBMITTED without it).
T=/var/tmp/fasth3/t295; M=$T/drv295.done
trap 'rc=$?; echo "exit=$rc $(date -u +%T)" >> $M.log; [ -e $M ] || echo "DRIVER_EXIT rc=$rc" > $M' EXIT
st() { tt-device-mcp status -j $1 2>&1 | awk '/^Status:/{print $2}'; }
J=""
for i in $(seq 1 240); do
  up=$(( $(date +%s) - $(date -d "$(uptime -s)" +%s) ))
  out=$(tt-device-mcp status 2>&1)
  run=$(echo "$out" | sed -n '/^RUNNING/,/^QUEUED/p'); q=$(echo "$out" | sed -n '/^QUEUED/,/^RECENT/p')
  if [ $up -lt 900 ] || echo "$out" | grep -qi upgrade || echo "$run$q" | grep -qiE 'hold|health|reset|fabric|smarton'; then sleep 30; continue; fi
  sub=$(tt-device-mcp run-bg "bash $T/job295.sh" -w /var/tmp/fasth3/t48 -e $T/env295.yaml -t 520 2>&1)
  echo "$(date -u +%T) $sub" >> $M.log
  J=$(echo "$sub" | sed -n 's/^Job \([0-9]*\) queued.*/\1/p'); break
done
[ -n "$J" ] || { echo "NOT_SUBMITTED $(date -u +%T)" > $M; exit 1; }
for i in $(seq 1 240); do s=$(st $J); case $s in running|queued|pending) sleep 30;; *) break;; esac; done
tt-device-mcp logs $J > $T/job$J.log 2>&1 || true
echo "DONE job=$J status=$s $(date -u +%T)" > $M
