#!/bin/bash
# g15blx02 driver: wait until no other project job runs on blx03, submit tmp/t41/job.sh, wait for it,
# write T41_DRIVE_DONE. Log: tmp/t41/drive.log (job id on the JOB[...] line).
cd "$(dirname "$0")"
args="$*"
while true; do
  out=$(ssh -o BatchMode=yes g14blx03 "~/fasth3/tt-metal/tmp/blx03/submit.sh 5400 bash /home/smarton/fasth3/t41/tmp/t41/job.sh $args" 2>&1)
  rc=$?
  [ $rc -eq 75 ] || [ -z "$out" ] || echo "$out" | grep -q "busy" || break
  sleep 120
done
echo "$out"
id=$(echo "$out" | grep -oE '[Jj]ob[^0-9]*[0-9]+' | grep -oE '[0-9]+' | head -1)
echo "JOB[$id] submitted $(date +%T)"
[ -n "$id" ] || { echo T41_DRIVE_DONE; exit 1; }
while ssh -o BatchMode=yes g14blx03 "tt-device-mcp status -j $id" 2>&1 | grep -qiE 'running|queued'; do sleep 60; done
ssh -o BatchMode=yes g14blx03 "tt-device-mcp status -j $id" 2>&1 | tail -5
echo T41_DRIVE_DONE
