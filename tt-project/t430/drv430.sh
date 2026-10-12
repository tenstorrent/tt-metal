#!/bin/bash
# t430 step 2 driver (CPU only, blx01): wait for broker job 862 to leave the queue (max 60 min), then bench.
set -eo pipefail
cd /var/tmp/fasth3/t430
for i in $(seq 360); do
  tt-device-mcp status -j 862 2>&1 | grep -qiE 'Status: +(running|queued)' || break
  sleep 10
done
tt-device-mcp status -j 862 2>&1 | grep -E 'Status|Exit|Runtime'
ionice -c3 bash bench430.sh > bench.log 2>&1
cat bench.log
