#!/bin/bash
# retry.sh <script> <log> : rerun while the open fails with a hung-device / MMIO error (another slot's chip); max 40 tries x 4 min
S=/tmp/claude-1000/-mnt-tt-data-ssinghal-tests-tt-metal/9f313510-26a3-47cd-8736-0cc09623d5a0/scratchpad; P=/mnt/tt-data/ssinghal/hy4-logs/program/idx-1m
for i in $(seq 1 40); do
  $S/run.sh $1 $2
  if grep -qE "NOC0 is hung|MMIO per-op timeout" $P/$2 && ! grep -q "^R " $P/$2; then echo "$(date -u +%T) try $i blocked" >> $P/retry.log; sleep 240; else break; fi
done
