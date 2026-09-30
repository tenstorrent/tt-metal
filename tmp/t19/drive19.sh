#!/bin/bash
# t19 (runs on blx03): after t16's driver and every other smarton job are done, queue ONE job: LTX-2.5 1080p/145f,
# conv decoder, seeds 0,1,2 on the default prompt, stage-2 latents dumped to /var/tmp/fasth3/t19. Ends with DRIVE19_DONE.
cd /home/smarton/fasth3/t19
# Busy while another smarton job runs or queues, the broker runs a recovery step, or its latest hold row is still HELD.
busy() {
  local st; st=$(tt-device-mcp status 40 2>&1) || return 0
  echo "$st" | sed -n "/^RUNNING/,/^RECENT/p" | grep -qwE "smarton|broker" && return 0
  echo "$st" | sed -n "/^RECENT/,\$p" | grep -m1 "\[broker\]hold" | grep -q "HELD"
}
while true; do
  while busy; do sleep 60; done
  sleep $((RANDOM % 30)); busy || break
done
mkdir -p /var/tmp/fasth3/t19
out=$(tt-device-mcp run-bg "W=/home/smarton/fasth3/t19 bash tmp/blx03/run25.sh t19_conv3 LTX25_DIFFVAE=0 LTX_SEEDS=0,1,2 LTX_FRESH_PROMPTS=0 LTX_DUMP_LATENTS=/var/tmp/fasth3/t19/lat PYTEST_TIMEOUT=1140" -w "$PWD" -e tmp/t19/env19.yaml -t 1200 2>&1)
echo "$out"
id=$(echo "$out" | grep -oE '[0-9]{3,}' | head -1)
echo "JOB=$id"
while tt-device-mcp status -j $id 2>&1 | grep -qiE '^Status: *(running|queued)'; do sleep 30; done
tt-device-mcp status -j $id
echo DRIVE19_DONE
