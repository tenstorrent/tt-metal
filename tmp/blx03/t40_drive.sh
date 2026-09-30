#!/bin/bash
# t40 driver on blx03: profile job, then the 1080p/145f e2e with the encode-capture fix, one broker
# job at a time behind any other project job. Writes $OUT/DONE when both have finished.
OUT=/home/smarton/fasth3/out/t40
mkdir -p $OUT
submit() {  # label, timeout, command
  local out id
  while true; do
    out=$(~/fasth3/tt-metal/tmp/blx03/submit.sh "$2" "$3" 2>&1); [ $? -eq 75 ] || break
    sleep 60
  done
  id=$(echo "$out" | grep -oE '^Job [0-9]+' | grep -oE '[0-9]+')
  echo "$(date +%T) $1 job $id" >> $OUT/jobs.txt
  [ -n "$id" ] || { echo "$out" >> $OUT/jobs.txt; return; }
  while tt-device-mcp status -j $id 2>&1 | grep -qiE '^Status: *(running|queued)'; do sleep 30; done
  echo "$(date +%T) $1 job $id: $(tt-device-mcp status -j $id 2>&1 | grep -iE '^Status' | head -1)" >> $OUT/jobs.txt
}
submit prof 1200 "bash /home/smarton/fasth3/t40/tmp/blx03/t40_prof.sh"
submit e2e 1500 "W=/home/smarton/fasth3/t40 OUT=/home/smarton/fasth3/out/t40 PYTEST_TIMEOUT=1200 bash /home/smarton/fasth3/t40/tmp/blx03/run25.sh conv145_t40"
touch $OUT/DONE
