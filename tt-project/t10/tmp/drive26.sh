#!/bin/bash
# Runs the four LTX-2.5 baselines as separate broker jobs, one after another.
# Progress: tmp/drive26.log. Ends with a DRIVE_DONE line.
W=/home/smarton/fasth3/tt-metal/tt-project/worktrees/t10
OUT=/home/smarton/fasth3/tt-metal/tt-project/baselines/ltx25_1080p_6s
cd $W
run() {
  local label=$1; shift
  local out id
  out=$(tt-device-mcp run-bg "bash tmp/run25.sh $label $*" -w $W -e tmp/ltx25_env.yaml -t 600 2>&1)
  id=$(echo "$out" | grep -oiE 'job[^0-9]*[0-9]+' | grep -oE '[0-9]+' | head -1)
  echo "$(date +%T) SUBMIT $label job=$id :: $(echo $out | tr '\n' ' ' | cut -c1-200)"
  [ -z "$id" ] && return 1
  while tt-device-mcp status -j $id 2>&1 | grep -qiE '^Status:\s+(running|queued|pending)'; do sleep 60; done
  echo "$(date +%T) END $label job=$id $(tt-device-mcp status -j $id 2>&1 | grep -E '^(Status|Runtime|Log):' | tr '\n' ' ')"
}
run dv145
if grep -q "CACHE MISS: weight_load ltx-2.5-22b-distilled-transformer" $OUT/dv145/run.log 2>/dev/null || ! grep -q "RUN_EXIT\|passed" $OUT/dv145/run.log 2>/dev/null; then
  echo "$(date +%T) dv145 missed the DiT cache or did not finish; stopping"; echo DRIVE_DONE; exit 1
fi
run dv145_c211 DIFFVAE_NA_CHUNK_BRICKS=2,1,1 DIFFVAE_NA_UNSAFE_CHUNK=1
run dv153 NUM_FRAMES=153 FPS=25
run conv145 LTX25_DIFFVAE=0
echo DRIVE_DONE
