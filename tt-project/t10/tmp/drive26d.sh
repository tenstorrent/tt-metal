#!/bin/bash
# Run the four LTX-2.5 1080p/6s baselines as separate broker jobs, one at a time.
# Resumable: a label whose run.log already shows " passed" is skipped, so rerun after a reboot kills it.
# Progress: tmp/drive26d.log. Ends with a DRIVE_DONE line.
W=/home/smarton/fasth3/tt-metal/tt-project/worktrees/t10
OUT=/home/smarton/fasth3/tt-metal/tt-project/baselines/ltx25_1080p_6s
cd $W
wait_job() { while tt-device-mcp status -j $1 2>&1 | grep -qiE '^Status:\s+(running|queued|pending|starting)'; do sleep 60; done; }
run() {
  local label=$1; shift
  if grep -q " passed" $OUT/$label/run.log 2>/dev/null; then echo "$(date +%T) SKIP $label (passed)"; return 0; fi
  local out id
  # RESUME_<label>=<job id> waits on a job that is already queued instead of submitting a duplicate.
  local rv=RESUME_$label; id=${!rv}
  if [ -n "$id" ]; then echo "$(date +%T) RESUME $label job=$id"; wait_job $id
    echo "$(date +%T) END $label job=$id $(tt-device-mcp status -j $id 2>&1 | grep -E '^(Status|Cause|Runtime|Log):' | tr '\n' ' ' | cut -c1-300)"; return 0; fi
  out=$(tt-device-mcp run-bg "bash tmp/run25.sh $label $*" -w $W -e tmp/ltx25_env.yaml -t 600 2>&1)
  id=$(echo "$out" | grep -oiE 'job[^0-9]*[0-9]+' | grep -oE '[0-9]+' | head -1)
  echo "$(date +%T) SUBMIT $label job=$id :: $(echo $out | tr '\n' ' ' | cut -c1-200)"
  [ -z "$id" ] && return 1
  wait_job $id
  echo "$(date +%T) END $label job=$id $(tt-device-mcp status -j $id 2>&1 | grep -E '^(Status|Cause|Runtime|Log):' | tr '\n' ' ' | cut -c1-300)"
}
run dv145
run dv145_c211 DIFFVAE_NA_CHUNK_BRICKS=2,1,1 DIFFVAE_NA_UNSAFE_CHUNK=1
run dv153 NUM_FRAMES=153 FPS=25
run conv145 LTX25_DIFFVAE=0
echo DRIVE_DONE
