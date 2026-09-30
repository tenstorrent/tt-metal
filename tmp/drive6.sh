#!/usr/bin/env bash
# Attempt 6 driver: wait for capture job 651, compile its recipes off-device against the t22 tree
# (runs resolve kernels under t22, so the tool's root must be t22 or it skips them as foreign-tree),
# then, once no other project job is queued or running (one-device-job rule), queue ONE real e2e run.
# Writes the job ID to tmp/drive6.jobs.
cd "$(dirname "$0")/.."
W=$PWD
CACHE=/home/smarton/fasth3/tt-metal/tt-project/worktrees/t7/tmp/tt-metal-cache
busy() { tt-device-mcp status 1 | sed -n '/^RUNNING/,/^RECENT/p' | grep -q smarton; }
echo "== waiting for capture job 651 =="
until ! tt-device-mcp status -j 651 | grep -qiE 'Status: +(running|queued|pending)'; do sleep 60; done
tt-device-mcp status -j 651 | head -4
echo "== offline compile (root $W/) =="
env TT_METAL_KERNEL_PREWARM=1 TT_METAL_CACHE=$CACHE TT_METAL_HOME=$W/ $W/build_Release/tools/kernel_prewarm 2>&1 | grep -E "built|foreign|error" | head
echo "== waiting for the project queue to drain =="
while busy; do sleep 60; done
echo "== submit =="
tt-device-mcp run-bg "bash tmp/e2e.sh noisepf" -w $W -t 450 -e tmp/e2e_env.yaml | tee -a tmp/drive6.jobs
echo DRIVE6_SUBMITTED
