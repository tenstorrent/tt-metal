#!/usr/bin/env bash
# Prewarm kernels off-device, then queue one broker job per stage. Job IDs land in tmp/drive.log.
cd "$(dirname "$0")/.."
P=tt_metal/tools/kernel_prewarm/prewarm_and_submit.sh
bash $P -e tmp/env.yaml -t 600 -- "bash tmp/sweep.sh stage_1"
bash $P -e tmp/env.yaml -t 600 -c -- "bash tmp/sweep.sh stage_2"
echo DRIVE_DONE
