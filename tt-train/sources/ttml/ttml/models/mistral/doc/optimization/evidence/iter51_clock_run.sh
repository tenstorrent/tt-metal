#!/bin/bash
# Iteration 51 scoping: the delivered clock during the step phase. The MFU command of tp_measure.sh (12 steps instead
# of 8 for a longer step phase) with the 20 Hz pyluwen telemetry sampler alongside; summary per chip afterwards.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
TAG=iter51_clock
COOL=${COOL:-600}
echo "=== $TAG cool-down ${COOL}s $(date -u +%FT%TZ)"; sleep $COOL
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== $TAG TP=4 MFU 12 steps with 20 Hz telemetry $(date -u +%FT%TZ)"
/home/ttuser/.tenstorrent-venv/bin/python $O/iter51_telemetry_sampler.py $O/${TAG}_telemetry.csv 900 20 &
SAMPLER=$!
timeout 3600 $PY $TRAIN --tp 4 --layers 0 --steps 12 --warmup-steps 2 --random-init --random-data --loss-log $O/${TAG}_mfu_tp4.json > $O/${TAG}_mfu_tp4.log 2>&1; echo "rc=$?"
sleep 3; kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
grep -E "layers," $O/${TAG}_mfu_tp4.log | tail -1
python3 $O/iter51_clock_summary.py $O/${TAG}_telemetry.csv $O/${TAG}_mfu_tp4.log
echo "=== $TAG done $(date -u +%FT%TZ)"
