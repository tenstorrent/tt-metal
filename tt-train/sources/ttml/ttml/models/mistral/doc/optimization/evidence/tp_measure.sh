#!/bin/bash
# TP=4 measurement for one tag on cool chips: MFU run (8 steps) and a 4-layer Tracy profile, tt-smi sampled per chip.
set -u
TAG=$1; COOL=${COOL:-0}
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
sample() {
  while [ ! -f "$2" ]; do
    echo "$(date -u +%T) $(timeout 5 tt-smi -s 2>/dev/null | $PY -c "
import sys, json
try:
    devs = json.load(sys.stdin)['device_info']
    print(' '.join(f\"chip{i}: {d['telemetry']['aiclk'].strip()}MHz {d['telemetry']['asic_temperature'].strip()}C {d['telemetry']['power'].strip()}W\" for i, d in enumerate(devs[:4])))
except Exception: print('n/a')")"
    sleep 1
  done > "$1"
}
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== $TAG TP=4 MFU after ${COOL}s cool-down $(date -u +%FT%TZ)"; sleep $COOL
rm -f $O/${TAG}.flag; sample $O/${TAG}_mfu_tp4_tt-smi.txt $O/${TAG}.flag &
timeout 3600 $PY $TRAIN --tp 4 --layers 0 --steps 8 --warmup-steps 2 --random-init --random-data --loss-log $O/${TAG}_mfu_tp4.json > $O/${TAG}_mfu_tp4.log 2>&1; echo "rc=$?"; touch $O/${TAG}.flag; wait; grep -E "layers," $O/${TAG}_mfu_tp4.log | tail -1
echo "=== $TAG TP=4 Tracy 4 layers $(date -u +%FT%TZ)"
rm -rf $O/${TAG}_tracy_tp4_4layers; rm -f $O/${TAG}.flag; sample $O/${TAG}_tracy_tp4_4layers_tt-smi.txt $O/${TAG}.flag &
timeout 3600 $PY -m tracy -r -p --op-support-count 60000 -o $O/${TAG}_tracy_tp4_4layers $TRAIN --tp 4 --layers 4 --steps 4 --warmup-steps 2 --random-init --random-data > $O/${TAG}_tracy_tp4_4layers.log 2>&1; echo "rc=$?"; touch $O/${TAG}.flag; wait; grep -E "layers," $O/${TAG}_tracy_tp4_4layers.log | tail -1
echo "=== done $(date -u +%FT%TZ)"
