#!/bin/bash
# One loop iteration's device evidence, serial on chip 0 (+ the TP=4 full model at the end): harness, Tracy at
# 1/2/4/8 layers, single-chip MFU at 8 layers, TP=4 full-model MFU with tt-smi alongside.
set -u
TAG=$1
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== $TAG harness $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/${TAG}_harness.log 2>&1; echo "harness rc=$?"; grep -E "passed|failed" $O/${TAG}_harness.log | tail -1
echo "=== $TAG tracy 1/2/4/8 $(date -u +%FT%TZ)"
for L in 1 2 4 8; do
  rm -rf $O/${TAG}_tracy_${L}layers
  timeout 3600 $PY -m tracy -r -p --op-support-count 40000 -o $O/${TAG}_tracy_${L}layers $TRAIN --layers $L --steps 4 --warmup-steps 2 --random-init --random-data > $O/${TAG}_tracy_${L}layers.log 2>&1
  echo "tracy $L rc=$?"
done
sample() {  # $1 = output file, $2 = stop flag: one tt-smi line per second for chips 0-3
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
COOL=${COOL:-0}
echo "=== $TAG single-chip MFU, 8 layers (cool-down ${COOL}s first) $(date -u +%FT%TZ)"
sleep $COOL
rm -f $O/${TAG}.flag; sample $O/${TAG}_mfu_8layers_tt-smi.txt $O/${TAG}.flag &
timeout 1800 $PY $TRAIN --layers 8 --steps 10 --warmup-steps 2 --random-init --random-data --loss-log $O/${TAG}_mfu_8layers.json > $O/${TAG}_mfu_8layers.log 2>&1; echo "rc=$?"; touch $O/${TAG}.flag; wait; grep -E "layers" $O/${TAG}_mfu_8layers.log | tail -1
echo "=== $TAG TP=4 full model MFU (cool-down ${COOL}s first) $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
rm -f /dev/shm/sm_segment.* 2>/dev/null
sleep $COOL
rm -f $O/${TAG}.flag; sample $O/${TAG}_mfu_tp4_tt-smi.txt $O/${TAG}.flag &
timeout 3600 $PY $TRAIN --tp 4 --layers 0 --steps 8 --warmup-steps 2 --random-init --random-data --loss-log $O/${TAG}_mfu_tp4.json > $O/${TAG}_mfu_tp4.log 2>&1; echo "rc=$?"; touch $O/${TAG}.flag; wait; grep -E "layers" $O/${TAG}_mfu_tp4.log | tail -1
echo "=== done $(date -u +%FT%TZ)"
