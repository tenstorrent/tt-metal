#!/bin/bash
# Full 40-layer model with TP=4 over the box (user-authorised), tt-smi sampled alongside. Serial device use.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
TAG=${1:-baseline}
rm -f /dev/shm/sm_segment.* 2>/dev/null
sample() {  # $1 = output file; samples tt-smi until the file $2 exists
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
echo "=== TP=4, 40 layers, random init, MFU run $(date -u +%FT%TZ)"
rm -f $O/done.flag; sample $O/full_tp4_${TAG}_random_tt-smi.txt $O/done.flag &
timeout 3600 $PY $TRAIN --tp 4 --layers 0 --steps 8 --warmup-steps 2 --random-init --random-data --loss-log $O/full_tp4_${TAG}_random.json > $O/full_tp4_${TAG}_random.log 2>&1; echo "rc=$?"; touch $O/done.flag; wait
grep -E "^step|layers,|Error|error" $O/full_tp4_${TAG}_random.log | tail -12
echo "=== TP=4, 40 layers, real weights, 10 steps $(date -u +%FT%TZ)"
rm -f $O/done.flag; sample $O/full_tp4_${TAG}_real_tt-smi.txt $O/done.flag &
timeout 7200 $PY $TRAIN --tp 4 --layers 0 --steps 10 --warmup-steps 2 --loss-log $O/full_tp4_${TAG}_real.json --data-log $O/full_tp4_${TAG}_real_data.json --dump-lora-init $O/full_tp4_${TAG}_real_lora_init.npz > $O/full_tp4_${TAG}_real.log 2>&1; echo "rc=$?"; touch $O/done.flag; wait
grep -E "^step|layers,|Loaded|Error|error" $O/full_tp4_${TAG}_real.log | tail -14
echo "=== done $(date -u +%FT%TZ)"
