#!/bin/bash
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
TAG=${1:-baseline}
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== TP=4, 40 layers, real weights, 10 steps $(date -u +%FT%TZ)"
timeout 7200 $PY sources/examples/mistral_train/train.py --tp 4 --layers 0 --steps 10 --warmup-steps 2 --loss-log $O/full_tp4_${TAG}_real.json --data-log $O/full_tp4_${TAG}_real_data.json --dump-lora-init $O/full_tp4_${TAG}_real_lora_init.npz > $O/full_tp4_${TAG}_real.log 2>&1; echo "rc=$?"
grep -E "^step|layers,|Loaded|Error" $O/full_tp4_${TAG}_real.log | tail -14
echo "=== done $(date -u +%FT%TZ)"
