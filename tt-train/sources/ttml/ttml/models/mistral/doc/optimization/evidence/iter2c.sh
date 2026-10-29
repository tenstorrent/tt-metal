#!/bin/bash
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
echo "=== harness at final iteration-2 code $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/iter2c_harness.log 2>&1; echo "rc=$?"; grep -E "passed|failed" $O/iter2c_harness.log | tail -1
echo "=== TP smoke 2 layers $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 1800 $PY sources/examples/mistral_train/train.py --tp 4 --layers 2 --steps 3 --warmup-steps 1 --random-init --random-data > $O/iter2c_tp_smoke.log 2>&1; echo "rc=$?"; grep -E "transposed:|layers,|Error" $O/iter2c_tp_smoke.log | tail -3
COOL=420 $O/tp_measure.sh iter2c_transposed_tp
