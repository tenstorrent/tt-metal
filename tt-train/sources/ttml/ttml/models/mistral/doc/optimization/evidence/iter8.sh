#!/bin/bash
# Iteration 8: frozen vocab-parallel embedding table untilized once (TP only). Gradient check (loss must stay
# identical to the single-chip reference), TP=4 measurement after a cool-down.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== iter8 TP=4 gradient check $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/iter8_tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/iter8_tp_grad_check_tp4.log 2>&1; echo "tp4 rc=$?"; grep -E "^tp |PASS|FAIL|loss|Error" $O/iter8_tp_grad_check_tp4.log | head -6
unset TT_MESH_GRAPH_DESC_PATH
COOL=600 $O/tp_measure.sh iter8_tp
echo "=== iter8 chain finished $(date -u +%FT%TZ)"
