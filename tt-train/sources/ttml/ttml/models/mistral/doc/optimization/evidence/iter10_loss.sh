#!/bin/bash
# Loss match on the final path after iterations 9 and 10: 8-layer real-weight host-loss comparison against the torch
# fp64 references (tol 0.10) and the TP=4 40-layer real-weight 10-step run (trajectory vs the baseline run).
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
E=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/composite/evidence
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== iter10 8 layers real weights, host loss $(date -u +%FT%TZ)"
timeout 3600 $PY $TRAIN --layers 8 --steps 10 --warmup-steps 2 --host-loss --loss-log $O/iter10_loss_8layers_hostloss.json > $O/iter10_train_8layers_hostloss.log 2>&1; echo "rc=$?"; grep -E "^step|layers," $O/iter10_train_8layers_hostloss.log | tail -2
$PY sources/examples/mistral_train/compare_loss.py --ttml $O/iter10_loss_8layers_hostloss.json --torch $E/loss_torch_8layers.json --torch-bf16 $E/loss_torch_8layers_bf16params.json --tol 0.10 > $O/iter10_loss_comparison.txt 2>&1; echo "compare rc=$?"; tail -3 $O/iter10_loss_comparison.txt
echo "=== iter10 TP=4 40 layers real weights $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
sleep 300
timeout 7200 $PY $TRAIN --tp 4 --layers 0 --steps 10 --warmup-steps 2 --loss-log $O/iter10_full_tp4_real.json > $O/iter10_full_tp4_real.log 2>&1; echo "rc=$?"; grep -E "^step|layers," $O/iter10_full_tp4_real.log | tail -12
echo "=== iter10 loss chain finished $(date -u +%FT%TZ)"
