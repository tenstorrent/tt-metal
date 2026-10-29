#!/bin/bash
# Review round 3 follow-ups: finding 1 (graph-free frozen embedding / norm on a no-grad input) -> TP gradient
# check, TP=4 measurement + 4-layer profile (expect 17/17 collectives, 8+8 norm backward ops); finding 3 -> the
# loss match reruns on HEAD+fixes (8-layer real-weight host-loss vs the torch references, TP=4 real-weight 10 steps).
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
E=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/composite/evidence
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== review3 harness $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/review3_harness.log 2>&1; echo "harness rc=$?"; grep -E "passed|failed" $O/review3_harness.log | tail -1
echo "=== review3 TP=4 gradient check $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/review3_tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/review3_tp_grad_check_tp4.log 2>&1; echo "tp4 rc=$?"; grep -E "^tp |PASS|FAIL|Error" $O/review3_tp_grad_check_tp4.log | head -4
unset TT_MESH_GRAPH_DESC_PATH
echo "=== review3 single-chip Tracy 8 layers (finding 1 single-chip, finding 7d diff) $(date -u +%FT%TZ)"
rm -rf $O/review3_tracy_8layers
timeout 3600 $PY -m tracy -r -p --op-support-count 40000 -o $O/review3_tracy_8layers $TRAIN --layers 8 --steps 4 --warmup-steps 2 --random-init --random-data > $O/review3_tracy_8layers.log 2>&1; echo "tracy 8 rc=$?"
COOL=600 $O/tp_measure.sh review3_tp
echo "=== review3 8 layers real weights, host loss $(date -u +%FT%TZ)"
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 3600 $PY $TRAIN --layers 8 --steps 10 --warmup-steps 2 --host-loss --loss-log $O/review3_loss_8layers_hostloss.json > $O/review3_train_8layers_hostloss.log 2>&1; echo "rc=$?"; grep -E "^step|layers," $O/review3_train_8layers_hostloss.log | tail -2
$PY sources/examples/mistral_train/compare_loss.py --ttml $O/review3_loss_8layers_hostloss.json --torch $E/loss_torch_8layers.json --torch-bf16 $E/loss_torch_8layers_bf16params.json --tol 0.10 > $O/review3_loss_comparison.txt 2>&1; echo "compare rc=$?"; tail -3 $O/review3_loss_comparison.txt
echo "=== review3 TP=4 40 layers real weights $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
sleep 300
timeout 7200 $PY $TRAIN --tp 4 --layers 0 --steps 10 --warmup-steps 2 --loss-log $O/review3_full_tp4_real.json > $O/review3_full_tp4_real.log 2>&1; echo "rc=$?"; grep -E "^step|layers," $O/review3_full_tp4_real.log | tail -12
echo "=== review3 chain finished $(date -u +%FT%TZ)"
