#!/bin/bash
# Iteration 47 chain (the measurements ran in iter47_measure.sh / the alternate run): harness, SwiGLU op tests,
# TP=4 gradient check, 8-layer loss match vs torch (single chip: no overlap), TP=4 40 layers real weights with the
# overlap.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
E=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/composite/evidence
TAG=iter47
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} harness $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/${TAG}_harness.log 2>&1; echo "harness rc=$?"; grep -E "passed|failed" $O/${TAG}_harness.log | tail -1
timeout 900 $PY -m pytest tests/python/test_swiglu_packed.py -q -p no:cacheprovider > $O/${TAG}_swiglu_op_tests.log 2>&1; echo "swiglu op tests rc=$?"; grep -E "passed|failed" $O/${TAG}_swiglu_op_tests.log | tail -1
echo "=== ${TAG} TP=4 gradient check $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/${TAG}_tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/${TAG}_tp_grad_check_tp4.log 2>&1; echo "tp4 rc=$?"; grep -E "^tp |PASS|FAIL|Error" $O/${TAG}_tp_grad_check_tp4.log | head -4
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} 8 layers real weights, host loss $(date -u +%FT%TZ)"
timeout 3600 $PY $TRAIN --layers 8 --steps 10 --warmup-steps 2 --host-loss --loss-log $O/${TAG}_loss_8layers_hostloss.json > $O/${TAG}_train_8layers_hostloss.log 2>&1; echo "rc=$?"
$PY sources/examples/mistral_train/compare_loss.py --ttml $O/${TAG}_loss_8layers_hostloss.json --torch $E/loss_torch_8layers.json --torch-bf16 $E/loss_torch_8layers_bf16params.json --tol 0.10 > $O/${TAG}_loss_comparison.txt 2>&1; echo "compare rc=$?"; tail -3 $O/${TAG}_loss_comparison.txt
echo "=== ${TAG} TP=4 40 layers real weights $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
sleep 300
timeout 7200 $PY $TRAIN --tp 4 --layers 0 --steps 10 --warmup-steps 2 --mem-probe --loss-log $O/${TAG}_full_tp4_real.json > $O/${TAG}_full_tp4_real.log 2>&1; echo "rc=$?"; grep -E "^step|layers,|overlap" $O/${TAG}_full_tp4_real.log | tail -13
echo "=== ${TAG} chain finished $(date -u +%FT%TZ)"
