#!/bin/bash
# Iteration 50 chain (the qkv dX^T on 110 cores: a program-config change): the same chain as 49.
# check, 8-layer loss matches (one chip; TP=4 against torch), cooled TP=4 MFU + 4-layer profile (cycle diff against
# 48c), 40-layer real weights, harness, SwiGLU op tests.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
E=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/composite/evidence
TAG=iter50
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} TP=4 gradient check $(date -u +%FT%TZ)"
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/${TAG}_tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/${TAG}_tp_grad_check_tp4.log 2>&1; echo "tp4 rc=$?"; grep -E "^tp |PASS|FAIL|Error" $O/${TAG}_tp_grad_check_tp4.log | head -4
echo "=== ${TAG} 8 layers real weights, host loss, TP=4 $(date -u +%FT%TZ)"
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 3600 $PY $TRAIN --tp 4 --layers 8 --steps 10 --warmup-steps 2 --host-loss --loss-log $O/${TAG}_loss_8layers_tp4.json > $O/${TAG}_train_8layers_tp4.log 2>&1; echo "rc=$?"
$PY sources/examples/mistral_train/compare_loss.py --ttml $O/${TAG}_loss_8layers_tp4.json --torch $E/loss_torch_8layers.json --torch-bf16 $E/loss_torch_8layers_bf16params.json --tol 0.10 > $O/${TAG}_loss_comparison_tp4.txt 2>&1; echo "compare rc=$?"; grep -E "max \|diff\||PASS|FAIL" $O/${TAG}_loss_comparison_tp4.txt | head -2
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} 8 layers real weights, host loss, one chip $(date -u +%FT%TZ)"
timeout 3600 $PY $TRAIN --layers 8 --steps 10 --warmup-steps 2 --host-loss --loss-log $O/${TAG}_loss_8layers_hostloss.json > $O/${TAG}_train_8layers_hostloss.log 2>&1; echo "rc=$?"
$PY sources/examples/mistral_train/compare_loss.py --ttml $O/${TAG}_loss_8layers_hostloss.json --torch $E/loss_torch_8layers.json --torch-bf16 $E/loss_torch_8layers_bf16params.json --tol 0.10 > $O/${TAG}_loss_comparison.txt 2>&1; echo "compare rc=$?"; grep -E "max \|diff\||PASS|FAIL" $O/${TAG}_loss_comparison.txt | head -2
COOL=600 $O/tp_measure.sh ${TAG}_tp
CSV=$(ls $O/${TAG}_tp_tracy_tp4_4layers/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 40 > $O/${TAG}_tp_ranked_ops.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 400 > $O/${TAG}_tp_ranked_ops_full.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_analysis.py "$CSV" > $O/${TAG}_tp_analysis.txt 2>&1
python3 $O/cycle_diff.py $O/iter49_tp_ranked_ops_full.txt $O/${TAG}_tp_ranked_ops_full.txt --label "49 -> 50" --min 0.02 > $O/${TAG}_cycle_diff.txt
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} TP=4 40 layers real weights $(date -u +%FT%TZ)"; sleep 300
timeout 7200 $PY $TRAIN --tp 4 --layers 0 --steps 10 --warmup-steps 2 --mem-probe --loss-log $O/${TAG}_full_tp4_real.json > $O/${TAG}_full_tp4_real.log 2>&1; echo "rc=$?"; grep -E "layers," $O/${TAG}_full_tp4_real.log | tail -1
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} harness $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/${TAG}_harness.log 2>&1; echo "harness rc=$?"; grep -E "passed|failed" $O/${TAG}_harness.log | tail -1
timeout 900 $PY -m pytest tests/python/test_swiglu_packed.py -q -p no:cacheprovider > $O/${TAG}_swiglu_op_tests.log 2>&1; echo "swiglu op tests rc=$?"; grep -E "passed|failed" $O/${TAG}_swiglu_op_tests.log | tail -1
echo "=== ${TAG} chain finished $(date -u +%FT%TZ)"
