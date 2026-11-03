#!/bin/bash
# Iteration 48c: the landing chain with the collective overlap off by default: TP gradient check, cooled TP=4 MFU +
# 4-layer profile (cycle diffs against 47 and 46c), the 40-layer real weights, harness, SwiGLU op tests.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
TAG=iter48c
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} TP=4 gradient check $(date -u +%FT%TZ)"
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/${TAG}_tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/${TAG}_tp_grad_check_tp4.log 2>&1; echo "tp4 rc=$?"; grep -E "^tp |PASS|FAIL|Error" $O/${TAG}_tp_grad_check_tp4.log | head -4
unset TT_MESH_GRAPH_DESC_PATH
COOL=600 $O/tp_measure.sh ${TAG}_tp
CSV=$(ls $O/${TAG}_tp_tracy_tp4_4layers/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 40 > $O/${TAG}_tp_ranked_ops.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 400 > $O/${TAG}_tp_ranked_ops_full.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_analysis.py "$CSV" > $O/${TAG}_tp_analysis.txt 2>&1
python3 $O/cycle_diff.py $O/iter47_tp_ranked_ops_full.txt $O/${TAG}_tp_ranked_ops_full.txt --label "47 -> 48c" --min 0.02 > $O/${TAG}_cycle_diff.txt
python3 $O/cycle_diff.py $O/iter46c_tp_ranked_ops_full.txt $O/${TAG}_tp_ranked_ops_full.txt --label "46c -> 48c" --min 0.02 > $O/${TAG}_cycle_diff_vs46c.txt
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} TP=4 40 layers real weights $(date -u +%FT%TZ)"; sleep 300
timeout 7200 $PY $TRAIN --tp 4 --layers 0 --steps 10 --warmup-steps 2 --mem-probe --loss-log $O/${TAG}_full_tp4_real.json > $O/${TAG}_full_tp4_real.log 2>&1; echo "rc=$?"; grep -E "layers,|overlap" $O/${TAG}_full_tp4_real.log | tail -2
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} harness $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/${TAG}_harness.log 2>&1; echo "harness rc=$?"; grep -E "passed|failed" $O/${TAG}_harness.log | tail -1
timeout 900 $PY -m pytest tests/python/test_swiglu_packed.py -q -p no:cacheprovider > $O/${TAG}_swiglu_op_tests.log 2>&1; echo "swiglu op tests rc=$?"; grep -E "passed|failed" $O/${TAG}_swiglu_op_tests.log | tail -1
echo "=== ${TAG} chain finished $(date -u +%FT%TZ)"
