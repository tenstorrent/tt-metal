#!/bin/bash
# Iteration 30 (SDPA forward: the diagonal chunk computes only the K tiles up to the diagonal).
# Same chain as iteration 25.

# Steps: full harness, TP=4 gradient check vs the single-chip reference, cooled TP=4 full-model MFU + 4-layer TP
# profile, ranked per-op tables.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
TAG=${TAG:-iter30}
PY=../python_env/bin/python
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} harness $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/${TAG}_harness.log 2>&1; echo "harness rc=$?"; grep -E "passed|failed" $O/${TAG}_harness.log | tail -1
echo "=== ${TAG} TP=4 gradient check $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/${TAG}_tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/${TAG}_tp_grad_check_tp4.log 2>&1; echo "tp4 rc=$?"; grep -E "^tp |PASS|FAIL|Error" $O/${TAG}_tp_grad_check_tp4.log | head -6
unset TT_MESH_GRAPH_DESC_PATH
COOL=600 $O/tp_measure.sh ${TAG}_tp
CSV=$(ls $O/${TAG}_tp_tracy_tp4_4layers/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 40 > $O/${TAG}_tp_ranked_ops.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_analysis.py "$CSV" > $O/${TAG}_tp_analysis.txt 2>&1
echo "=== ${TAG} MFU chain finished $(date -u +%FT%TZ)"
# Loss match (the matmul config changes the fp32 accumulation order by one bf16 ulp; review round 8 hard-check gap):
# 8-layer host loss vs torch, then TP=4 40 layers real weights.
E=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/composite/evidence
TRAIN=sources/examples/mistral_train/train.py
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== ${TAG} 8 layers real weights, host loss $(date -u +%FT%TZ)"
timeout 3600 $PY $TRAIN --layers 8 --steps 10 --warmup-steps 2 --host-loss --loss-log $O/${TAG}_loss_8layers_hostloss.json > $O/${TAG}_train_8layers_hostloss.log 2>&1; echo "rc=$?"
$PY sources/examples/mistral_train/compare_loss.py --ttml $O/${TAG}_loss_8layers_hostloss.json --torch $E/loss_torch_8layers.json --torch-bf16 $E/loss_torch_8layers_bf16params.json --tol 0.10 > $O/${TAG}_loss_comparison.txt 2>&1; echo "compare rc=$?"; tail -3 $O/${TAG}_loss_comparison.txt
echo "=== ${TAG} TP=4 40 layers real weights $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
sleep 300
timeout 7200 $PY $TRAIN --tp 4 --layers 0 --steps 10 --warmup-steps 2 --loss-log $O/${TAG}_full_tp4_real.json > $O/${TAG}_full_tp4_real.log 2>&1; echo "rc=$?"; grep -E "^step|layers," $O/${TAG}_full_tp4_real.log | tail -12
echo "=== ${TAG} chain finished $(date -u +%FT%TZ)"
