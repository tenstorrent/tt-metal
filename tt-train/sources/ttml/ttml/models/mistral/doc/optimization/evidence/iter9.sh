#!/bin/bash
# Iteration 9 (residual add fused into the all-core RMSNorm forward/backward) + the standalone check of the
# iteration-10 op (skinny_matmul vs ttnn.matmul, correctness and wall time) before it is wired in.
# Steps: skinny_matmul check, full harness, TP=4 gradient check vs the single-chip reference, single-chip 8-layer
# Tracy (attribution), cooled TP=4 full-model MFU + 4-layer TP profile, ranked per-op tables.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
SCR=/tmp/claude-1000/-home-ttuser-agobeljic-tt-metal/0cbc71c0-30d2-4e45-a0c0-f0a2063e85a7/scratchpad
PY=../python_env/bin/python
TRAIN=sources/examples/mistral_train/train.py
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== iter10 skinny_matmul standalone check $(date -u +%FT%TZ)"
cp $SCR/skinny_check.py $O/iter10_skinny_check.py
timeout 900 $PY $O/iter10_skinny_check.py > $O/iter10_skinny_check.log 2>&1; echo "rc=$?"; grep "^M=" $O/iter10_skinny_check.log
echo "=== iter9 harness $(date -u +%FT%TZ)"
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/iter9_harness.log 2>&1; echo "harness rc=$?"; grep -E "passed|failed" $O/iter9_harness.log | tail -1
echo "=== iter9 TP=4 gradient check $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/iter9_tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/iter9_tp_grad_check_tp4.log 2>&1; echo "tp4 rc=$?"; grep -E "^tp |PASS|FAIL|Error|fused into" $O/iter9_tp_grad_check_tp4.log | head -6
unset TT_MESH_GRAPH_DESC_PATH
echo "=== iter9 single-chip Tracy 8 layers $(date -u +%FT%TZ)"
rm -f /dev/shm/sm_segment.* 2>/dev/null
rm -rf $O/iter9_tracy_8layers
timeout 3600 $PY -m tracy -r -p --op-support-count 40000 -o $O/iter9_tracy_8layers $TRAIN --layers 8 --steps 4 --warmup-steps 2 --random-init --random-data > $O/iter9_tracy_8layers.log 2>&1; echo "tracy 8 rc=$?"
CSV=$(ls $O/iter9_tracy_8layers/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0 --top 40 > $O/iter9_tracy_8layers_ranked_ops.txt 2>&1
COOL=600 $O/tp_measure.sh iter9_tp
CSV=$(ls $O/iter9_tp_tracy_tp4_4layers/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 40 > $O/iter9_tp_ranked_ops.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_analysis.py "$CSV" > $O/iter9_tp_analysis.txt 2>&1
echo "=== iter9 chain finished $(date -u +%FT%TZ)"
