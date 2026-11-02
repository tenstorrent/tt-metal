#!/bin/bash
# Iteration 46c: the TP gradient check and the cooled MFU + 4-layer profile after the dX skip for inputs without a
# gradient (the first profile ran 4 qkv dX matmuls per 4 layers instead of 3).
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== iter46c TP=4 gradient check $(date -u +%FT%TZ)"
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/iter46c_tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/iter46c_tp_grad_check_tp4.log 2>&1; echo "tp4 rc=$?"; grep -E "^tp |PASS|FAIL|Error" $O/iter46c_tp_grad_check_tp4.log | head -4
unset TT_MESH_GRAPH_DESC_PATH
COOL=600 $O/tp_measure.sh iter46c_tp
CSV=$(ls $O/iter46c_tp_tracy_tp4_4layers/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 40 > $O/iter46c_tp_ranked_ops.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 400 > $O/iter46c_tp_ranked_ops_full.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_analysis.py "$CSV" > $O/iter46c_tp_analysis.txt 2>&1
python3 $O/cycle_diff.py $O/iter44_tp_ranked_ops_full.txt $O/iter46c_tp_ranked_ops_full.txt --label "44 -> 46c" --min 0.02 > $O/iter46c_cycle_diff.txt
echo "=== iter46c finished $(date -u +%FT%TZ)"
