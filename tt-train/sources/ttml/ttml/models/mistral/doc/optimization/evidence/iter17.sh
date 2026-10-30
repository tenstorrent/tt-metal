#!/bin/bash
# Iteration 17 (single-phase RMSNorm backward: rmsnorm_bw_rows, operands resident, partial dots exchanged over the NoC).
# Steps: full harness, TP=4 gradient check vs the single-chip reference, cooled TP=4 full-model MFU + 4-layer TP
# profile, ranked per-op tables.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
unset TT_MESH_GRAPH_DESC_PATH
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== iter17 harness $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/iter17_harness.log 2>&1; echo "harness rc=$?"; grep -E "passed|failed" $O/iter17_harness.log | tail -1
echo "=== iter17 TP=4 gradient check $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/iter17_tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/iter17_tp_grad_check_tp4.log 2>&1; echo "tp4 rc=$?"; grep -E "^tp |PASS|FAIL|Error" $O/iter17_tp_grad_check_tp4.log | head -6
unset TT_MESH_GRAPH_DESC_PATH
COOL=600 $O/tp_measure.sh iter17_tp
CSV=$(ls $O/iter17_tp_tracy_tp4_4layers/reports/*/ops_perf_results*.csv 2>/dev/null | head -1)
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_rank_ops.py "$CSV" --devices 0,2 --top 40 > $O/iter17_tp_ranked_ops.txt 2>&1
[ -n "$CSV" ] && $PY sources/examples/mistral_train/tp_analysis.py "$CSV" > $O/iter17_tp_analysis.txt 2>&1
echo "=== iter17 chain finished $(date -u +%FT%TZ)"
