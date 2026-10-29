#!/bin/bash
# Iteration 5: batched TP all-reduce of the partial LoRA gradients (no per-layer [T, r] collectives).
# Steps: single-chip harness (the single-chip path is untouched, run for the record), TP=4 gradient check against the
# single-chip reference, TP=4 full-model MFU after a cool-down with per-chip tt-smi, 4-layer TP profile; then narrow
# 1D program-config sweeps of the remaining TP-local projection shapes (the gate_up forward at TP width showed ttnn's
# own choice within 1 % of the best, iter5_matmul_sweep1d_gateup_fwd_tp.log).
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== iter5 harness $(date -u +%FT%TZ)"
timeout 3600 $PY -m pytest tests/python/test_mistral.py -q -p no:cacheprovider > $O/iter5_harness.log 2>&1; echo "harness rc=$?"; grep -E "passed|failed" $O/iter5_harness.log | tail -1
echo "=== iter5 TP=4 gradient check vs single-chip reference $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD
rm -f /dev/shm/sm_segment.* 2>/dev/null
if [ ! -f $O/tp_grad_check_single.npz ]; then
  unset TT_MESH_GRAPH_DESC_PATH
  timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 1 --out $O/tp_grad_check_single.npz > $O/tp_grad_check_single_iter5.log 2>&1; echo "single rc=$?"
  export TT_MESH_GRAPH_DESC_PATH=$MGD
fi
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/iter5_tp_grad_check_tp4_batched.npz --compare $O/tp_grad_check_single.npz > $O/iter5_tp_grad_check_tp4_batched.log 2>&1; echo "tp4 batched rc=$?"; grep -E "^tp |^  |PASS|FAIL|partial LoRA" $O/iter5_tp_grad_check_tp4_batched.log
unset TT_MESH_GRAPH_DESC_PATH
COOL=600 $O/tp_measure.sh iter5_tp
echo "=== iter5 narrow TP-shape sweeps $(date -u +%FT%TZ)"
S=sources/examples/mistral_train/matmul_sweep1d.py
F="--ref-grid 11x10 --in0-block-w 2,3,4 --subs 4x1,2x2 --grids 11x10,10x10 --top 10"
for spec in "w2_fwd_tp:--m 1024 --k 8256 --n 5120" "gateup_dx_tp:--m 5184 --k 16384 --n 1024 --transpose-b" "w2_dx_tp:--m 8256 --k 5120 --n 1024 --transpose-b" "qkv_fwd_tp:--m 1024 --k 5184 --n 1536" "out_fwd_tp:--m 1024 --k 1088 --n 5120"; do
  tag=${spec%%:*}; args=${spec#*:}
  echo "--- $tag $(date -u +%T)"
  timeout 1500 $PY $S $args $F > $O/iter5_matmul_sweep1d_${tag}.log 2>&1; echo "rc=$?"
  grep "ttnn's own" $O/iter5_matmul_sweep1d_${tag}.log | cut -c1-120
  grep " ms " $O/iter5_matmul_sweep1d_${tag}.log | sort -k1 -n | head -2 | cut -c1-150
done
echo "=== done $(date -u +%FT%TZ)"
