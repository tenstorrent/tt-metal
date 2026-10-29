#!/bin/bash
# Iteration 7 trial: qkv / out TP-local shapes added to TRANSPOSED_SHAPES so they take the transposed + fused LoRA
# path at TP=4. Gradient check, TP=4 measurement after a cool-down, then the pending narrow TP-shape sweeps.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
MGD=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== iter7 TP=4 gradient check $(date -u +%FT%TZ)"
export TT_MESH_GRAPH_DESC_PATH=$MGD
timeout 1800 $PY sources/examples/mistral_train/tp_gradient_check.py --tp 4 --out $O/iter7_tp_grad_check_tp4.npz --compare $O/tp_grad_check_single.npz > $O/iter7_tp_grad_check_tp4.log 2>&1; echo "tp4 rc=$?"; grep -E "^tp |PASS|FAIL|partial|RMSNorm" $O/iter7_tp_grad_check_tp4.log
unset TT_MESH_GRAPH_DESC_PATH
COOL=600 $O/tp_measure.sh iter7_tp
echo "=== narrow TP-shape sweeps $(date -u +%FT%TZ)"
S=sources/examples/mistral_train/matmul_sweep1d.py
F="--ref-grid 11x10 --in0-block-w 2,3,4 --subs 4x1,2x2 --grids 11x10,10x10 --top 10"
for spec in "w2_fwd_tp:--m 1024 --k 8256 --n 5120" "gateup_dx_tp:--m 5184 --k 16384 --n 1024 --transpose-b" "w2_dx_tp:--m 8256 --k 5120 --n 1024 --transpose-b" "qkv_fwd_tp:--m 1024 --k 5184 --n 1536" "out_fwd_tp:--m 1024 --k 1088 --n 5120"; do
  tag=${spec%%:*}; args=${spec#*:}
  echo "--- $tag $(date -u +%T)"
  timeout 1500 $PY $S $args $F > $O/iter7_matmul_sweep1d_${tag}.log 2>&1; echo "rc=$?"
  grep "ttnn's own" $O/iter7_matmul_sweep1d_${tag}.log | cut -c1-120
  grep " ms " $O/iter7_matmul_sweep1d_${tag}.log | sort -k1 -n | head -2 | cut -c1-150
done
echo "=== iter7 chain finished $(date -u +%FT%TZ)"
