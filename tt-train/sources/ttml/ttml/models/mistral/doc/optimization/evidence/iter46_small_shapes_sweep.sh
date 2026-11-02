#!/bin/bash
# Iteration 46 scoping: 2D config sweep of the four small attention matmuls (TP=4 per-chip shapes) against ttnn's
# default choice, one chip, HiFi4 + fp32 accumulate. qkv fwd [1024,5120]x[1536,5120]^T (80 cores today),
# out fwd [1024,1024]x[5120,1024]^T (88), qkv dX [1024,1536]x[1536,5120] (88), out dX [1024,5120]x[5120,1024] (88).
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
rm -f /dev/shm/sm_segment.* 2>/dev/null
for spec in "1024 5120 1536 --tb" "1024 1024 5120 --tb" "1024 1536 5120" "1024 5120 1024"; do
  set -- $spec; tag=$(echo "$spec" | tr ' ' '_' | tr -d '-')
  echo "=== $spec $(date -u +%T)"
  timeout 1800 $PY $O/iter46_matmul_sweep2d.py $spec > $O/iter46_sweep2d_${tag}.log 2>&1; echo "rc=$?"
  grep -A5 "best 5" $O/iter46_sweep2d_${tag}.log
done
