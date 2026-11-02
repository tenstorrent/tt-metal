#!/bin/bash
# Iteration 46 scoping, part 2: the attention projections' matmul shapes once they join the fused LoRA path at TP=4
# (K = in + 64 tail): qkv fwd [1024,5184]x[5184,1536], qkv dX [5184,1536]x[1024,1536]^T, out dX plain
# [1088,5120]x[5120,1024]. Waits for part 1 to finish (one chip at a time).
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
until grep -q "scoping finished" $O/iter46_scoping.log; do sleep 15; done
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH
PY=../python_env/bin/python
rm -f /dev/shm/sm_segment.* 2>/dev/null
for spec in "1024 5184 1536" "5184 1536 1024 --tb" "1088 5120 1024"; do
  set -- $spec; tag=$(echo "$spec" | tr ' ' '_' | tr -d '-')
  echo "=== $spec $(date -u +%T)"
  timeout 1800 $PY $O/iter46_matmul_sweep2d.py $spec > $O/iter46_sweep2d_${tag}.log 2>&1; echo "rc=$?"
  grep -A5 "best 5" $O/iter46_sweep2d_${tag}.log
done
echo "=== fused shapes sweep finished $(date -u +%T)"
