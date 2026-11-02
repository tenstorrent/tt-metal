#!/bin/bash
# Iteration 46 scoping on one chip: the out-projection forward in the transposed orientation (110 cores) and the
# 2D config sweep of the four small attention matmuls.
set -u
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
unset TT_MESH_GRAPH_DESC_PATH
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
PY=../python_env/bin/python
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== wo transposed bench $(date -u +%T)"
timeout 1200 $PY $O/iter46_wo_transposed_bench.py > $O/iter46_wo_transposed_bench.log 2>&1; echo "rc=$?"
grep -A12 "median us" $O/iter46_wo_transposed_bench.log
$O/iter46_small_shapes_sweep.sh
echo "=== scoping finished $(date -u +%T)"
