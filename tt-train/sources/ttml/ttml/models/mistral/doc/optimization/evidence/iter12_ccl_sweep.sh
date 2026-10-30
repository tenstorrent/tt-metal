#!/bin/bash
# Iteration 12 candidate: CCL parameter sweep at the TP=4 collective shapes (env knobs in ttnn_ops.cpp).
set -u
O=/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/optimization/evidence
echo "=== build $(date -u +%FT%TZ)"
cd /home/ttuser/agobeljic/tt-metal/build_Release && cmake . > /dev/null 2>&1 && timeout 3600 ninja _ttml > $O/iter12_ninja.log 2>&1; echo "build rc=$?"; grep -E "error:" $O/iter12_ninja.log | head -5
cd /home/ttuser/agobeljic/tt-metal/tt-train
export TT_METAL_HOME=/home/ttuser/agobeljic/tt-metal TT_METAL_RUNTIME_ROOT=/home/ttuser/agobeljic/tt-metal
export TT_MESH_GRAPH_DESC_PATH=/home/ttuser/agobeljic/tt-metal/tt-train/configs/mgd/bh_p300x2_1_4_ring_ring.textproto
rm -f /dev/shm/sm_segment.* 2>/dev/null
echo "=== sweep $(date -u +%FT%TZ)"
timeout 1800 ../python_env/bin/python $O/iter12_ccl_sweep.py 2>&1 | grep -E "reduce_scatter|all_gather|Error|Traceback" | grep -v riscv
echo "=== done $(date -u +%FT%TZ)"
